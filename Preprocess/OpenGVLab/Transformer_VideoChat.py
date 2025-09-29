#!/usr/bin/env python3
"""
Transformer-based Video Captioning with VideoChat-R1_7B_caption (Qwen-2-VL FT) model.

Model params:
- Model: OpenGVLab/VideoChat-R1_7B_caption
- Dtype: auto
- L40S ~17437MiB 30s

Usage:
    python3 Transformer_VideoChat.py -v <video_path> -o <output_path> [options]
Options:
    -h, --help        Show this help message and exit

    -v, --video-path  Path to input video file or directory (mandatory)
    -o, --output-path Output csv file (mandatory)

    -n, --num         Number of videos to generate (default: 1)
    --repeat          Enable repeating prompts (default: False)
    --workers N       Number of workers for data loading (default: 1)
"""
import argparse
import logging
import os
import random
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import cv2
import pandas as pd
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

import multiprocessing
import sys
import time

# import warnings
# warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# Set decord environment variable to suppress corrupted frame warnings
os.environ["DECORD_DUPLICATE_WARNING_THRESHOLD"] = "1.0"
MODEL_ID = "OpenGVLab/VideoChat-R1_7B_caption"
EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
CSV_FIELDS = ["name", "abs_path", "fps", "frame_cnt", "width", "height", "caption", "objects", ]
MSG_TEMPLETE = lambda video_path, w, h, fps: [{
    "role": "user",
    "content": [
        {"type": "video", "video": video_path, "max_pixels": w * h, "fps": fps, },
        {
            "type": "text",
            "text": (
                "Provide a concise, factual caption of the video, focusing on visible objects, "
                "actions, motions, and the overall scene. Avoid speculation—keep it objective and specific. "
                "Then structure the output as follows:\n"
                "1. Enclose the reasoning process inside <think></think> tags.\n"
                "2. Enclose the final result inside <answer></answer> tags, with:\n"
                "   2.1 A single-sentence scene and story summary inside <summary></summary>.\n"
                "   2.2 No more than 5 key objects, each key object inside a separate <key></key> tag."
            ),
        },
    ],
}]
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", flags=re.IGNORECASE | re.DOTALL)
_KEY_RE = re.compile(r"<key>(.*?)</key>", flags=re.IGNORECASE | re.DOTALL)


class TimeoutError(Exception):
    pass


# keep the old signal-based timeout as a fallback but we won't rely on it
@contextmanager
def timeout(seconds=240):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    try:
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
    except Exception:
        # Signal/alarm may not be available in all environments; ignore and rely on multiprocessing
        old_handler = None

    try:
        yield
    finally:
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass


# =========== Log setup ===========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logging.info("===== Starting Transformers Video Captioning using VideoChat-R1_7B_caption model =====")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video-path", type=str, required=True, help="Path to input video file or directory", )
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to output csv", )
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of videos to generate (default: 1)")
    parser.add_argument("--repeat", action="store_true", help="Re-generate even if output exists (default: False)")
    parser.add_argument("--timeout", type=int, default=240, help="Per-video timeout in seconds (default: 240)")
    # internal flag to run only a single video in the current process (not used by default)
    parser.add_argument("--_single_video", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_ori_videos(args: argparse.Namespace):
    videos_root_path = Path(args.video_path).expanduser()
    if videos_root_path.is_file():
        # Single file input
        if videos_root_path.suffix.lower() in EXTS:
            video_list = [str(videos_root_path)]
        else:
            raise ValueError(f"video-path {args.video_path} is not a video file")
    elif videos_root_path.is_dir():
        # Directory input: collect common video extensions
        video_list = []
        for ext in EXTS:
            video_list.extend(videos_root_path.glob(f"*{ext}"))
        video_list = [str(p) for p in sorted(video_list)]
        if len(video_list) == 0:
            raise ValueError(f"video-path {args.video_path} does not contain any video files")
    else:
        raise ValueError(f"video-path {args.video_path} is neither a file nor a directory")

    # Rand shuffle the list to avoid any ordering bias
    random.shuffle(video_list)
    return video_list


def setup_pipeline():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    return model, processor


def parse_output(output_text: str, video_name: str = "") -> Tuple[str, str]:
    """
    Returns (caption, objects_csv) parsed from the model output.
    Fallbacks:
      - If <answer>...</answer> missing → use whole (de-thinked) text
      - If <summary>...</summary> missing → use the chosen segment as caption
      - Objects from repeated <key>…</key>, de-duplicated (case-insensitive), order preserved
    """
    try:
        text = (output_text or "").strip()

        # Remove any number of <think>...</think> blocks (non-greedy, DOTALL)
        text = _THINK_RE.sub("", text).strip()

        # Focus on the <answer>…</answer> segment if present
        m_answer = _ANSWER_RE.search(text)
        segment = (m_answer.group(1).strip() if m_answer else text)

        # Caption: prefer <summary>…</summary>, else the whole segment
        m_sum = _SUMMARY_RE.search(segment)
        caption = (m_sum.group(1).strip() if m_sum else segment).strip()

        # Objects: all <key>…</key>, cleaned and de-duplicated
        raw_keys = _KEY_RE.findall(segment)
        seen = set()
        keys_clean = []
        for k in raw_keys:
            k_norm = re.sub(r"\s+", " ", (k or "").strip())
            k_key = k_norm.lower()
            if k_norm and k_key not in seen:
                seen.add(k_key)
                keys_clean.append(k_norm)

        objects = ", ".join(keys_clean)

    except Exception as e:
        logging.warning(f"Error parsing output for {video_name}: {e!r}")
        caption = (output_text or "").strip()
        objects = ""

    logging.info(f"Generated caption for {video_name}: {caption[:100]}...")
    return caption, objects


# New: worker that loads model once and handles inference tasks sent via a Queue
def worker_main(task_queue, result_queue):
    """Worker process: loads model+processor then processes tasks.
    Each task is a dict: {"video_path": str, "h": int, "w": int, "fps": int, "frame_cnt": int}
    Result is a dict: {"status": "ok"/"error", "caption": str, "objects": str, "error": str}
    """
    try:
        model, processor = setup_pipeline()
    except Exception as e:
        # Can't proceed; notify and exit
        result_queue.put({"status": "fatal", "error": f"Failed to load model: {e!r}"})
        return

    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            video_path = task["video_path"]
            h = task["h"]
            w = task["w"]
            fps = task["fps"]
            frame_cnt = task["frame_cnt"]

            messages = MSG_TEMPLETE(video_path, w, h, fps)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            caption, objects = parse_output(output_text, os.path.splitext(os.path.basename(video_path))[0])
            result_queue.put({"status": "ok", "caption": caption, "objects": objects})
        except Exception as e:
            # Send back exception info
            result_queue.put({"status": "error", "error": str(e)})


def start_worker():
    ctx = multiprocessing.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    p = ctx.Process(target=worker_main, args=(task_q, result_q), daemon=True)
    p.start()
    return p, task_q, result_q


def caption_videos(video_list, args, model_unused, processor_unused, output_path):
    # We'll offload heavy inference to a dedicated worker process which loads the model once.
    worker_proc, task_q, result_q = start_worker()

    for video_path in video_list:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Check if result already exists and --repeat flag logic
        if output_path.exists() and not args.repeat:
            try:
                existing_df = pd.read_csv(output_path)
                if "abs_path" in existing_df.columns:
                    mask = existing_df["abs_path"].astype(str).eq(str(video_path))
                    if mask.any():
                        logging.info(f"Video {video_name} already processed, skipping (use --repeat to regenerate)")
                        continue
            except Exception as e:
                logging.warning(f"Error reading existing CSV: {e}")

        # Get video info by the first available frame
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        if not success:
            logging.warning(f"Failed to read video {video_path}, skipping.")
            continue
        h, w, _ = image.shape
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        frame_cnt = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vidcap.release()

        logging.info(f"Processing video: {video_name} ({w}x{h}, {fps}fps, {frame_cnt} frames)")

        # Send the task to worker and wait for result with timeout
        try:
            task_q.put({"video_path": video_path, "h": h, "w": w, "fps": fps, "frame_cnt": frame_cnt})

            # Wait for result; if worker dies or timeout happens, handle it
            wait_start = time.time()
            try:
                res = result_q.get(timeout=args.timeout)
            except Exception:
                # Timeout waiting for result
                res = None

            if res is None:
                # Mark as timeout, kill and restart worker to recover from stuck GPU/C calls
                update_csv(video_name, video_path, fps, frame_cnt, w, h, "TIMEOUT", "TIMEOUT", args)
                logging.error(f"Video {video_name} processing timed out after {args.timeout} seconds, skipping")
                try:
                    worker_proc.terminate()
                    worker_proc.join(timeout=5)
                except Exception:
                    pass
                # restart worker for next videos
                worker_proc, task_q, result_q = start_worker()
                continue

            if res.get("status") == "ok":
                caption, objects = res["caption"], res["objects"]
                logging.info(f"Caption {video_name}: {caption[:60]}...")
                logging.info(f"Objects {video_name}: {objects}...")
                update_csv(video_name, video_path, fps, frame_cnt, w, h, caption, objects, args)
            elif res.get("status") == "error":
                logging.error(f"Error processing video {video_name}: {res.get('error')}")
                update_csv(video_name, video_path, fps, frame_cnt, w, h, "ERROR", "ERROR", args)
            elif res.get("status") == "fatal":
                logging.error(f"Worker failed to start/model load error: {res.get('error')}")
                update_csv(video_name, video_path, fps, frame_cnt, w, h, "ERROR", "ERROR", args)
                # try to restart worker
                try:
                    worker_proc.terminate()
                    worker_proc.join(timeout=5)
                except Exception:
                    pass
                worker_proc, task_q, result_q = start_worker()
            else:
                logging.error(f"Unknown response from worker for {video_name}: {res}")
                update_csv(video_name, video_path, fps, frame_cnt, w, h, "ERROR", "ERROR", args)

        except Exception as e:
            logging.error(f"Error processing video {video_name}: {e}")
            update_csv(video_name, video_path, fps, frame_cnt, w, h, "ERROR", "ERROR", args)
            # Ensure worker is alive; restart if not
            try:
                if not worker_proc.is_alive():
                    worker_proc.terminate()
                    worker_proc.join(timeout=5)
                    worker_proc, task_q, result_q = start_worker()
            except Exception:
                pass
            continue

    # shutdown worker
    try:
        task_q.put(None)
        worker_proc.join(timeout=5)
    except Exception:
        try:
            worker_proc.terminate()
            worker_proc.join(timeout=5)
        except Exception:
            pass


def update_csv(name, abs_path, fps, frame_cnt, width, height, caption, objects, args):
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            df = pd.read_csv(output_path)
        except Exception:
            # If the existing file is corrupted, start a fresh DataFrame
            df = pd.DataFrame(columns=CSV_FIELDS)
    else:
        df = pd.DataFrame(columns=CSV_FIELDS)

    # Ensure required columns exist
    for col in CSV_FIELDS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    new_row = {"name": name, "abs_path": abs_path, "fps": fps,
               "frame_cnt": frame_cnt, "width": width, "height": height,
               "caption": caption, "objects": objects, }

    # Upsert by abs_path (or name if abs_path missing)
    mask = df["abs_path"].astype(str).eq(str(abs_path)) if "abs_path" in df.columns else None
    if mask is not None and mask.any():
        # Update existing row only if --repeat is True; otherwise keep old
        if getattr(args, "repeat", False):
            idx = df.index[mask][0]
            for k, v in new_row.items():
                df.at[idx, k] = v
        else:
            # No update; keep existing row
            pass
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    video_list = load_ori_videos(args)
    if len(video_list) == 0:
        raise ValueError(f"No videos found in {args.video_path}")
    if args.num > 0:
        video_list = video_list[:args.num]
    logging.info(f"Found {len(video_list)} videos to process.")

    # The main process will not load the model; worker will handle inference.
    # Provide placeholders for the API compatibility of caption_videos
    model, processor = None, None
    output_path = Path(args.output_path).expanduser()
    if output_path.suffix.lower() != ".csv":
        output_path = output_path / "video_captions.csv"
        # create parent dir if not exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    caption_videos(video_list, args, model, processor, output_path)
