#!/usr/bin/env python3
"""
Transformer-based Video Captioning with VideoChat-R1_7B_caption (Qwen-2-VL FT) model.

Now supports multi-GPU parallelism via `torchrun` (one process per GPU).
Each rank processes a shard of the input list and writes to rank-local CSVs,
then rank 0 merges them into the final output.

Usage (single node):
  torchrun --standalone --nproc_per_node=${NUM_GPUS} Transformer_VideoChat_ddp.py \
      -v <video_or_dir> -o <final_output_csv> -n <num>

If run without torchrun, it behaves like the original single-process version.
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

# ==== NEW: torch / distributed (optional) ====
import torch
import torch.distributed as dist

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("DECORD_DUPLICATE_WARNING_THRESHOLD", "1.0")

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
def setup_logging(rank: int = 0):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    if rank == 0:
        logging.info("===== Starting Transformers Video Captioning (DDP-enabled) =====")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video-path", type=str, required=True, help="Path to input video file or directory", )
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to output csv", )
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of videos to generate (default: 1)")
    parser.add_argument("--repeat", action="store_true", help="Re-generate even if output exists (default: False)")
    parser.add_argument("--timeout", type=int, default=240, help="Per-video timeout in seconds (default: 240)")
    parser.add_argument("--_single_video", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def maybe_init_distributed():
    """Return (is_dist, rank, world_size, local_rank, device)"""
    # torchrun sets these env vars
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    is_dist = local_rank >= 0 and world_size > 1
    if is_dist:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=os.environ.get("TORCH_DDP_BACKEND", "nccl"))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        rank = 0
        world_size = 1
    return is_dist, rank, world_size, local_rank, device


def load_ori_videos(args: argparse.Namespace):
    videos_root_path = Path(args.video_path).expanduser()
    if videos_root_path.is_file():
        if videos_root_path.suffix.lower() in EXTS:
            video_list = [str(videos_root_path)]
        else:
            raise ValueError(f"video-path {args.video_path} is not a video file")
    elif videos_root_path.is_dir():
        video_list = []
        for ext in EXTS:
            video_list.extend(videos_root_path.glob(f"*{ext}"))
        video_list = [str(p) for p in sorted(video_list)]
        if len(video_list) == 0:
            raise ValueError(f"video-path {args.video_path} does not contain any video files")
    else:
        raise ValueError(f"video-path {args.video_path} is neither a file nor a directory")

    random.shuffle(video_list)
    return video_list


def setup_pipeline(device: torch.device):
    # Avoid device_map="auto" under multi-process; instead place the model on the per-rank device.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", attn_implementation="sdpa",
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    return model, processor, device


def parse_output(output_text: str, video_name: str = "") -> Tuple[str, str]:
    try:
        text = (output_text or "").strip()
        text = _THINK_RE.sub("", text).strip()
        m_answer = _ANSWER_RE.search(text)
        segment = (m_answer.group(1).strip() if m_answer else text)
        m_sum = _SUMMARY_RE.search(segment)
        caption = (m_sum.group(1).strip() if m_sum else segment).strip()
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


def worker_main(task_queue, result_queue, device_str):
    """Worker process: loads model+processor then processes tasks.
    Each task: {"video_path": str, "h": int, "w": int, "fps": int, "frame_cnt": int}
    """
    try:
        device = torch.device(device_str)
        model, processor, _ = setup_pipeline(device)
    except Exception as e:
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
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
            ).to(device)

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
            result_queue.put({"status": "error", "error": str(e)})


def start_worker(device: torch.device):
    ctx = multiprocessing.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    p = ctx.Process(target=worker_main, args=(task_q, result_q, str(device)), daemon=True)
    p.start()
    return p, task_q, result_q


def caption_videos(video_list, args, device, rank: int, world_size: int, output_path: Path):
    # Shard the list across ranks (deterministically)
    shard = video_list[rank::world_size]
    logging.info(f"[rank {rank}] will process {len(shard)} / {len(video_list)} videos")

    # Per-rank worker (to keep minimal changes)
    worker_proc, task_q, result_q = start_worker(device)

    # Rank-local CSV path
    rank_csv = output_path.with_suffix(f".rank{rank}.csv")

    for video_path in shard:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if rank_csv.exists() and not args.repeat:
            try:
                existing_df = pd.read_csv(rank_csv)
                if "abs_path" in existing_df.columns:
                    mask = existing_df["abs_path"].astype(str).eq(str(video_path))
                    if mask.any():
                        logging.info(f"[rank {rank}] {video_name} already processed, skipping (--repeat to regen)")
                        continue
            except Exception as e:
                logging.warning(f"[rank {rank}] Error reading existing rank CSV: {e}")

        # Get basic video info
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        if not success:
            logging.warning(f"[rank {rank}] Failed to read video {video_path}, skipping.")
            continue
        h, w, _ = image.shape
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        frame_cnt = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vidcap.release()

        logging.info(f"[rank {rank}] Processing: {video_name} ({w}x{h}, {fps}fps, {frame_cnt} frames)")

        try:
            task_q.put({"video_path": video_path, "h": h, "w": w, "fps": fps, "frame_cnt": frame_cnt})
            try:
                res = result_q.get(timeout=args.timeout)
            except Exception:
                res = None

            if res is None:
                update_csv(rank_csv, video_name, video_path, fps, frame_cnt, w, h, "TIMEOUT", "TIMEOUT", args)
                logging.error(f"[rank {rank}] Timeout after {args.timeout}s: {video_name}")
                try:
                    worker_proc.terminate()
                    worker_proc.join(timeout=5)
                except Exception:
                    pass
                worker_proc, task_q, result_q = start_worker(device)
                continue

            if res.get("status") == "ok":
                caption, objects = res["caption"], res["objects"]
                update_csv(rank_csv, video_name, video_path, fps, frame_cnt, w, h, caption, objects, args)
            elif res.get("status") == "error":
                logging.error(f"[rank {rank}] Error: {video_name}: {res.get('error')}")
                update_csv(rank_csv, video_name, video_path, fps, frame_cnt, w, h, "ERROR", "ERROR", args)
            elif res.get("status") == "fatal":
                logging.error(f"[rank {rank}] Worker start/model load error: {res.get('error')}")
                update_csv(rank_csv, video_name, video_path, fps, frame_cnt, w, h, "ERROR", "ERROR", args)
                try:
                    worker_proc.terminate()
                    worker_proc.join(timeout=5)
                except Exception:
                    pass
                worker_proc, task_q, result_q = start_worker(device)
            else:
                logging.error(f"[rank {rank}] Unknown worker response for {video_name}: {res}")
                update_csv(rank_csv, video_name, video_path, fps, frame_cnt, w, h, "ERROR", "ERROR", args)

        except Exception as e:
            logging.error(f"[rank {rank}] Exception processing {video_name}: {e}")
            update_csv(rank_csv, video_name, video_path, fps, frame_cnt, w, h, "ERROR", "ERROR", args)
            try:
                if not worker_proc.is_alive():
                    worker_proc.terminate()
                    worker_proc.join(timeout=5)
                    worker_proc, task_q, result_q = start_worker(device)
            except Exception:
                pass
            continue

    # Shutdown worker
    try:
        task_q.put(None)
        worker_proc.join(timeout=5)
    except Exception:
        try:
            worker_proc.terminate()
            worker_proc.join(timeout=5)
        except Exception:
            pass

    return rank_csv


def update_csv(csv_path: Path, name, abs_path, fps, frame_cnt, width, height, caption, objects, args):
    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.DataFrame(columns=CSV_FIELDS)
    else:
        df = pd.DataFrame(columns=CSV_FIELDS)

    for col in CSV_FIELDS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    new_row = {"name": name, "abs_path": abs_path, "fps": fps, "frame_cnt": frame_cnt,
               "width": width, "height": height, "caption": caption, "objects": objects}

    mask = df["abs_path"].astype(str).eq(str(abs_path)) if "abs_path" in df.columns else None
    if mask is not None and mask.any():
        if getattr(args, "repeat", False):
            idx = df.index[mask][0]
            for k, v in new_row.items():
                df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)


def merge_rank_csvs(final_csv: Path, rank_csvs):
    # Concatenate then de-duplicate on 'abs_path' (keep last write)
    frames = []
    for p in rank_csvs:
        if Path(p).exists():
            try:
                frames.append(pd.read_csv(p))
            except Exception as e:
                logging.warning(f"[rank0] Skipping unreadable {p}: {e}")
    if not frames:
        logging.warning("[rank0] No rank CSVs to merge; final CSV will not be written.")
        return
    df = pd.concat(frames, ignore_index=True)
    if "abs_path" in df.columns:
        df = df.drop_duplicates(subset=["abs_path"], keep="last")
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(final_csv, index=False)
    logging.info(f"[rank0] Wrote merged CSV: {final_csv}  ({len(df)} rows)")


def main():
    args = parse_args()

    # Distributed init (if any)
    is_dist, rank, world_size, local_rank, device = maybe_init_distributed()
    setup_logging(rank)

    # Deterministic sharding
    random.seed(1234 + rank)

    # Resolve output path (final CSV)
    output_path = Path(args.output_path).expanduser()
    if output_path.is_dir() or output_path.suffix.lower() != ".csv":
        output_path = output_path / "video_captions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    video_list = load_ori_videos(args)
    if len(video_list) == 0:
        raise ValueError(f"No videos found in {args.video_path}")
    if args.num > 0:
        video_list = video_list[:args.num]

    if rank == 0:
        logging.info(f"Found {len(video_list)} videos total; world_size={world_size}")

    # Process shard
    rank_csv = caption_videos(video_list, args, device, rank, world_size, output_path)

    # Sync and merge at the end
    if is_dist:
        dist.barrier()
        # Gather rank CSV paths to rank0
        obj_list = [rank_csv]
        gather_list = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(obj_list[0], gather_list, dst=0)
        if rank == 0:
            merge_rank_csvs(output_path, gather_list)
        dist.barrier()
        dist.destroy_process_group()
    else:
        # Single-process: just ensure final CSV exists at rank_csv path
        if Path(rank_csv) != output_path:
            merge_rank_csvs(output_path, [rank_csv])

if __name__ == "__main__":
    main()