#!/usr/bin/env python3
"""
Diffusers Video-to-Video Extrapolation using CosmosPredict 2 model.

Model params:
- Model: nvidia/Cosmos-Predict2-14B-Video2World
- Dtype: bfloat16
- VAE Tiling: enabled
- L40S * 2 : 27891MiB + 39257MiB
- A100 OOM
- H100 : 95095MiB
===== 2B small model =====
- Model: nvidia/Cosmos-Predict2-2B-Video2World
- L40S : 43393MiB ~
- H100 : 61905MiB ~1h40min

Video params (new extrapolated clip):
- Frames: 129 (with 40 input condition frames and 89 generated frames)
- FPS: 25
- Length: 5s
- Guidance Scale: 7.0
- Inference Steps: 50

Usage:
    python3 DiffusersV2V_CosmosPredict2.py -v <video_path> -o <output_path> [options]
Options:
    -h, --help        Show this help message and exit

    -v, --video-path  Path to input video file or directory (mandatory)
    -o, --output-path Output directory for generated videos (mandatory)

    -n, --num         Number of videos to generate (default: 1)
    --repeat          Re-generate even if output exists (default: False)
"""
import argparse
import glob
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from diffusers import Cosmos2VideoToWorldPipeline
from diffusers.utils import export_to_video, load_video
from torch import nn
import torch.distributed as dist
import os

TensorOrImages = Union[torch.Tensor, List["Image.Image"]]

# import warnings
# warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ========== GLOBAL SETTINGS ==========
# We have originally unknown frames of video,
# We have 129 / 25fps videos other path
# 50 last frame for condition
# 129 frames generated (including condition)
# Replace 50 pre-condition frames with original clip
COND_VIDEO_FRAMES = 40  # number of input frames to condition on
VIDEO_PARAMS = {
    "num_frames": 129,
    "fps": 25,
    "num_inference_steps": 35,
    "guidance_scale": 7.0,
    "generator": torch.Generator(device="cuda").manual_seed(42),
    "seed": 42,
    "height": 704,
    "width": 1280,
}
MODEL_PARAMS = {
    # Available checkpoints: nvidia/Cosmos-Predict2-2B-Video2World, nvidia/Cosmos-Predict2-14B-Video2World
    # "pretrained_model_name_or_path": "nvidia/Cosmos-Predict2-2B-Video2World",
    "pretrained_model_name_or_path": "nvidia/Cosmos-Predict2-14B-Video2World",
    "torch_dtype": torch.bfloat16,
}

# =========== Log setup ===========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logging.info("===== Starting Diffusers Video-to-Video Generation using CosmosPredict 2 model =====")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video-path", type=str, required=True, help="Path to input video file or directory", )
    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Path to output directory for generated videos", )
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of videos to generate (default: 1)")
    parser.add_argument("--repeat", action="store_true", help="Re-generate even if output exists (default: False)")
    return parser.parse_args()


def load_ori_videos(args: argparse.Namespace):
    """
    Load ORIGINAL VIDEOS list. Supports:
      - Single file video
      - Directory of videos
      - CSV file with columns (preferred): name, abs_path, caption
        (abs_path required; caption optional). If 'name' missing, use stem of abs_path.
    Returns:
      videos_root_path: Path
      name_to_video_path: dict[name -> abs path]
      names_selected: list[str]
      name_to_caption: dict[name -> caption] (may be empty)
    """
    videos_root_path = Path(args.video_path).expanduser()
    name_to_caption = {}

    if videos_root_path.is_file() and videos_root_path.suffix.lower() == ".csv":
        df = pd.read_csv(videos_root_path)
        # Normalise likely column names
        col_path = None
        for cand in ["abs_path", "path", "video", "video_path"]:
            if cand in df.columns:
                col_path = cand
                break
        if col_path is None:
            raise ValueError("CSV must include a column 'abs_path' (or 'path'/'video'/'video_path').")
        has_name = "name" in df.columns
        has_caption = "caption" in df.columns

        name_to_video_path = {}
        for _, row in df.iterrows():
            # Coerce the table value to str first to avoid pandas/numpy type issues
            p = str(Path(str(row[col_path])).expanduser())
            if not os.path.isfile(p):
                logging.warning(f"CSV path does not exist, skipping: {p}")
                continue
            name = str(row["name"]) if has_name and pd.notna(row["name"]) else Path(p).stem
            name_to_video_path[name] = p
            if has_caption and pd.notna(row["caption"]):
                name_to_caption[name] = str(row["caption"])

        names = list(name_to_video_path.keys())

    elif videos_root_path.is_file():
        # Single raw video file
        if videos_root_path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            raise ValueError(f"Unsupported video file type: {videos_root_path.suffix}")
        names = [videos_root_path.stem]
        name_to_video_path = {videos_root_path.stem: str(videos_root_path)}

    elif videos_root_path.is_dir():
        # Directory of videos
        exts = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")
        files = []
        for ptn in exts:
            files.extend(sorted(glob.glob(os.path.join(videos_root_path, ptn))))
        if not files:
            raise ValueError(f"No video files found under directory: {videos_root_path}")
        names = [Path(p).stem for p in files]
        name_to_video_path = {Path(p).stem: p for p in files}
    else:
        raise ValueError(f"video-path {videos_root_path} is neither a file, a directory, nor a CSV")

    # Prepare output dir and filter out already-generated results (unless --repeat)
    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    done_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(output_path)
        if f.endswith(".mp4") and os.path.getsize(os.path.join(output_path, f)) > 0
    ]
    names_todo = [n for n in names if args.repeat or n not in done_names]

    if len(names_todo) < args.num:
        logging.warning(f"Only {len(names_todo)} videos available to generate, less than requested {args.num}")
        names_selected = names_todo
    else:
        names_selected = random.sample(names_todo, args.num)

    logging.info(
        f"Under {args.output_path},\n"
        f"Video source {videos_root_path}:\n"
        f"\t Total videos found: {len(names)}\n"
        f"\t Already done: {len(done_names)}\n"
        f"\t To do: {len(names_todo)}\n"
        f"\t Selected: {len(names_selected)} for this run"
    )

    return videos_root_path, name_to_video_path, names_selected, name_to_caption


class CosmosSafetySkipper(nn.Module):
    """
    No-op safety checker compatible with Cosmos2VideoToWorldPipeline.
    - Provides .dtype/.device via a registered buffer so .to(device, dtype) works.
    - Implements check_text_safety(), check_image_safety(), and forward().
    - Always passes content through unchanged; reports 'blocked=False'.
    """

    def __init__(self, log: bool = True) -> None:
        super().__init__()
        self.log = log
        self.register_buffer("_compat_buf", torch.zeros((), dtype=torch.float32), persistent=False)

    @property
    def dtype(self):
        return self._compat_buf.dtype

    @property
    def device(self):
        return self._compat_buf.device

    # ---- Text safety hook expected by the Cosmos pipeline ----
    @torch.no_grad()
    def check_text_safety(self, text: Union[str, List[str]], **kwargs: Any) -> bool:
        # Return True => safe. Accept both str and list[str].
        return True

    # ---- Image safety hook (diffusers style) ----
    @torch.no_grad()
    def check_image_safety(
            self,
            images: TensorOrImages,
            **kwargs: Any
    ) -> Tuple[TensorOrImages, List[Dict[str, Any]]]:
        if isinstance(images, torch.Tensor):
            b = images.shape[0] if images.ndim >= 4 else 1
        else:
            b = len(images)
        reports = [dict(blocked=False, reason=None, score=0.0) for _ in range(b)]
        if self.log:
            print(f"[CosmosSafetySkipper] image safety skipped, batch={b}")
        return images, reports

    @torch.no_grad()
    def check_video_safety(self, video, **kwargs):
        """
        No-op video hook expected by Cosmos2VideoToWorldPipeline.
        Must return the same type that the pipeline passed in (e.g., list[PIL.Image]
        or a Tensor of frames). We just pass it through unchanged.
        """
        if self.log:
            try:
                # best-effort shape/len logging without assuming a type
                if hasattr(video, "shape"):
                    print(f"[CosmosSafetySkipper] video safety skipped, shape={tuple(video.shape)}")
                else:
                    print(f"[CosmosSafetySkipper] video safety skipped, len={len(video)}")
            except Exception:
                print("[CosmosSafetySkipper] video safety skipped.")
        return video

    # ---- Make it also behave like the usual diffusers safety checker ----
    @torch.no_grad()
    def forward(self, images: TensorOrImages, **kwargs: Any):
        return self.check_image_safety(images, **kwargs)


def setup_pipeline() -> Cosmos2VideoToWorldPipeline:
    # ========== Pipeline setup  ==========
    logging.info(f"Loading Cosmos2VideoToWorldPipeline model {MODEL_PARAMS['pretrained_model_name_or_path']}...")
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        MODEL_PARAMS["pretrained_model_name_or_path"],
        # revision=args.revision,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        # device_map="balanced",
        safety_checker=CosmosSafetySkipper().eval()
    )
    logging.info("Model loaded. Enabling memory-savers and moving to GPU/CPU offload...")
    # pipe.to("cuda")
    # pipe.vae.enable_tiling()
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    return pipe


def generate_video(
        args: argparse.Namespace,
        pipe: Cosmos2VideoToWorldPipeline,
        videos_root_path: Path,
        name_to_video_path: dict,
        names_selected: list[str],
        name_to_caption: dict | None = None,
):
    # ========== Generation and export ==========
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logging.info(f"Running rank {rank}/{world_size} on device cuda:{rank}")

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    VIDEO_PARAMS["generator"] = torch.Generator(device=device).manual_seed(VIDEO_PARAMS["seed"])
    # device = "cuda"
    pipe.to(device)

    # scatter work
    names_local = names_selected[rank::world_size]
    total = len(names_local)
    logging.info(f"Rank {rank}: processing {total} videos")

    # total = len(names_selected)
    base_prompt = (
        "Continue the input video realistically and smoothly, following its objects' features, motion and scene. "
        "Avoid pixelization, blurriness, flicker, ghosting, compression artifacts, or low resolution. "
        "Keep the same style, colour tone, and quality as the input video."
    )
    # prompt = "Continue the input video realistically and smoothly, following its objects' feature, motion and scene. No any pixelization, blurriness, flicker, ghosting, compression artifacts, low resolution. Keep the same style, color tone and quality as the input video."
    negative_prompt = "jitter, flicker, sudden scene jump, heavy blur, ghosting, compression artifacts, low resolution, pixelization on any non-sexual content, pixelization on faces"

    # for idx, name in enumerate(names_selected):
    for idx, name in enumerate(names_local):
        logging.info(f"Progress: {idx + 1}/{total} ({name})")

        # Load the input video frames
        video_path = name_to_video_path[name]
        try:
            input_frames = load_video(video_path)[-COND_VIDEO_FRAMES:]
        except Exception as e:
            logging.exception(f"Failed to load video {video_path}: {e}")
            continue

        try:
            # Making sure all the pip is on GPU
            pipe.to(device)
            # Build prompt (if CSV used and caption available, prepend it)
            caption_text = ""
            if name_to_caption and name in name_to_caption:
                cap = str(name_to_caption.get(name, "")).strip()
                if cap!= "TIMEOUT" and cap != "nan":
                    caption_text = cap
            prompt = f"{caption_text}. {base_prompt}" if caption_text else base_prompt

            # Run generation
            with torch.autocast("cuda", dtype=MODEL_PARAMS["torch_dtype"], cache_enabled=False):
                output = pipe(
                    image=None,
                    video=input_frames,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    generator=VIDEO_PARAMS["generator"],
                    num_inference_steps=VIDEO_PARAMS["num_inference_steps"],
                    guidance_scale=VIDEO_PARAMS["guidance_scale"],
                    height=VIDEO_PARAMS["height"],
                    width=VIDEO_PARAMS["width"],
                    num_frames=VIDEO_PARAMS["num_frames"],
                    fps=VIDEO_PARAMS["fps"],
                ).frames[0]
        except Exception as e:
            logging.exception(f"Generation failed for {name}: {e}")
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()

        # ========== Post-processing and export ==========
        # The pipeline returns a sequence of frames of length VIDEO_PARAMS['num_frames'] (including the condition frames).
        # We want the final exported clip to be that generated clip but with the first COND_VIDEO_FRAMES frames replaced
        # by the corresponding original frames (to avoid artifacts on conditioned frames).
        logging.info(
            f"Generation finished: conditioned frames: {COND_VIDEO_FRAMES}, generated frames: {len(output)}")
        from typing import List
        import PIL.Image
        # Pillow compatibility: pick a resampling filter that exists across versions.
        # Use getattr only to avoid static-analysis issues when some constants aren't present.
        Resampling = getattr(PIL.Image, "Resampling", None)
        if Resampling is not None:
            resample_filter = getattr(Resampling, "LANCZOS", getattr(Resampling, "BICUBIC", 3))
        else:
            # Fallback to module-level constants if present, else numeric BICUBIC (3)
            resample_filter = getattr(PIL.Image, "LANCZOS", getattr(PIL.Image, "BICUBIC", 3))

        # Load original video frames (full)
        ori_video: List[PIL.Image.Image] = load_video(video_path)

        # Ensure PIL Image size matches generation target; resize original if needed
        target_size = (VIDEO_PARAMS["width"], VIDEO_PARAMS["height"])
        if len(ori_video) == 0:
            logging.warning(f"Original video {video_path} contains no frames; skipping post-processing.")
            continue
        if hasattr(ori_video[0], "size") and ori_video[0].size != target_size:
            logging.info(f"Resizing original video frames from {ori_video[0].size} to {target_size}")
            ori_video = [img.resize(target_size, resample_filter) for img in ori_video]

        # Ensure output is a mutable list of PIL Images
        generated: List[PIL.Image.Image] = list(output)

        gen_len = len(generated)
        if gen_len < 1:
            logging.warning(f"Generated video for {name} has no frames, skipping.")
            continue

        # Determine how many condition frames we can actually replace (can't exceed either list length)
        cond_replace = min(COND_VIDEO_FRAMES, gen_len, len(ori_video))

        if cond_replace <= 0:
            logging.warning(
                f"No frames available to replace conditioned portion for {name}; exporting generated frames as-is.")
            final_frames = generated
        else:
            # Use the last cond_replace frames from original video to overwrite the first cond_replace frames of generated
            orig_tail = ori_video[-cond_replace:]
            # If orig_tail aligns with generated positions, simply assign
            for i in range(cond_replace):
                # Replace generated frame i with the corresponding original tail frame
                generated[i] = orig_tail[i]

            # Final frames are the modified generated frames (length = gen_len, typically VIDEO_PARAMS['num_frames'])
            final_frames = generated

        logging.info(f"Final clip for {name}: {len(final_frames)} frames (cond replaced: {cond_replace}).")

        # export
        save_path = Path(args.output_path).expanduser() / f"{name}.mp4"
        try:
            export_to_video(final_frames, str(save_path), fps=VIDEO_PARAMS["fps"])
            logging.info(f"{save_path} saved.")
        except Exception as e:
            logging.exception(f"Failed to export video for {name}: {e}")


if __name__ == "__main__":
    args = parse_args()
    videos_root_path, name_to_video_path, names_selected, name_to_caption = load_ori_videos(args)
    pipe = setup_pipeline()

    logging.info("========= Starting generation... =========")
    generate_video(args, pipe, videos_root_path, name_to_video_path, names_selected, name_to_caption)
    logging.info("========= All done! =========")
