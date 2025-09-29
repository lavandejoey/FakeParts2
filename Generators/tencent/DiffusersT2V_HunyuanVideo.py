#!/usr/bin/env python3
"""
Diffusers Text-to-Video Generation using HunyuanVideo model

Model params:
- Model: hunyuanvideo-community/HunyuanVideo
- Dtype: float16
- VAE Tiling: Enabled
- H100 ~1h20min
- L40S ~1h30min
- A100 ----
- 46GiB using seq CPU offload

Video params:
- Frames: 129 (5s at 25 FPS)
- FPS: 25
- Length: 5s

Usage:
    python3 DiffusersT2V_HunyuanVideo.py -p <prompt_path> -o <output_path> [options]
Options:
    -h, --help        Show this help message and exit

    -p, --prompts Prompt text file directory (mandatory)
    -o, --output-path Output directory for generated videos (mandatory)

    -n, --num         Number of videos to generate (default: 1)
    --repeat          Enable repeating prompts (default: False)
    --workers N       Number of workers for data loading (default: 1)
"""
import argparse
import glob
import logging
import os
import random
from pathlib import Path
from pathlib import Path
from multiprocessing import Process, Queue, set_start_method

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

# import warnings
# warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ========== GLOBAL SETTINGS ==========
VIDEO_PARAMS = {
    "num_frames": 129,  # 5s at 25 FPS
    "fps": 25,
    "num_inference_steps": 50,
    "generator": torch.Generator(device="cuda").manual_seed(42),
}
MODEL_PARAMS = {
    "pretrained_model_name_or_path": "hunyuanvideo-community/HunyuanVideo",
    "torch_dtype": torch.float16,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logging.info("===== Starting Diffusers Text-to-Video Generation using HunyuanVideo model =====")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompts", type=str, required=True, help="Path to data directory containing prompts", )
    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Path to output directory for generated videos", )
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of videos to generate (default: 1)")
    parser.add_argument("--repeat", action="store_true", help="Enable repeat using prompts (default: False)")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace):
    # ========== Prompt preparation ==========
    # args.prompts can be either:
    #  - a single text file with one prompt per line, or
    #  - a directory containing .txt files (each file is a prompt; filename used as output name)
    prompts_path = Path(args.prompts).expanduser()
    prompts_list = []
    if prompts_path.is_file():
        # Interpret the provided path as a file containing prompt texts (one per line)
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts_list = [line.strip() for line in f if line.strip()]
        # generate safe output names for each prompt (prompt_1, prompt_2, ...)
        names = [f"prompt_{i + 1}" for i in range(len(prompts_list))]
        prompts_from_files = False
    elif prompts_path.is_dir():
        all_txt = sorted(glob.glob(os.path.join(prompts_path, "*.txt")))
        names = [os.path.splitext(os.path.basename(p))[0] for p in all_txt]
        prompts_from_files = True
    else:
        raise ValueError(f"Prompts path {args.prompts} is neither a file nor a directory")

    # If prompts were provided in a single file, build a mapping from generated name -> prompt text
    name_to_prompt = dict(zip(names, prompts_list)) if not prompts_from_files else {}

    # Filter out used prompts
    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    done_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(output_path)
        if f.endswith(".mp4") and os.path.getsize(os.path.join(output_path, f)) > 0
    ]
    names_todo = [n for n in names if args.repeat or n not in done_names]
    if len(names_todo) < args.num:
        logging.warning(f"Only {len(names_todo)} prompts available to generate, less than requested {args.num}")
        names_selected = names_todo
    else:
        names_selected = random.sample(names_todo, args.num)
    logging.info(f"Under {args.output_path},\n"
                 f"Prompt {prompts_path}:\n"
                 f"\t Total prompts found: {len(names)}\n"
                 f"\t Already done: {len(done_names)}\n"
                 f"\t To do: {len(names_todo)}\n"
                 f"\t Selected: {len(names_selected)} for this run")

    return prompts_path, name_to_prompt, names_selected


def setup_pipeline() -> HunyuanVideoPipeline:
    # ========== Pipline setup  ==========
    logging.info(f"Loading HunyuanVideoPipeline model {MODEL_PARAMS['pretrained_model_name_or_path']}...")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        MODEL_PARAMS["pretrained_model_name_or_path"],
        subfolder="transformer",
        torch_dtype=MODEL_PARAMS["torch_dtype"],
        use_safetensors=True,
        device_map="balanced",
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        MODEL_PARAMS["pretrained_model_name_or_path"],
        transformer=transformer,
        torch_dtype=MODEL_PARAMS["torch_dtype"],
        use_safetensors=True,
        device_map="balanced",
    )
    logging.info(f"Model loaded. Moving to GPU...")
    # Enable memory savings
    pipe.enable_vae_tiling()
    # pipe.enable_model_cpu_offload()

    return pipe


def generate_video(args: argparse.Namespace, pipe: HunyuanVideoPipeline,
                   prompts_path: Path, name_to_prompt: dict, names_selected: list[str]):
    # ========== Generation and export ==========
    total = len(names_selected)
    for idx, name in enumerate(names_selected):
        logging.info(f"Progress: {idx + 1}/{total} ({name})")
        # read prompt (either from mapping or from a .txt file in the prompts directory)
        if name in name_to_prompt:
            prompt = name_to_prompt[name]
        else:
            with open(os.path.join(prompts_path, f"{name}.txt"), "r", encoding="utf-8") as f:
                prompt = f.read().strip()

        # Run generation with autocast using the same dtype as the model
        try:
            with torch.inference_mode(), torch.autocast(
                    "cuda",
                    dtype=MODEL_PARAMS["torch_dtype"],
                    cache_enabled=False):
                frames = pipe(
                    prompt,
                    num_inference_steps=VIDEO_PARAMS["num_inference_steps"],
                    num_frames=VIDEO_PARAMS["num_frames"],
                    generator=VIDEO_PARAMS["generator"],
                ).frames[0]
        except Exception as e:
            logging.exception(f"Generation failed for {name}: {e}")
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()

        # export
        save_path = Path(args.output_path).expanduser() / f"{name}.mp4"
        export_to_video(frames, str(save_path), fps=VIDEO_PARAMS["fps"])
        logging.info(f"{save_path} saved.")


if __name__ == "__main__":
    args = parse_args()
    prompts_path, name_to_prompt, names_selected = load_prompts(args)
    pipe = setup_pipeline()

    logging.info("========= Starting generation... =========")
    generate_video(args, pipe, prompts_path, name_to_prompt, names_selected)
    logging.info("========= All done! =========")
