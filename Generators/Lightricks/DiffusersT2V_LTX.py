#!/usr/bin/env python3
"""
Diffusers Text-to-Video Generation using Lightricks 0.9.7 distilled model

Model params:
- Model: Lightricks/LTX-Video-0.9.7-distilled (older version)
- Model: Lightricks/LTX-Video-0.9.8-13B-distilled
- Dtype: bfloat16
- VAE Tiling: Enabled
- L40S MiB ~1min
- A100 MiB ~TODO
- TODO GiB using CPU offload

Video params:
- Frames: 129 (5s at 25 FPS)
- FPS: 25
- Length: 5s
- Guidance Scale: 1.0
- Downscale factor: 2/3 (to fit in memory)
- Height: 720
- Width: 1280

Usage:
    python3 DiffusersT2V_LTX.py -p <prompt_path> -o <output_path> [options]
Options:
    -h, --help        Show this help message and exit

    -p, --prompts Prompt text file directory (mandatory)
    -o, --output-path Output directory for generated videos (mandatory)

    -n, --num         Number of videos to generate (default: 1)
    --repeat          Enable repeating prompts (default: False)
"""
import argparse
import glob
import logging
import os
import random
from pathlib import Path

import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video

# import warnings
# warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# ========== GLOBAL SETTINGS ==========
VIDEO_PARAMS = {
    "num_frames": 129,  # 5s at 25 FPS
    "fps": 25,
    "guidance_scale": 1.0,
    "generator": torch.Generator(device="cuda").manual_seed(42),
    "downscale_factor": 2 / 3,
    "height": 720,
    "width": 1280,
}
MODEL_PARAMS = {
    "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.7-distilled",
    # "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.8-13B-distilled",
    "torch_dtype": torch.bfloat16,
}

# =========== Log setup ===========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logging.info("===== Starting Diffusers Text-to-Video Generation using Mochi 1 Preview model =====")


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
                 f"\t Total prompts found: {len(names)}\n"
                 f"\t Already done: {len(done_names)}\n"
                 f"\t To do: {len(names_todo)}\n"
                 f"\t Selected: {len(names_selected)} for this run")

    return prompts_path, name_to_prompt, names_selected


def setup_pipeline() -> tuple[LTXConditionPipeline, LTXLatentUpsamplePipeline]:
    # ========== Pipline setup  ==========
    logging.info(f"Loading LTXConditionPipeline model {MODEL_PARAMS['pretrained_model_name_or_path']}...")
    pipe = LTXConditionPipeline.from_pretrained(
        MODEL_PARAMS["pretrained_model_name_or_path"],
        torch_dtype=MODEL_PARAMS["torch_dtype"],
        use_safetensors=True,
    )
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        "Lightricks/ltxv-spatial-upscaler-0.9.7",
        vae=pipe.vae,
        torch_dtype=MODEL_PARAMS["torch_dtype"],
        use_safetensors=True,
    )

    logging.info(f"Model loaded. Moving to GPU...")
    # Enable memory savings
    pipe.to("cuda")
    pipe_upsample.to("cuda")
    pipe.vae.enable_tiling()

    return pipe, pipe_upsample


def generate_video(args: argparse.Namespace, pipe: LTXConditionPipeline, pipe_upsample: LTXLatentUpsamplePipeline,
                   prompts_path: Path, name_to_prompt: dict, names_selected: list[str]):
    def round_to_nearest_resolution_acceptable_by_vae(height, width):
        height = height - (height % pipe.vae_spatial_compression_ratio)
        width = width - (width % pipe.vae_spatial_compression_ratio)
        return height, width

    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

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
            with torch.inference_mode(), torch.autocast("cuda",
                                                        dtype=MODEL_PARAMS["torch_dtype"], cache_enabled=False):
                # Part 1. Generate video at smaller resolution
                downscaled_height = int(VIDEO_PARAMS["height"] * VIDEO_PARAMS["downscale_factor"])
                downscaled_width = int(VIDEO_PARAMS["width"] * VIDEO_PARAMS["downscale_factor"])
                downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height,
                                                                                                    downscaled_width)
                latents = pipe(
                    conditions=None,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=downscaled_width,
                    height=downscaled_height,
                    num_frames=VIDEO_PARAMS["num_frames"],
                    num_inference_steps=20,
                    decode_timestep=0.05,
                    guidance_scale=VIDEO_PARAMS["guidance_scale"],
                    decode_noise_scale=0.025,
                    generator=VIDEO_PARAMS["generator"],
                    output_type="latent",
                ).frames

                # Part 2. Upscale generated video using latent upsampler with fewer inference steps
                # The available latent upsampler upscales the height/width by 2x
                upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
                upscaled_latents = pipe_upsample(latents=latents, output_type="latent").frames

                # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
                frames = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=upscaled_width,
                    height=upscaled_height,
                    num_frames=VIDEO_PARAMS["num_frames"],
                    denoise_strength=0.3,  # Effectively, 4 inference steps out of 10
                    num_inference_steps=30,
                    latents=upscaled_latents,
                    decode_timestep=0.05,
                    guidance_scale=VIDEO_PARAMS["guidance_scale"],
                    decode_noise_scale=0.025,
                    image_cond_noise_scale=0.025,
                    generator=VIDEO_PARAMS["generator"],
                    output_type="pil",
                ).frames[0]
                # Part 4. Downscale the video to the expected resolution
                frames = [frame.resize((VIDEO_PARAMS["width"], VIDEO_PARAMS["height"])) for frame in frames]
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
    pipe, pipe_upsample = setup_pipeline()

    logging.info("========= Starting generation... =========")
    generate_video(args, pipe, pipe_upsample, prompts_path, name_to_prompt, names_selected)
    logging.info("========= All done! =========")
