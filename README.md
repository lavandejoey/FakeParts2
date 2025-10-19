# FakeParts2

## Environment Setup

- Clone the complete repository
    ```bash
    git clone --recurse-submodules https://github.com/lavandejoey/FakeParts2.git
    cd FakeParts2
    ```
- Create a conda environment and install dependencies
    ```bash
    conda env create -f environment.yaml --name fakeparts2
    conda activate fakeparts2
    pip install -r requirements.txt
    ```

## Video Generation

### T2V Models and Prompts

| Model Name                           | Num  | Cost                                                                   | Frames / Length | Resolution | Link                                                                           |
|--------------------------------------|------|------------------------------------------------------------------------|-----------------|------------|--------------------------------------------------------------------------------|
| hunyuanvideo-community/HunyuanVideo  | 1581 | L40S ~1h30min;<br> H100 ~1h20min                                       | 129f / 5s       | 1280x720   | [🤗 Hugging Face](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)  |
| genmo/mochi-1-preview                | 567  | L40S 40459MiB ~13min;<br> A100 39019MiB ~13min                         | 129f / 5s       | 848x480    | [🤗 Hugging Face](https://huggingface.co/genmo/mochi-1-preview)                |
| THUDM/CogVideoX-5b                   | 1722 | L40S 11263MiB ~22min;<br> 3096 11263MiB ~6min;<br> A100 11263MiB ~4min | 129f / 5s       | 720x480    | [🤗 Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b)                   |
| Lightricks/LTX-Video-0.9.7-distilled | 2606 | L40S 43781MiB ~2min                                                    | 129f / 5s       | 1280x720   | [🤗 Hugging Face](https://huggingface.co/Lightricks/LTX-Video-0.9.7-distilled) |
|                                      | 6476 |                                                                        |                 |            |                                                                                |||
 

## Benchmarking

# Data Preparation

HF repository upload folder: `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition`

Dataset structure

Relative paths has 2 patterns, SUBSET in `{fake_frames, fake_videos, real_frames, real_videos}`:

```text
<TASK>/<METHOD>/<SUBSET>/
<TASK>/<METHOD>/<S_METHOD>/<SUBSET>/
```

After that, the files are organized as:

```text
<v_name>/frame_<%06d>.jpg 
<v_name>.mp4
```

### Soft Links for Data

Data Physically stored in

- `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only/` for frames' folders.
- `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_generation_dectection/` for videos.
- Soft-linking: `FakeParts_data_addition_videos_only` -- to --> `FakeParts_data_addition_frames_generation_dectection`
  for videos.
- Soft-linking: `FakeParts_data_addition_frames_0_1_labels` -- to -->
  `FakeParts_data_addition_frames_only/{0_real,1_fake}` for frames with 0/1 labels.

# Detection Methods Execution
## Conventional DNN Methods
### [lavandejoey/AIGVDet](https://github.com/lavandejoey/AIGVDet)
- `conda activate fakevlm310`
-
### DeMamba: WHERE IS MODEL?

## CLIP-Based Methods

### [FatFormer]()

### [C2P](https://github.com/Triocrossing/c2p_FakeParts)
- `conda activate fakevlm310`
- `bash Detectors/c2p_FakeParts/C2PEval.sh`
- DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
- $\rightarrow$ `./results/{date_time}_C2P-CLIP/predictions.csv`

### [lavandejoey/UniversalFakeDetect](https://github.com/lavandejoey/UniversalFakeDetect)
- `conda activate fakevlm310`
- `bash Detectors/UniversalFakeDetect/UniFakeDetEval.sh`
- DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
- $\rightarrow$ `./results/{date_time}_clip_vitl14/predictions.csv`

### [lavandejoey/De-Fake](https://github.com/lavandejoey/De-Fake)
- Download ckpt first:
  - [clip_linear.pt](https://drive.google.com/file/d/1qI7x5iodaCFq0S61LKw4wWjql7cYou_4/view?usp=sharing)
  - [finetune_clip.pt](https://drive.google.com/file/d/1SuenxJP10VwArC6zW0SHMUGObMRqQhBD/view?usp=sharing)
- `conda activate defake`
- `bash Detectors/De-Fake/DeFakeEval.sh`
- DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
- $\rightarrow$ `./results/{date_time}_DeFake_ViTB32/predictions.csv`

### [lavandejoey/D3](https://github.com/lavandejoey/D3)
- `conda activate fakevlm310`
- `bash Detectors/D3/D3Eval.sh`
- DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
- $\rightarrow$ `./results/{date_time}_DeFake_ViTB32/predictions.csv`

## VLM Based Methods

### [lavandejoey/FakeVLM](https://github.com/lavandejoey/FakeVLM)
- `conda activate fakevlm310`
- `bash Detectors/FakeVLM/FakeVLMEval.sh`
- DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
- $\rightarrow$ `./results/{date_time}_fakeVLM/predictions.csv`

### [lavandejoey/SIDA](https://github.com/lavandejoey/SIDA)
- `conda activate sida311`
- `bash Detectors/SIDA/SIDAEval.sh`
- DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
- $\rightarrow$ `./results/{date_time}_SIDA-13B_eval/predictions.csv`

### [lavandejoey/AntifakePrompt](https://github.com/lavandejoey/AntifakePrompt)
- Download ckpt with [download_checkpoints.sh](Detectors/AntifakePrompt/ckpt/download_checkpoints.sh):
  - [COCO_150k_SD3_SD2IP.pth](https://drive.google.com/file/d/1EUnVG4OZZPXeOyWaa5P590yCKGH-nunQ/view?usp=drive_link)
  - [COCO_150k_SD3_SD2IP_lama.pth](https://drive.google.com/file/d/1qnZfCknNHgC-Nhlwbab9Jg3x9sOof3gG/view?usp=drive_link)
- `conda activate antifake310`
- `bash Detectors/AntifakePrompt/AntifakeEval.sh`
- DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
- $\rightarrow$ `./results/{date_time}_Antifake_blip2/predictions.csv`

### [fira7s/MM-Det](https://github.com/fira7s/MM-Det.git)
- `conda activate MM_Det`
- `bash Detectors/MM-Det/MMDetEval.sh`
- DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
- $\rightarrow$ `./results/{date_time}_MMDet/predictions.csv`
### BusterX
- ？

## Diffusion-Based Methods
### DRCT
### ~~DIRE - Unscientific Output~~

# Data Output

CSV files with columns:

| Col         | Description                                                                                                 |
|-------------|-------------------------------------------------------------------------------------------------------------|
| `sample_id` | unique id per row (string or int) you use to join with index                                                |
| `task`      | `Change_of_style`, `Extrapolation`, `Faceswap`, `I2V`, `IT2V`, `Inpainting`, `Interpolation`, `Real`, `T2V` |
| `method`    | `AnyV2V`, `RAVE`, `Cosmos-Predict2`, `Insightface`, etc.                                                    |
| `subset`    | `real_videos`, `fake_videos`, `real_frames`, `fake_frames`                                                  |
| `label`     | 0=real, 1=fake (ground truth)                                                                               |
| `model`     | model name or identifier                                                                                    |
| `mode`      | 'video' or 'frame'                                                                                          |
| `score`     | real-valued score, higher => more likely fake, -1 indicating unavailable                                    |
| `pred`      | hard prediction in {0,1} produced by the model                                                              | 
