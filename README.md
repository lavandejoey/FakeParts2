# FakeParts2

Table of Contents
=================

- [Environment Setup](#environment-setup)
- [Video Generation](#video-generation)
    - [T2V Models and Prompts](#t2v-models-and-prompts)
- [Data Preparation](#data-preparation)
    - [Dataset structure](#dataset-structure)
    - [Soft Links for Data](#soft-links-for-data)
- [Testing Different Detectors](#testing-different-detectors)
    - [Conda env Configuration:](#conda-env-configuration)
    - [Conventional DNN Methods](#conventional-dnn-methods)
    - [CLIP-Based Methods](#clip-based-methods)
    - [VLM Based Methods](#vlm-based-methods)
    - [Diffusion-Based Methods](#diffusion-based-methods)
- [Data Output](#data-output)

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

## Data Preparation

HF repository upload folder: `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition`

### Dataset structure

**The structure sample in tree form: [dataset_tree.md](dataset_tree.md)**

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

and soft-linked as:

- Soft-linking: `FakeParts_data_addition_videos_only` -- to --> `FakeParts_data_addition_frames_generation_dectection`
  for videos.
- Soft-linking: `FakeParts_data_addition_frames_0_1_labels` -- to -->
  `FakeParts_data_addition_frames_only/{0_real,1_fake}` for frames with 0/1 labels.

## Testing Different Detectors

### Conda env Configuration:

Models sharing the same environment are grouped together.

| Environment file                          | Related repositories                                                                                                                                                                    |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `conda env create -f env_antifake310.yml` | [lavandejoey/AntifakePrompt](#AntifakePrompt)                                                                                                                                           |
| `conda env create -f env_fakevlm310.yml`  | [lavandejoey/FakeVLM](#FakeVLM)<br>[lavandejoey/UniversalFakeDetect](#UniversalFakeDetect)<br>[lavandejoey/D3](#D3)<br>[Triocrossing/FatFormer](#FatFormer)<br>[Triocrossing/C2P](#C2P) |
| `conda env create -f env_MM_Det.yml`      | [fira7s/MM-Det](#MM-Det)                                                                                                                                                                |
| `conda env create -f env_sida311.yml`     | [lavandejoey/SIDA](#SIDA)                                                                                                                                                               |

### Conventional DNN Methods

- [lavandejoey/AIGVDet](https://github.com/lavandejoey/AIGVDet)<a id="AIGVDet"></a>
    - `conda activate fakevlm310`
    -
- DeMamba: WHERE IS MODEL?
---
### CLIP-Based Methods

- [Triocrossing/FatFormer](https://github.com/Triocrossing/fatformer_fakeparts)<a id="FatFormer"></a>
  - Download ckpt first:
      - [fatformer_fakeparts_ckpt.pth](https://drive.google.com/file/d/1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj/view?usp=sharing)
      - [ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)
  - DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
  - $\rightarrow$ `./results/{date_time}_fakeVLM/predictions.csv`
  ```bash
  cd Detectors/fatformer_fakeparts || exit
  # conda env create -f env_fakevlm310.yml
  conda activate fakevlm310
  [ -f pretrained/fatformer_4class_ckpt.pth ] && [ -f pretrained/ViT-L-14.pt ] && echo "All ckpt done" || echo "Download ckpt first"
  bash FatFormerEval.sh
  ```
---
- [Triocrossing/C2P](https://github.com/Triocrossing/c2p_FakeParts)<a id="C2P"></a>
  - DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
  - $\rightarrow$ `./results/{date_time}_C2P-CLIP/predictions.csv`
  ```bash
  cd Detectors/c2p_FakeParts || exit
  # conda env create -f env_fakevlm310.yml
  conda activate fakevlm310
  bash C2PEval.sh
  ``` 
---
- [lavandejoey/UniversalFakeDetect](https://github.com/lavandejoey/UniversalFakeDetect)<a id="UniversalFakeDetect"></a>
    ```bash
    cd Detectors/UniversalFakeDetect || exit
    # conda env create -f env_fakevlm310.yml
    conda activate fakevlm310
    bash UniFakeDetEval.sh
    ``` 
    - DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
    - $\rightarrow$ `./results/{date_time}_clip_vitl14/predictions.csv`
---
- [lavandejoey/De-Fake](https://github.com/lavandejoey/De-Fake)<a id="De-Fake"></a>
  - Download ckpt first:
      - [clip_linear.pt](https://drive.google.com/file/d/1qI7x5iodaCFq0S61LKw4wWjql7cYou_4/view?usp=sharing)
      - [finetune_clip.pt](https://drive.google.com/file/d/1SuenxJP10VwArC6zW0SHMUGObMRqQhBD/view?usp=sharing)
  - DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
  - $\rightarrow$ `./results/{date_time}_DeFake_ViTB32/predictions.csv`
  ```bash
  cd Detectors/De-Fake || exit
  # conda env create -f env_fakevlm310.yml
  conda activate fakevlm310
  [ -f clip_linear.pt ] && [ -f finetune_clip.pt ] && echo "All ckpt done" || echo "Download ckpt first"
  bash UniFakeDetEval.sh
  ``` 
---
- [lavandejoey/D3](https://github.com/lavandejoey/D3)<a id="D3"></a>
  - DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
  - $\rightarrow$ `./results/{date_time}_DeFake_ViTB32/predictions.csv`
  ```bash
  cd Detectors/D3 || exit
  # conda env create -f env_fakevlm310.yml
  conda activate fakevlm310
  bash D3Eval.sh
  ``` 
---
### VLM Based Methods

- [lavandejoey/FakeVLM](https://github.com/lavandejoey/FakeVLM)<a id="FakeVLM"></a>
  - DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
  - $\rightarrow$ `./results/{date_time}_fakeVLM/predictions.csv`
  ```bash
  cd Detectors/FakeVLM || exit
  # conda env create -f env_fakevlm310.yml
  conda activate fakevlm310
  bash FakeVLMEval.sh
  ``` 
---
- [lavandejoey/SIDA](https://github.com/lavandejoey/SIDA)<a id="SIDA"></a>
  - Download ckpt: [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
  - DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
  - $\rightarrow$ `./results/{date_time}_SIDA-13B_eval/predictions.csv`
  ```bash
  cd Detectors/SIDA || exit
  # conda env create -f env_sida311.yml
  conda activate sida311
  huggingface-cli download saberzl/SIDA-13B --local-dir ./ck/SIDA-13B --resume-download
  [ -f ck/sam_vit_h_4b8939.pth ] && [ -d ck/SIDA-13B ] && echo "All ckpt done" || echo "Download ckpt first"
  bash SIDAEval.sh
  ``` 
---
- [lavandejoey/AntifakePrompt](https://github.com/lavandejoey/AntifakePrompt)<a id="AntifakePrompt"></a>
  - Download ckpt with [download_checkpoints.sh](Detectors/AntifakePrompt/ckpt/download_checkpoints.sh):
      - [COCO_150k_SD3_SD2IP.pth](https://drive.google.com/file/d/1EUnVG4OZZPXeOyWaa5P590yCKGH-nunQ/view?usp=drive_link)
      - [COCO_150k_SD3_SD2IP_lama.pth](https://drive.google.com/file/d/1qnZfCknNHgC-Nhlwbab9Jg3x9sOof3gG/view?usp=drive_link)
  - Edit [blip2_instruct_vicuna7b_textinv.yaml](Detectors/AntifakePrompt/lavis/configs/models/blip2/blip2_instruct_vicuna7b_textinv.yaml), **_finetuned_** args.
  - DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
  - $\rightarrow$ `./results/{date_time}_Antifake_blip2/predictions.csv`
  ```bash
  cd Detectors/MM-Det || exit
  # conda env create -f env_antifake310.yml
  conda activate antifake310
  [ -f ckpt/COCO_150k_SD3_SD2IP.pth ] && echo "All ckpt done" || echo "Download ckpt first"
  bash MMDetEval.sh
  ```
---
- [fira7s/MM-Det](https://github.com/fira7s/MM-Det.git)<a id="MM-Det"></a>
  - Download ckpt first: [current_model.pth](https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy?usp=sharing)
  - DATA = `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only`
  - $\rightarrow$ `./results/{date_time}_MMDet/predictions.csv`
  ```bash
  cd Detectors/MM-Det || exit
  # conda env create -f env_MM_Det.yml
  conda activate MM_Det
  cd LLaVA && pip install -e . && cd ..
  [ -f weights/current_model.pth ] && echo "All ckpt done" || echo "Download ckpt first"
  bash MMDetEval.sh
  ```

- ~~BusterX~~ : No open-source code

### Diffusion-Based Methods

- DRCT
- ~~DIRE - Unscientific Output~~

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
