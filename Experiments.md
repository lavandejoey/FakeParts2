# Data Preparation

HF repository upload folder: `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition`

Dataset structure

Relative paths has 2 patterns, SUBSET in {fake_frames, fake_videos, real_frames, real_videos}:

```text
<TASK>/<METHOD>/<SUBSET>/
<TASK>/<METHOD>/<S_METHOD>/<SUBSET>/
```

After that, the files are organized as:

```text
<v_name>/frame_<%06d>.jpg 
<v_name>.mp4
```

```text
Change_of_style/AnyV2V/fake_frames/<v_name>/frame_<%06d>.jpg
Change_of_style/AnyV2V/fake_videos/<v_name>.mp4
Change_of_style/RAVE/fake_frames/<v_name>/frame_<%06d>.jpg
Change_of_style/RAVE/fake_videos/<v_name>.mp4
Extrapolation/Cosmos-Predict2/fake_frames/<v_name>/frame_<%06d>.jpg
Extrapolation/Cosmos-Predict2/fake_videos/<v_name>.mp4
Extrapolation/Cosmos-Predict2/real_frames/<v_name>/frame_<%06d>.jpg
Faceswap/Insightface/fake_frames/<v_name>/frame_<%06d>.jpg
Faceswap/Insightface/fake_videos/<v_name>.mp4
Inpainting/DiffuEraser/fake_frames/<v_name>/frame_<%06d>.jpg
Inpainting/DiffuEraser/fake_videos/<v_name>.mp4
Inpainting/Propainter/fake_frames/<v_name>/frame_<%06d>.jpg
Inpainting/Propainter/fake_videos/<v_name>.mp4
Inpainting/ROVI/Inpainting_MP4/<v_name>.mp4
Inpainting/ROVI/REAL_JPEG_MP4/<v_name>.mp4
Interpolation/Framer/fake_frames/<v_name>/frame_<%06d>.jpg
Interpolation/Framer/fake_videos/<v_name>.mp4
Real/TenKReal/real_frames/<v_name>/frame_<%06d>.jpg
Real/TenKReal/real_videos/<v_name>.mp4
T2V/CogVideoX/fake_frames/<v_name>/frame_<%06d>.jpg
T2V/CogVideoX/fake_videos/<v_name>.mp4
T2V/HunyuanVideo/fake_frames/<v_name>/frame_<%06d>.jpg
T2V/HunyuanVideo/fake_videos/<v_name>.mp4
T2V/LTXVideo/fake_frames/<v_name>/frame_<%06d>.jpg
T2V/LTXVideo/fake_videos/<v_name>.mp4
T2V/Mochi1/fake_frames/<v_name>/frame_<%06d>.jpg
T2V/Mochi1/fake_videos/<v_name>.mp4
T2V/Open-Sora/fake_frames/<v_name>/frame_<%06d>.jpg
T2V/Open-Sora/fake_videos/<v_name>.mp4
T2V/Wan_21/fake_frames/<v_name>/frame_<%06d>.jpg
T2V/Wan_21/fake_videos/<v_name>.mp4
```

### Soft Links for Data

Data Physically stored in 
- `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only/` for frames' folders.
- `/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_generation_dectection/` for videos.
- Soft-linking: `FakeParts_data_addition_videos_only` -- to --> `FakeParts_data_addition_frames_generation_dectection` for videos.
- Soft-linking: `FakeParts_data_addition_frames_0_1_labels` -- to --> `FakeParts_data_addition_frames_only/{0_real,1_fake}` for frames with 0/1 labels.

e.g. 

```text
SRC: /projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only/Extrapolation/Cosmos-Predict2/fake_frames
DST: /projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_0_1_labels/1_fake/Extrapolation/Cosmos-Predict2/fake_frames

SRC: /projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only/Extrapolation/Cosmos-Predict2/real_frames
DST: /projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_0_1_labels/0_real/Extrapolation/Cosmos-Predict2/real_frames

SRC: /projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only/Inpainting/Propainter/fake_frames
DST: /projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_0_1_labels/1_fake/Inpainting/Propainter/fake_frames

SRC: /projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only/Real/TenKReal/real_frames
DST: /projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_0_1_labels/0_real/Real/TenKReal/real_frames

...
```

AIGVDet
AntifakePrompt
BusterX
D3
- De-Fake
  - Download ckpt first: 
    - [clip_linear.pt](https://drive.google.com/file/d/1qI7x5iodaCFq0S61LKw4wWjql7cYou_4/view?usp=sharing)
    - [finetune_clip.pt](https://drive.google.com/file/d/1SuenxJP10VwArC6zW0SHMUGObMRqQhBD/view?usp=sharing)
  - `conda activate defake`
  - DATA = `"/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"`
  - $\rightarrow$ `./results/{date_time}_DeFake_ViTB32/predictions.csv`
- DeMamba: WHERE IS MODEL?
- FakeVLM
  - `conda activate fakevlm310`
  - DATA = `"/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"`
  - $\rightarrow$ `./results/{date_time}_fakeVLM/predictions.csv`
- MM-Det
- SIDA
  - `conda activate sida311`
  - DATA = `"/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"`
  - $\rightarrow$ `./results/{date_time}_SIDA-13B_eval/predictions.csv`
- UniversalFakeDetect
  - `conda activate fakevlm310`
  - DATA = `"/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"`
  - $\rightarrow$ `./results/{date_time}_clip_vitl14/predictions.csv`

# Data Output

## Raw results structure