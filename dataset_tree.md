### Frame Folder Structure 

`/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only/` for frames' folders, with the structure as:

- `<DATA_ROOT> / <Task Category> / <Method Name> / <Subset (Label)> / <Video Name> / <Frames Ordinal>{.jpg,.png}`
- `<DATA_ROOT> / <Task Category> / <Method Name> / <Sub-Methods Name> / <Subset (Label)> / <Video Name> / <Frames Ordinal>{.jpg,.png}`

```text
.
├── Change_of_style/
│   ├── AnyV2V/
│   │   └── fake_frames/
│   │       ├── 1/
│   │       └── 2/
│   └── ...
├── Extrapolation/
│   └── Cosmos-Predict2/
│       ├── fake_frames/
│       │   ├── 1/
│       │   └── 2/
│       └── real_frames/
│           ├── 1/
│           └── 2/
├── ...
├── Inpainting/
│   ...
│   └── ROVI/
│       ├── Inpainting_MP4/
│       │   └── fake_frames/
│       │       ├── 1/
│       │       └── 10/
│       └── ...
└── Real/
    └── TenKReal/
        └── real_frames/
            ├── 1/
            └── 10/
```

### Video Folder Structure

`/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_generation_dectection/` for videos, with the structure as:
- `<DATA_ROOT> / <Task Category> / <Method Name> / <Subset (Label)> / <Video Name>.mp4`
- `<DATA_ROOT> / <Task Category> / <Method Name> / <Sub-Methods Name> / <Subset (Label)> / <Video Name>.mp4`

```text
.
├── Change_of_style/
│   ├── AnyV2V/
│   │   └── fake_videos/
│   │       ├── 1.mp4
│   │       └── 2.mp4
│   └── ...
├── Extrapolation/
│   └── Cosmos-Predict2/
│       └── fake_videos/
│           ├── 1.mp4
│           └── 2.mp4
├── ...
├── Inpainting/
│   ...
│   └── ROVI/
│       ├── Inpainting_MP4/
│       │   └── fake_videos/
│       │       ├── 1.mp4
│       │       └── 10.mp4
│       └── ...
└── Real/
    └── TenKReal/
        └── real_videos/
            ├── 1.mp4
            └── 10.mp4
```
