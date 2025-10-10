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
 

### `Detectors/AIGVDet` 