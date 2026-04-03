# Align and Attend: Multimodal Summarization with Dual Contrastive Losses (CVPR2023)
### [Project Page](https://boheumd.github.io/A2Summ/) | [Paper](https://arxiv.org/abs/2303.07284)
The official repository of our paper "**Align and Attend: Multimodal Summarization with Dual Contrastive Losses**".

<p align="center">
<img src="figs/teaser.png" alt="teaser" width="80%">
</p>


## Model Overview
<p align="center">
<img src="figs/model.png" alt="model" width="80%">
</p>


## Requirements
This project now uses `uv` to manage dependencies.

```bash
uv python install 3.10.20
uv sync
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Notes:
- `torch` and `torchvision` are pinned to the Tsinghua PyPI mirror through `pyproject.toml`.
- To use the CUDA 13 line from the Tsinghua mirror, the project targets Python 3.10 and installs `torch==2.11.0` with `torchvision==0.26.0`.
- These PyPI wheels resolve to CUDA 13 runtime dependencies such as `nvidia-cudnn-cu13` through `uv sync`.
- `nltk` resources `punkt` and `punkt_tab` are required for ROUGE evaluation during training and validation.

If you prefer a one-off sync with an existing interpreter, you can also run:
```bash
uv sync --python 3.10.20
```

## Dataset
We evaluate our A2Summ on two multimodal summarization multimodal output datasets ([CNN, Daily_Mail](https://aclanthology.org/2021.naacl-main.473.pdf)) and two standard video summarization datasets ([SumMe](https://gyglim.github.io/me/papers/GygliECCV14_vsum.pdf), [TVSum](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf)).
We also collected a large-scale multimodal summarization dataset BLiSS which consists of livestream videos and transcripts with annotated summary.
Before running the code, please download the pre-processed datasets from [google drive link](https://drive.google.com/drive/folders/1rqXEIelRzq4mb7NaBk3GXxh7jlfP_Snm?usp=share_link).
Unzip it under the `data/` folder and make sure the data structure is as below.

   ```
    ├── data
        └── BLiSS
            ├── annotation
            ├── feature
        └── CNN
            ├── annotation
            ├── feature
        └── Daily_Mail
            ├── annotation
            ├── feature
        └── SumMe
            ├── caption
            ├── feature
            ├── splits.yml
        └── TVSum
            ├── caption
            ├── feature
            ├── splits.yml

   ```
### BLiSS Dataset
For the BLiSS dataset, due to the copyright issue, we only provide the extracted video/thumbnail features instead of the original videos/thunmbnails. If you need access to the original videos, please email me (bohe@umd.edu) for the public URLs of each video.


## Running

### Training
We train the model on a single GTX-1080ti GPU. To train the model on different dataset, please execute the following command.
```bash
uv run python train.py --dataset ${dataset}
```

### Testing
First, download the [checkpoints](https://drive.google.com/file/d/1LuXWjW3BcAXCOals4o2UUVYMx-FYnJ3T/view?usp=sharing) into "saved_model" directory and pass it as the checkpoint flag. 

```bash
uv run python train.py --dataset ${dataset} \
    --test --checkpoint saved_model/${dataset}
```

### Raw Video Inference
The repository can also run a lightweight end-to-end inference path for `SumMe` / `TVSum` style checkpoints:

1. Sample frames from a raw video with `ffmpeg`
2. Extract GoogleNet pool5-style frame features
3. Run Whisper ASR to get timestamped speech segments
4. Build a DSNet/VASNet-like `.h5` file plus ASR-backed text features and alignment masks
4. Reuse the current `Model_VideoSumm` checkpoints to export summary JSON

Preprocess only:
```bash
uv run python preprocess_video_to_h5.py \
    --video /path/to/input.mp4 \
    --output-dir /path/to/artifacts \
    --asr-model openai/whisper-base
```

End-to-end inference:
```bash
uv run python infer_video_summary.py \
    --video /path/to/input.mp4 \
    --dataset SumMe \
    --checkpoint-dir saved_model/SumMe \
    --work-dir logs/raw_video_infer \
    --output-json logs/raw_video_infer/result.json \
    --asr-model openai/whisper-base
```

Notes:
- This path is format-compatible with the existing `SumMe` / `TVSum` pipeline, but it is only an approximation of the original dataset preprocessing.
- The script now uses Whisper ASR segments to build text features and cross-modal alignment, which is closer to the original multimodal setting than the previous proxy-text fallback.
- Generated artifacts include `*.h5`, `*_text_roberta.npy`, `*_alignment.npz`, `*_asr.json`, `*_metadata.json`, screenshots, and the final summary JSON.


## Citation
If you find our code or our paper useful for your research, please **[★star]** this repo and **[cite]** the following paper:

```latex
@inproceedings{he2023a2summ,
  title = {Align and Attend: Multimodal Summarization with Dual Contrastive Losses},
  author={He, Bo and Wang, Jun and Qiu, Jielin and Bui, Trung and Shrivastava, Abhinav and Wang, Zhaowen},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}
```

## Acknowledgement
We referenced the repos below for the code
- [DSNet](https://github.com/li-plus/DSNet)
- [UMT](https://github.com/TencentARC/UMT)
