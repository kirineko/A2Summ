import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from torchvision.models import GoogLeNet_Weights, googlenet
from torchvision.models.feature_extraction import create_feature_extractor


_WHISPER_CACHE = {}


@dataclass
class ASRSegment:
    text: str
    start_sec: float
    end_sec: float


@dataclass
class VideoSample:
    key: str
    features: np.ndarray
    text_features: np.ndarray
    change_points: np.ndarray
    n_frames: int
    n_frame_per_seg: np.ndarray
    picks: np.ndarray
    frame_to_text: np.ndarray
    text_to_frame: np.ndarray
    sample_every: int
    fps: float
    asr_segments: List[ASRSegment]


def probe_video(video_path: Path) -> Tuple[int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    frame_count = int(stream["nb_read_frames"])
    fps_num, fps_den = stream["r_frame_rate"].split("/")
    fps = float(fps_num) / float(fps_den)
    return frame_count, fps


def extract_frames(video_path: Path, frame_dir: Path, sample_every: int) -> List[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / "frame_%06d.jpg"
    vf = f"select=not(mod(n\\,{sample_every}))"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if frame_paths:
        return frame_paths

    fallback = frame_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(fallback),
    ]
    subprocess.run(cmd, check=True)
    return sorted(frame_dir.glob("frame_*.jpg"))


def extract_audio(video_path: Path, audio_path: Path, sample_rate: int = 16000) -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    return audio_path


def build_feature_extractor(device: torch.device, use_imagenet_weights: bool = True):
    weights = GoogLeNet_Weights.DEFAULT if use_imagenet_weights else None
    model = googlenet(weights=weights, aux_logits=True if weights is not None else False)
    model.eval().to(device)
    extractor = create_feature_extractor(model, return_nodes={"avgpool": "features"})
    transform = weights.transforms() if weights is not None else GoogLeNet_Weights.DEFAULT.transforms()
    return extractor, transform


def get_whisper_components(model_name: str, device: torch.device):
    key = (model_name, str(device))
    if key in _WHISPER_CACHE:
        return _WHISPER_CACHE[key]

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    _WHISPER_CACHE[key] = (model, processor, asr_pipe)
    return _WHISPER_CACHE[key]


def extract_googlenet_features(
    frame_paths: List[Path],
    device: torch.device,
    batch_size: int = 32,
    use_imagenet_weights: bool = True,
) -> np.ndarray:
    extractor, transform = build_feature_extractor(device, use_imagenet_weights=use_imagenet_weights)
    outputs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                images.append(transform(image))
            batch = torch.stack(images, dim=0).to(device)
            feats = extractor(batch)["features"].flatten(1).cpu().numpy().astype(np.float32)
            outputs.append(feats)

    return np.concatenate(outputs, axis=0)


def transcribe_with_whisper(
    audio_path: Path,
    model_name: str,
    device: torch.device,
    language: str | None = None,
) -> List[ASRSegment]:
    model, processor, asr_pipe = get_whisper_components(model_name, device)
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language

    result = asr_pipe(
        str(audio_path),
        return_timestamps=True,
        chunk_length_s=30,
        batch_size=8,
        generate_kwargs=generate_kwargs,
    )

    segments: List[ASRSegment] = []
    for chunk in result.get("chunks", []):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        start_sec, end_sec = chunk.get("timestamp", (None, None))
        if start_sec is None:
            start_sec = 0.0
        if end_sec is None:
            end_sec = start_sec
        segments.append(
            ASRSegment(
                text=text,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
            )
        )

    if segments:
        return segments

    fallback_text = result.get("text", "").strip()
    if fallback_text:
        return [ASRSegment(text=fallback_text, start_sec=0.0, end_sec=0.0)]
    return []


def build_asr_text_features(
    segments: List[ASRSegment],
    model_name: str,
    device: torch.device,
    text_dim: int = 768,
) -> np.ndarray:
    model, processor, _ = get_whisper_components(model_name, device)
    tokenizer = processor.tokenizer
    embed_tokens = model.model.decoder.embed_tokens

    if not segments:
        return np.zeros((1, text_dim), dtype=np.float32)

    pooled = []
    with torch.no_grad():
        for segment in segments:
            tokenized = tokenizer(
                segment.text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            input_ids = tokenized["input_ids"].to(device)
            token_embeddings = embed_tokens(input_ids).mean(dim=1)
            pooled.append(token_embeddings.squeeze(0).cpu().numpy())

    pooled = np.stack(pooled, axis=0).astype(np.float32)
    feat_dim = pooled.shape[1]
    if feat_dim == text_dim:
        return pooled

    rng = np.random.default_rng(0)
    proj = rng.standard_normal((feat_dim, text_dim), dtype=np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True).clip(min=1e-6)
    return (pooled @ proj).astype(np.float32)


def build_alignment_masks_from_segments(
    num_steps: int,
    n_frames: int,
    sample_every: int,
    fps: float,
    segments: List[ASRSegment],
) -> Tuple[np.ndarray, np.ndarray]:
    if not segments:
        frame_to_text = np.ones((num_steps, 1), dtype=np.int64)
        text_to_frame = np.ones((1, num_steps), dtype=np.int64)
        return frame_to_text, text_to_frame

    num_sentences = len(segments)
    frame_to_text = np.zeros((num_steps, num_sentences), dtype=np.int64)
    text_to_frame = np.zeros((num_sentences, num_steps), dtype=np.int64)

    sampled_frame_seconds = (np.arange(num_steps) * sample_every) / fps
    for idx, segment in enumerate(segments):
        start_sec = max(segment.start_sec, 0.0)
        end_sec = max(segment.end_sec, start_sec + 1.0 / fps)
        mask = (sampled_frame_seconds >= start_sec) & (sampled_frame_seconds <= end_sec)
        if not np.any(mask):
            nearest = int(np.clip(round(start_sec * fps / sample_every), 0, num_steps - 1))
            mask[nearest] = True
        frame_to_text[mask, idx] = 1
        text_to_frame[idx, mask] = 1

    return frame_to_text, text_to_frame


def build_change_points(n_frames: int, sample_every: int, segment_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    picks = np.arange(0, n_frames, sample_every, dtype=np.int64)
    if picks.size == 0:
        picks = np.array([0], dtype=np.int64)

    segment_size = sample_every * segment_len
    starts = np.arange(0, n_frames, segment_size, dtype=np.int64)
    change_points = []
    n_frame_per_seg = []
    for start in starts:
        end = min(start + segment_size, n_frames) - 1
        change_points.append([int(start), int(end)])
        n_frame_per_seg.append(int(end - start + 1))

    return (
        np.asarray(change_points, dtype=np.int64),
        np.asarray(n_frame_per_seg, dtype=np.int64),
        picks,
    )


def preprocess_video(
    video_path: Path,
    sample_every: int,
    segment_len: int,
    sentence_span: int,
    batch_size: int,
    device: torch.device,
    use_imagenet_weights: bool = True,
    asr_model_name: str = "openai/whisper-base",
    asr_language: str | None = None,
) -> VideoSample:
    video_path = video_path.resolve()
    key = video_path.stem
    n_frames, fps = probe_video(video_path)

    with tempfile.TemporaryDirectory(prefix="a2summ_preproc_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        frame_dir = tmp_root / "frames"
        frame_paths = extract_frames(video_path, frame_dir, sample_every=sample_every)
        features = extract_googlenet_features(
            frame_paths,
            device=device,
            batch_size=batch_size,
            use_imagenet_weights=use_imagenet_weights,
        )
        audio_path = extract_audio(video_path, tmp_root / "audio.wav")
        asr_segments = transcribe_with_whisper(
            audio_path=audio_path,
            model_name=asr_model_name,
            device=device,
            language=asr_language,
        )

    text_features = build_asr_text_features(
        asr_segments,
        model_name=asr_model_name,
        device=device,
    )
    frame_to_text, text_to_frame = build_alignment_masks_from_segments(
        num_steps=features.shape[0],
        n_frames=n_frames,
        sample_every=sample_every,
        fps=fps,
        segments=asr_segments,
    )
    change_points, n_frame_per_seg, picks = build_change_points(
        n_frames=n_frames, sample_every=sample_every, segment_len=segment_len
    )

    return VideoSample(
        key=key,
        features=features,
        text_features=text_features,
        change_points=change_points,
        n_frames=n_frames,
        n_frame_per_seg=n_frame_per_seg,
        picks=picks,
        frame_to_text=frame_to_text,
        text_to_frame=text_to_frame,
        sample_every=sample_every,
        fps=fps,
        asr_segments=asr_segments,
    )


def write_sample_h5(sample: VideoSample, output_path: Path, video_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        group = handle.create_group(sample.key)
        group.create_dataset("features", data=sample.features.astype(np.float32))
        group.create_dataset("gtscore", data=np.zeros(sample.features.shape[0], dtype=np.float32))
        group.create_dataset("gtsummary", data=np.zeros(sample.features.shape[0], dtype=np.float32))
        group.create_dataset("change_points", data=sample.change_points.astype(np.int64))
        group.create_dataset("n_frame_per_seg", data=sample.n_frame_per_seg.astype(np.int64))
        group.create_dataset("n_frames", data=np.int64(sample.n_frames))
        group.create_dataset("n_steps", data=np.int64(sample.features.shape[0]))
        group.create_dataset("picks", data=sample.picks.astype(np.int64))
        group.create_dataset("user_summary", data=np.zeros((1, sample.n_frames), dtype=np.float32))
        group.create_dataset("video_name", data=np.bytes_(video_name))


def write_text_features(sample: VideoSample, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, {sample.key: sample.text_features.astype(np.float32)}, allow_pickle=True)


def write_metadata(sample: VideoSample, output_path: Path, source_video: Path) -> None:
    payload = {
        "key": sample.key,
        "source_video": str(source_video.resolve()),
        "num_original_frames": sample.n_frames,
        "fps": sample.fps,
        "sample_every": sample.sample_every,
        "num_sampled_frames": int(sample.features.shape[0]),
        "num_text_tokens": int(sample.text_features.shape[0]),
        "change_points": sample.change_points.tolist(),
        "asr_segments": [
            {
                "text": segment.text,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
            }
            for segment in sample.asr_segments
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def export_segment_screenshot(video_path: Path, output_path: Path, timestamp_sec: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(timestamp_sec, 0.0):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def write_alignment(sample: VideoSample, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        frame_to_text=sample.frame_to_text.astype(np.int64),
        text_to_frame=sample.text_to_frame.astype(np.int64),
    )


def write_asr_segments(sample: VideoSample, output_path: Path) -> None:
    payload = [
        {
            "text": segment.text,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
        }
        for segment in sample.asr_segments
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
