import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from config import _DATASET_HYPER_PARAMS
from helpers.bbox_helper import nms
from helpers.vsumm_helper import bbox2summary
from models import Model_VideoSumm
from video_pipeline import (
    export_segment_screenshot,
    preprocess_video,
    write_alignment,
    write_asr_segments,
    write_metadata,
    write_sample_h5,
    write_text_features,
)


def build_model_args(dataset: str, device: str):
    params = _DATASET_HYPER_PARAMS[dataset]
    return argparse.Namespace(
        dataset=dataset,
        device=device,
        num_input_video=params["num_input_video"],
        num_input_text=params["num_input_text"],
        num_hidden=params["num_hidden"],
        num_layers=params["num_layers"],
        dropout_video=params["dropout_video"],
        dropout_text=params["dropout_text"],
        dropout_attn=params["dropout_attn"],
        dropout_fc=params["dropout_fc"],
        ratio=params["ratio"],
        nms_thresh=0.4,
    )


def load_sample(h5_path: Path, text_path: Path, alignment_path: Path | None = None, asr_path: Path | None = None):
    with h5py.File(h5_path, "r") as handle:
        key = next(iter(handle.keys()))
        group = handle[key]
        sample = {
            "key": key,
            "features": group["features"][...].astype(np.float32),
            "change_points": group["change_points"][...].astype(np.int64),
            "n_frames": int(group["n_frames"][...]),
            "n_frame_per_seg": group["n_frame_per_seg"][...].astype(np.int64),
            "picks": group["picks"][...].astype(np.int64),
        }

    text_payload = np.load(text_path, allow_pickle=True).item()
    sample["text_features"] = text_payload[sample["key"]].astype(np.float32)
    if alignment_path and alignment_path.exists():
        align_payload = np.load(alignment_path)
        sample["frame_to_text"] = align_payload["frame_to_text"].astype(np.int64)
        sample["text_to_frame"] = align_payload["text_to_frame"].astype(np.int64)
    if asr_path and asr_path.exists():
        sample["asr_segments"] = json.loads(asr_path.read_text(encoding="utf-8"))
    return sample


def build_alignment_masks(num_steps: int, num_sentences: int):
    sentence_span = max(1, int(np.ceil(num_steps / num_sentences)))
    video_to_text = np.zeros((num_steps, num_sentences), dtype=np.int64)
    text_to_video = np.zeros((num_sentences, num_steps), dtype=np.int64)
    for idx in range(num_sentences):
        start = idx * sentence_span
        end = min((idx + 1) * sentence_span, num_steps)
        video_to_text[start:end, idx] = 1
        text_to_video[idx, start:end] = 1
    return video_to_text, text_to_video


def list_checkpoints(checkpoint_dir: Path):
    checkpoints = sorted(checkpoint_dir.glob("model_best_split*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No split checkpoints found in {checkpoint_dir}")
    return checkpoints


def run_checkpoint(model, checkpoint_path: Path, sample, device: torch.device, nms_thresh: float):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    video = torch.from_numpy(sample["features"]).unsqueeze(0).to(device)
    text = torch.from_numpy(sample["text_features"]).unsqueeze(0).to(device)
    mask_video = torch.ones((1, sample["features"].shape[0]), dtype=torch.long, device=device)
    mask_text = torch.ones((1, sample["text_features"].shape[0]), dtype=torch.long, device=device)
    video_label = torch.zeros((1, sample["features"].shape[0]), dtype=torch.long, device=device)
    text_label = torch.zeros((1, sample["text_features"].shape[0]), dtype=torch.long, device=device)

    if "frame_to_text" in sample and "text_to_frame" in sample:
        frame_to_text = sample["frame_to_text"]
        text_to_frame = sample["text_to_frame"]
    else:
        frame_to_text, text_to_frame = build_alignment_masks(
            sample["features"].shape[0], sample["text_features"].shape[0]
        )
    video_to_text_mask_list = [torch.from_numpy(frame_to_text).to(device)]
    text_to_video_mask_list = [torch.from_numpy(text_to_frame).to(device)]

    with torch.no_grad():
        pred_cls_batch, pred_bboxes_batch = model.predict(
            video=video,
            text=text,
            mask_video=mask_video,
            mask_text=mask_text,
            video_label=video_label,
            text_label=text_label,
            video_to_text_mask_list=video_to_text_mask_list,
            text_to_video_mask_list=text_to_video_mask_list,
        )

    pred_cls = pred_cls_batch[0]
    pred_bboxes = np.clip(pred_bboxes_batch[0], 0, sample["features"].shape[0]).round().astype(np.int32)
    pred_cls, pred_bboxes = nms(pred_cls, pred_bboxes, nms_thresh)
    _, pred_summ_upsampled, pred_score, pred_score_upsampled = bbox2summary(
        sample["features"].shape[0],
        pred_cls,
        pred_bboxes,
        sample["change_points"],
        sample["n_frames"],
        sample["n_frame_per_seg"],
        sample["picks"],
        proportion=0.15,
        seg_score_mode="mean",
    )
    return {
        "checkpoint": checkpoint_path.name,
        "pred_cls": pred_cls,
        "pred_bboxes": pred_bboxes,
        "pred_score": pred_score,
        "pred_score_upsampled": pred_score_upsampled,
        "pred_summary_upsampled": pred_summ_upsampled,
    }


def summary_segments(summary_mask: np.ndarray, fps: float):
    indices = np.flatnonzero(summary_mask.astype(bool))
    if indices.size == 0:
        return []

    segments = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        segments.append((start, prev + 1))
        start = idx
        prev = idx
    segments.append((start, prev + 1))

    payload = []
    for start_frame, end_frame in segments:
        payload.append(
            {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_sec": round(start_frame / fps, 3),
                "end_sec": round(end_frame / fps, 3),
                "duration_sec": round((end_frame - start_frame) / fps, 3),
            }
        )
    return payload


def enrich_segments_with_screenshots(video_path: Path, work_dir: Path, segments):
    screenshot_dir = work_dir / "screenshots"
    enriched = []
    for idx, segment in enumerate(segments):
        mid_sec = (segment["start_sec"] + segment["end_sec"]) / 2
        screenshot_path = screenshot_dir / f"segment_{idx:02d}.jpg"
        export_segment_screenshot(video_path, screenshot_path, mid_sec)
        segment = dict(segment)
        segment["midpoint_sec"] = round(mid_sec, 3)
        segment["screenshot"] = str(screenshot_path.resolve())
        enriched.append(segment)
    return enriched


def _collect_asr_text(asr_segments):
    if not asr_segments:
        return ""
    texts = []
    for segment in asr_segments:
        text = str(segment.get("text", "")).strip()
        if text:
            texts.append(text)
    if not texts:
        return ""
    merged = "，".join(texts[:3])
    return merged[:80]


def build_summary_description(segments, total_duration_sec: float, summary_ratio: float, asr_segments=None):
    if not segments:
        return "当前模型没有选出明确的摘要片段。"

    speech_summary = _collect_asr_text(asr_segments or [])

    if len(segments) == 1 and summary_ratio >= 0.95:
        base = (
            f"模型将这段视频整体视为一个连续事件，保留了几乎完整的片段"
            f"（{segments[0]['start_sec']:.2f}s-{segments[0]['end_sec']:.2f}s，共 {total_duration_sec:.2f}s）。"
        )
        if speech_summary:
            return base + f" 结合语音内容看，这段摘要的核心对白大致是：“{speech_summary}”。"
        return base

    durations = "、".join(
        f"{seg['start_sec']:.2f}s-{seg['end_sec']:.2f}s" for seg in segments
    )
    base = (
        f"模型选出了 {len(segments)} 个摘要片段，共覆盖视频约 {summary_ratio:.1%} 的时长，"
        f"主要区间为：{durations}。"
    )
    if speech_summary:
        return base + f" 这些片段中的关键语音信息大致是：“{speech_summary}”。"
    return base


def main():
    parser = argparse.ArgumentParser(description="End-to-end raw video to A2Summ summary JSON.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--dataset", default="SumMe", choices=["SumMe", "TVSum"])
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("logs/raw_video_infer"))
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--segment-len", type=int, default=16)
    parser.add_argument("--sentence-span", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-imagenet-weights", action="store_true")
    parser.add_argument("--asr-model", default="openai/whisper-base")
    parser.add_argument("--asr-language", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    sample = preprocess_video(
        video_path=args.video,
        sample_every=args.sample_every,
        segment_len=args.segment_len,
        sentence_span=args.sentence_span,
        batch_size=args.batch_size,
        device=device,
        use_imagenet_weights=not args.no_imagenet_weights,
        asr_model_name=args.asr_model,
        asr_language=args.asr_language,
    )

    stem = args.video.stem
    h5_path = args.work_dir / f"{stem}.h5"
    text_path = args.work_dir / f"{stem}_text_roberta.npy"
    alignment_path = args.work_dir / f"{stem}_alignment.npz"
    asr_path = args.work_dir / f"{stem}_asr.json"
    metadata_path = args.work_dir / f"{stem}_metadata.json"
    write_sample_h5(sample, h5_path, video_name=args.video.name)
    write_text_features(sample, text_path)
    write_alignment(sample, alignment_path)
    write_asr_segments(sample, asr_path)
    write_metadata(sample, metadata_path, source_video=args.video)

    loaded = load_sample(h5_path, text_path, alignment_path=alignment_path, asr_path=asr_path)
    model_args = build_model_args(args.dataset, device=str(device))
    model = Model_VideoSumm(model_args).to(device)

    split_results = []
    for checkpoint_path in list_checkpoints(args.checkpoint_dir):
        split_results.append(run_checkpoint(model, checkpoint_path, loaded, device, model_args.nms_thresh))

    avg_score = np.mean([item["pred_score_upsampled"] for item in split_results], axis=0)
    summary_mask = avg_score > 0
    segments = summary_segments(summary_mask, fps=sample.fps)
    segments = enrich_segments_with_screenshots(args.video, args.work_dir, segments)
    total_duration_sec = sample.n_frames / sample.fps if sample.fps else 0.0
    summary_ratio = round(float(summary_mask.mean()) if summary_mask.size else 0.0, 6)
    summary_description = build_summary_description(
        segments,
        total_duration_sec=total_duration_sec,
        summary_ratio=summary_ratio,
        asr_segments=loaded.get("asr_segments", []),
    )

    payload = {
        "source_video": str(args.video.resolve()),
        "dataset_profile": args.dataset,
        "work_dir": str(args.work_dir.resolve()),
        "generated_h5": str(h5_path.resolve()),
        "generated_text_features": str(text_path.resolve()),
        "generated_alignment": str(alignment_path.resolve()),
        "generated_asr": str(asr_path.resolve()),
        "generated_metadata": str(metadata_path.resolve()),
        "num_original_frames": sample.n_frames,
        "fps": sample.fps,
        "num_sampled_frames": int(sample.features.shape[0]),
        "num_text_tokens": int(sample.text_features.shape[0]),
        "asr_segments": loaded.get("asr_segments", []),
        "summary_ratio": summary_ratio,
        "summary_description": summary_description,
        "selected_segments": segments,
        "ensemble": [
            {
                "checkpoint": item["checkpoint"],
                "nonzero_score_frames": int(np.count_nonzero(item["pred_score_upsampled"] > 0)),
            }
            for item in split_results
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
