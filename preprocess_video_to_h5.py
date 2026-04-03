import argparse
from pathlib import Path

import torch

from video_pipeline import (
    preprocess_video,
    write_alignment,
    write_asr_segments,
    write_metadata,
    write_sample_h5,
    write_text_features,
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess a raw video into A2Summ-compatible h5/text artifacts.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--segment-len", type=int, default=16)
    parser.add_argument("--sentence-span", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-imagenet-weights", action="store_true")
    parser.add_argument("--asr-model", default="openai/whisper-base")
    parser.add_argument("--asr-language", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = preprocess_video(
        video_path=args.video,
        sample_every=args.sample_every,
        segment_len=args.segment_len,
        sentence_span=args.sentence_span,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        use_imagenet_weights=not args.no_imagenet_weights,
        asr_model_name=args.asr_model,
        asr_language=args.asr_language,
    )

    stem = args.video.stem
    h5_path = args.output_dir / f"{stem}.h5"
    text_path = args.output_dir / f"{stem}_text_roberta.npy"
    alignment_path = args.output_dir / f"{stem}_alignment.npz"
    asr_path = args.output_dir / f"{stem}_asr.json"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    write_sample_h5(sample, h5_path, video_name=args.video.name)
    write_text_features(sample, text_path)
    write_alignment(sample, alignment_path)
    write_asr_segments(sample, asr_path)
    write_metadata(sample, metadata_path, source_video=args.video)

    print(f"h5: {h5_path}")
    print(f"text: {text_path}")
    print(f"alignment: {alignment_path}")
    print(f"asr: {asr_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
