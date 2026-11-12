"""발화 목록 CSV → WavLM 층별 평균 특징 [N, 13, 768] (float16 .npy) + utt_id 색인.

실행:
    python -m emotion_scorer.extract_features --csv data/utterances.csv \
        --out features/features.npy --index features/features_index.txt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

from . import config
from .audio import load_audio

CSV_COLUMNS = ["utt_id", "speaker", "emotion", "sentence_id", "text", "split", "path", "duration"]


def load_wavlm(device):
    """고정된(eval, 학습 안 함) WavLM과 입력 전처리기를 불러온다."""
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.MODEL_NAME)
    model = WavLMModel.from_pretrained(config.MODEL_NAME).to(device).eval()
    model.requires_grad_(False)
    return feature_extractor, model


@torch.no_grad()
def extract_one(wav: np.ndarray, feature_extractor, model, device) -> np.ndarray:
    """16kHz 모노 음성 하나(배치 1, 패딩 없음) → [13, 768] float16."""
    inputs = feature_extractor(wav, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    out = model(inputs.input_values.to(device), output_hidden_states=True)
    hidden = torch.stack(out.hidden_states, dim=0)[:, 0]  # [13, T, 768]
    feat = hidden.mean(dim=1)  # 시간 축 평균 → [13, 768]
    assert feat.shape == (config.NUM_LAYERS, config.HIDDEN_SIZE), feat.shape
    return feat.float().cpu().numpy().astype(np.float16)


def features_for_file(path, feature_extractor, model, device) -> np.ndarray:
    """파일 경로 → [13, 768] float16. extract_features와 score가 같은 경로를 쓴다."""
    return extract_one(load_audio(path), feature_extractor, model, device)


def read_utterances(csv_path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: 없는 칸 {sorted(missing)}")
        rows = list(reader)
    ids = [r["utt_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{csv_path}: utt_id 중복")
    return rows


def resolve_path(p: str, base: Path) -> Path:
    """상대 경로는 CSV 파일이 있는 폴더 기준."""
    path = Path(p)
    return path if path.is_absolute() else base / path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=config.UTT_CSV)
    ap.add_argument("--out", type=Path, default=config.FEATURES_NPY)
    ap.add_argument("--index", type=Path, default=config.FEATURES_INDEX)
    args = ap.parse_args(argv)

    rows = read_utterances(args.csv)
    device = config.get_device()
    print(f"발화 {len(rows)}개, device={device}")
    feature_extractor, model = load_wavlm(device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.float16,
        shape=(len(rows), config.NUM_LAYERS, config.HIDDEN_SIZE),
    )
    base = args.csv.resolve().parent
    for i, r in enumerate(rows):
        feats[i] = features_for_file(resolve_path(r["path"], base), feature_extractor, model, device)
        if (i + 1) % 200 == 0 or i + 1 == len(rows):
            print(f"  {i + 1}/{len(rows)}")
    feats.flush()
    del feats

    args.index.write_text("".join(r["utt_id"] + "\n" for r in rows), encoding="utf-8")
    print(f"저장: {args.out} {(len(rows), config.NUM_LAYERS, config.HIDDEN_SIZE)} float16, 색인: {args.index}")


if __name__ == "__main__":
    main()
