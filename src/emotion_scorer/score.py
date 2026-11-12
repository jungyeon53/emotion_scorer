"""음성 파일 목록 또는 폴더 → 감정 확률 CSV.

출력 칸: file, p_neutral, p_happy, p_angry, p_sad, p_surprise, pred
입력 샘플레이트와 무관하게 audio.load_audio로 16kHz 변환 후, 체크포인트의 정규화 값을 적용한다.

실행:
    python -m emotion_scorer.score tts_outputs/ --checkpoint checkpoints/classifier.pt --out scores.csv
"""

import argparse
import csv
from pathlib import Path

import torch

from . import config
from .extract_features import features_for_file, load_wavlm
from .train_classifier import LayerWeightedClassifier

AUDIO_EXTS = {".wav", ".flac", ".ogg"}


def collect_paths(inputs) -> list[Path]:
    paths = []
    for p in map(Path, inputs):
        if p.is_dir():
            paths += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(p)
    return paths


def load_classifier(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["labels"] != config.LABELS or ckpt["model_name"] != config.MODEL_NAME:
        raise ValueError("체크포인트의 라벨 순서/모델 이름이 config와 다름")
    saved, current = ckpt.get("preprocess"), config.preprocess_config()
    if saved != current:
        raise ValueError(f"체크포인트 전처리 설정 {saved} != 현재 config {current}. 같은 전처리로 다시 학습하세요.")
    model = LayerWeightedClassifier()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["mean"].to(device), ckpt["std"].to(device)


@torch.no_grad()
def score_files(paths, checkpoint, device=None) -> list[dict]:
    device = device or config.get_device()
    feature_extractor, wavlm = load_wavlm(device)
    clf, mean, std = load_classifier(checkpoint, device)
    results = []
    for path in paths:
        feat = torch.from_numpy(features_for_file(path, feature_extractor, wavlm, device)).float().to(device)
        probs = torch.softmax(clf(((feat - mean) / std)[None]), dim=1)[0].cpu().tolist()
        row = {"file": str(path)}
        row.update({f"p_{name}": p for name, p in zip(config.LABELS, probs)})
        row["pred"] = config.LABELS[max(range(len(probs)), key=probs.__getitem__)]
        results.append(row)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="음성 파일 또는 폴더 (폴더는 하위까지 .wav/.flac/.ogg)")
    ap.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT)
    ap.add_argument("--out", type=Path, default=Path("scores.csv"))
    args = ap.parse_args(argv)

    paths = collect_paths(args.inputs)
    print(f"파일 {len(paths)}개")
    results = score_files(paths, args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file"] + [f"p_{n}" for n in config.LABELS] + ["pred"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
