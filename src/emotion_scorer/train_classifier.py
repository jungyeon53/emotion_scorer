"""특징 파일 + 발화 목록 → 층 가중합 분류기.

train으로 학습, eval 정확도로 early stopping/최고 모델 선택, test는 마지막에 한 번만 평가.

실행:
    python -m emotion_scorer.train_classifier --csv data/utterances.csv \
        --features features/features.npy --index features/features_index.txt \
        --out checkpoints/classifier.pt
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config
from .extract_features import read_utterances


class LayerWeightedClassifier(nn.Module):
    """13개 층 softmax 가중합 → [768] → Linear(768 → 5). 입력은 정규화된 [B, 13, 768]."""

    def __init__(self):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(config.NUM_LAYERS))
        self.linear = nn.Linear(config.HIDDEN_SIZE, len(config.LABELS))

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, x):
        pooled = (self.layer_weights()[None, :, None] * x).sum(dim=1)
        return self.linear(pooled)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(features_npy, index_path, csv_path):
    """특징 [N,13,768] float32, 라벨 [N], split [N]을 utt_id로 맞춰서 돌려준다."""
    feats = np.load(features_npy).astype(np.float32)
    utt_ids = Path(index_path).read_text(encoding="utf-8").split()
    if len(utt_ids) != len(feats):
        raise ValueError(f"색인 {len(utt_ids)}개 != 특징 {len(feats)}개")
    rows = {r["utt_id"]: r for r in read_utterances(csv_path)}
    missing = [u for u in utt_ids if u not in rows]
    if missing:
        raise ValueError(f"CSV에 없는 utt_id {len(missing)}개, 예: {missing[:3]}")
    label_of = {name: i for i, name in enumerate(config.LABELS)}
    labels = np.array([label_of[rows[u]["emotion"].strip().lower()] for u in utt_ids], dtype=np.int64)
    splits = np.array([rows[u]["split"].strip().lower() for u in utt_ids])
    return feats, labels, splits


@torch.no_grad()
def predict(model, x: torch.Tensor, device) -> np.ndarray:
    model.eval()
    preds = [model(x[i:i + 1024].to(device)).argmax(dim=1).cpu() for i in range(0, len(x), 1024)]
    return torch.cat(preds).numpy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    k = len(config.LABELS)
    cm = np.zeros((k, k), dtype=np.int64)  # 행=정답, 열=예측
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return {"accuracy": float(tp.sum() / max(cm.sum(), 1)), "recall": recall, "macro_f1": float(f1.mean()), "cm": cm}


def format_report(m: dict, layer_weights: np.ndarray, n_test: int) -> str:
    lines = [f"[test] 발화 {n_test}개", f"전체 정확도: {m['accuracy']:.4f}", f"macro F1: {m['macro_f1']:.4f}", "감정별 정답률:"]
    lines += [f"  {name:<9} {r:.4f}" for name, r in zip(config.LABELS, m["recall"])]
    lines.append("혼동 행렬 (행=정답, 열=예측):")
    lines.append(" " * 10 + "".join(f"{n[:7]:>8}" for n in config.LABELS))
    lines += [f"  {name:<8}" + "".join(f"{v:>8d}" for v in row) for name, row in zip(config.LABELS, m["cm"])]
    lines.append("층 가중치 (0=CNN 출력, 1~12=트랜스포머 층):")
    lines += [f"  layer {i:>2}: {w:.4f}" for i, w in enumerate(layer_weights)]
    return "\n".join(lines)


def train(features_npy, index_path, csv_path, out_path, max_epochs=config.MAX_EPOCHS):
    set_seed(config.SEED)
    device = config.get_device()
    feats, labels, splits = load_dataset(features_npy, index_path, csv_path)
    for s in ("train", "eval", "test"):
        if not (splits == s).any():
            raise ValueError(f"split '{s}' 발화가 없음")

    tr, ev, te = splits == "train", splits == "eval", splits == "test"
    # 정규화 값은 train 특징으로만 계산 (층별·차원별 [13, 768])
    mean = feats[tr].mean(axis=0)
    std = feats[tr].std(axis=0) + 1e-5
    x = torch.from_numpy((feats - mean) / std)
    y = torch.from_numpy(labels)
    print(f"device={device}, train {tr.sum()} / eval {ev.sum()} / test {te.sum()}")

    model = LayerWeightedClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_fn = nn.CrossEntropyLoss()
    x_tr, y_tr = x[tr], y[tr]
    gen = torch.Generator().manual_seed(config.SEED)

    best_acc, best_state, best_epoch, bad = -1.0, None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for idx in torch.randperm(len(x_tr), generator=gen).split(config.BATCH_SIZE):
            xb, yb = x_tr[idx].to(device), y_tr[idx].to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)
        eval_acc = float((predict(model, x[ev], device) == labels[ev]).mean())
        history.append({"epoch": epoch, "train_loss": total / n, "eval_acc": eval_acc})
        print(f"epoch {epoch:>2}  train_loss {total / n:.4f}  eval_acc {eval_acc:.4f}")
        if eval_acc > best_acc:
            best_acc, best_epoch, bad = eval_acc, epoch, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print(f"early stopping (eval_acc가 {config.PATIENCE} epoch 동안 안 오름)")
                break

    model.load_state_dict(best_state)
    print(f"최고 모델: epoch {best_epoch}, eval_acc {best_acc:.4f}")

    # test는 여기서 한 번만
    m = metrics(labels[te], predict(model, x[te], device))
    layer_weights = model.layer_weights().detach().cpu().numpy()
    report = format_report(m, layer_weights, int(te.sum()))
    print(report)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "labels": list(config.LABELS),
        "model_name": config.MODEL_NAME,
        "preprocess": config.preprocess_config(),
        "best_epoch": best_epoch,
        "best_eval_acc": best_acc,
    }, out_path)
    history_lines = ["", f"학습 기록 (최고 모델: epoch {best_epoch}):", "  epoch  train_loss  eval_acc"]
    history_lines += [f"  {h['epoch']:>5}  {h['train_loss']:>10.4f}  {h['eval_acc']:>8.4f}" for h in history]
    out_path.with_suffix(".report.txt").write_text(report + "\n" + "\n".join(history_lines) + "\n", encoding="utf-8")
    print(f"저장: {out_path}")
    return history, m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=config.UTT_CSV)
    ap.add_argument("--features", type=Path, default=config.FEATURES_NPY)
    ap.add_argument("--index", type=Path, default=config.FEATURES_INDEX)
    ap.add_argument("--out", type=Path, default=config.CHECKPOINT)
    args = ap.parse_args(argv)
    train(args.features, args.index, args.csv, args.out)


if __name__ == "__main__":
    main()
