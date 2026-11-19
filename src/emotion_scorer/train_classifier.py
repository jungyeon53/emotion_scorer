"""특징 파일 + 발화 목록 → 층 가중합 분류기.

train으로 학습, eval 정확도로 early stopping/최고 모델 선택, test는 마지막에 한 번만 평가.
holdout split(처음 보는 화자)이 있으면 test처럼 마지막에 한 번만 평가하고, 학습·early stopping·
최고 모델 선택·정규화 값 계산에는 쓰지 않는다.
CSV와 같은 폴더에 meta.json(build_manifest --revision이 만든 데이터셋 버전 기록)이 있으면
그 내용을 체크포인트의 dataset_meta에 같이 저장한다 (없으면 None).

실행:
    python -m emotion_scorer.train_classifier --csv data/utterances.csv \
        --features features/features.npy --index features/features_index.txt \
        --out checkpoints/classifier.pt
"""

import argparse
import copy
import json
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


def read_dataset_meta(csv_path) -> dict | None:
    """CSV 옆 meta.json → dict. 없으면 None."""
    meta_path = Path(csv_path).parent / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(features_npy, index_path, csv_path):
    """특징 [N,13,768] float32, 라벨 [N], split [N], 화자 [N]을 utt_id로 맞춰서 돌려준다."""
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
    speakers = np.array([rows[u]["speaker"].strip() for u in utt_ids])
    return feats, labels, splits, speakers


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


def format_split_report(name: str, m: dict, n: int) -> str:
    """한 split의 정확도, macro F1, 감정별 정답률, 혼동 행렬 (test와 holdout이 같은 형식을 쓴다)."""
    lines = [f"[{name}] 발화 {n}개", f"전체 정확도: {m['accuracy']:.4f}", f"macro F1: {m['macro_f1']:.4f}", "감정별 정답률:"]
    lines += [f"  {label:<9} {r:.4f}" for label, r in zip(config.LABELS, m["recall"])]
    lines.append("혼동 행렬 (행=정답, 열=예측):")
    lines.append(" " * 10 + "".join(f"{label[:7]:>8}" for label in config.LABELS))
    lines += [f"  {label:<8}" + "".join(f"{v:>8d}" for v in row) for label, row in zip(config.LABELS, m["cm"])]
    return "\n".join(lines)


def format_report(m: dict, layer_weights: np.ndarray, n_test: int) -> str:
    lines = [format_split_report("test", m, n_test), "층 가중치 (0=CNN 출력, 1~12=트랜스포머 층):"]
    lines += [f"  layer {i:>2}: {w:.4f}" for i, w in enumerate(layer_weights)]
    return "\n".join(lines)


def per_speaker_accuracy(y_true, y_pred, speakers) -> list[tuple[str, int, float]]:
    """[(화자, 발화 수, 정확도)] 화자 순서대로."""
    return [(s, int((speakers == s).sum()), float((y_pred[speakers == s] == y_true[speakers == s]).mean()))
            for s in sorted(set(speakers.tolist()))]


def format_speaker_report(test_m: dict, holdout_m: dict | None, by_speaker: dict) -> str:
    """test vs holdout 요약 한 줄 + 화자별 정확도. holdout_m이 None이면 요약 줄은 생략."""
    lines = [""]
    if holdout_m is not None:
        gap = test_m["accuracy"] - holdout_m["accuracy"]
        lines.append(f"[요약] test 정확도 {test_m['accuracy']:.4f} | holdout 정확도 {holdout_m['accuracy']:.4f}"
                     f" | 차이(test - holdout) {gap:+.4f}")
    lines.append("화자별 정확도:")
    for split, rows in by_speaker.items():
        lines += [f"  {split:<8} {spk:<6} 발화 {n:>5}개  정확도 {acc:.4f}" for spk, n, acc in rows]
    return "\n".join(lines)


def train(features_npy, index_path, csv_path, out_path, max_epochs=config.MAX_EPOCHS):
    set_seed(config.SEED)
    device = config.get_device()
    feats, labels, splits, speakers = load_dataset(features_npy, index_path, csv_path)
    dataset_meta = read_dataset_meta(csv_path)
    if dataset_meta is None:
        print("경고: CSV 옆에 meta.json이 없어 데이터셋 버전을 체크포인트에 기록하지 않음")
    else:
        print(f"데이터셋: {dataset_meta.get('dataset')} revision {dataset_meta.get('revision')}")
    for s in ("train", "eval", "test"):
        if not (splits == s).any():
            raise ValueError(f"split '{s}' 발화가 없음")

    tr, ev, te = splits == "train", splits == "eval", splits == "test"
    ho = splits == "holdout"  # 선택: 없으면 holdout 평가를 건너뜀
    train_speakers = sorted(set(speakers[tr].tolist()))
    holdout_speakers = sorted(set(speakers[ho].tolist()))
    leaked = set(holdout_speakers) & set(speakers[tr | ev | te].tolist())
    if leaked:
        raise ValueError(f"holdout 화자 {sorted(leaked)}가 train/eval/test에도 있음 (홀드아웃이 아님)")
    # 정규화 값은 train 특징으로만 계산 (층별·차원별 [13, 768]). eval/test/holdout은 쓰지 않는다.
    mean = feats[tr].mean(axis=0)
    std = feats[tr].std(axis=0) + 1e-5
    x = torch.from_numpy((feats - mean) / std)
    y = torch.from_numpy(labels)
    print(f"device={device}, train {tr.sum()} / eval {ev.sum()} / test {te.sum()} / holdout {ho.sum()}")

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

    # test와 holdout은 최고 모델 선택이 끝난 뒤 여기서 한 번만
    pred_te = predict(model, x[te], device)
    m = metrics(labels[te], pred_te)
    layer_weights = model.layer_weights().detach().cpu().numpy()
    report = format_report(m, layer_weights, int(te.sum()))
    by_speaker = {"test": per_speaker_accuracy(labels[te], pred_te, speakers[te])}
    m_ho = None
    if ho.any():
        pred_ho = predict(model, x[ho], device)
        m_ho = metrics(labels[ho], pred_ho)
        by_speaker["holdout"] = per_speaker_accuracy(labels[ho], pred_ho, speakers[ho])
        report += "\n\n" + format_split_report("holdout", m_ho, int(ho.sum()))
    report += "\n" + format_speaker_report(m, m_ho, by_speaker)
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
        "train_speakers": train_speakers,
        "holdout_speakers": holdout_speakers,
        "dataset_meta": dataset_meta,
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
