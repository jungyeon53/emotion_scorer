"""경로, 모델 이름, 라벨 순서, 학습 설정."""

from pathlib import Path

import torch

# 경로 (CLI 인자로 덮어쓸 수 있음)
ROOT = Path(__file__).resolve().parents[2]
UTT_CSV = ROOT / "data" / "utterances.csv"
FEATURES_NPY = ROOT / "features" / "features.npy"
FEATURES_INDEX = ROOT / "features" / "features_index.txt"
CHECKPOINT = ROOT / "checkpoints" / "classifier.pt"

# 모델
MODEL_NAME = "microsoft/wavlm-base-plus"
SAMPLE_RATE = 16000
NUM_LAYERS = 13  # CNN 출력(임베딩) 1 + 트랜스포머 12
HIDDEN_SIZE = 768

# 라벨 순서 (고정, 바꾸지 말 것)
LABELS = ["neutral", "happy", "angry", "sad", "surprise"]

# 학습 설정
SEED = 42
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 64


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
