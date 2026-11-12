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

# 전처리 (audio.load_audio): 앞뒤 무음 자르기 + 음량 맞추기
TRIM_TOP_DB = 40  # 가장 큰 프레임보다 이만큼(dB) 이상 작으면 무음
TRIM_PAD_SEC = 0.05  # 자른 뒤 앞뒤로 남길 여유
MIN_DURATION_SEC = 0.1  # 자른 결과가 이보다 짧으면 자르지 않음
TARGET_RMS = 0.05


def preprocess_config() -> dict:
    """체크포인트에 저장하고 채점 때 비교하는 전처리 설정."""
    return {
        "trim_top_db": TRIM_TOP_DB,
        "trim_pad_sec": TRIM_PAD_SEC,
        "min_duration_sec": MIN_DURATION_SEC,
        "target_rms": TARGET_RMS,
    }


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
