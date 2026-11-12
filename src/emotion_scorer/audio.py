"""음성 읽기 → 모노 → 16kHz. 모든 스크립트는 이 함수 하나로만 전처리한다."""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import SAMPLE_RATE


def load_audio(path) -> np.ndarray:
    """음성 파일을 읽어 16kHz 모노 float32 1차원 배열로 돌려준다."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    wav = wav.mean(axis=1)  # 모노
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g)
    return np.ascontiguousarray(wav, dtype=np.float32)
