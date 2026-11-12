"""음성 읽기 → 모노 → 16kHz → 앞뒤 무음 자르기 → 음량 맞추기.

모든 스크립트는 load_audio 하나로만 전처리한다.
"""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import config

FRAME_SEC = 0.025
HOP_SEC = 0.010
PEAK_LIMIT = 0.99
RMS_EPS = 1e-8  # 이보다 작으면 사실상 0 → 음량 조절 안 함


def trim_silence(wav: np.ndarray, sr: int) -> np.ndarray:
    """가장 큰 프레임보다 TRIM_TOP_DB 이상 작은 프레임을 무음으로 보고 앞뒤만 자른다."""
    frame, hop = int(FRAME_SEC * sr), int(HOP_SEC * sr)
    if len(wav) < frame:
        return wav
    frames = np.lib.stride_tricks.sliding_window_view(wav, frame)[::hop]  # [n_frames, frame]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    voiced = np.flatnonzero(db > db.max() - config.TRIM_TOP_DB)
    if len(voiced) == 0 or rms.max() < RMS_EPS:  # 전부 무음
        return wav
    pad = int(config.TRIM_PAD_SEC * sr)
    start = max(0, voiced[0] * hop - pad)
    end = min(len(wav), voiced[-1] * hop + frame + pad)
    if end - start < config.MIN_DURATION_SEC * sr:
        return wav
    return wav[start:end]


def normalize_loudness(wav: np.ndarray) -> np.ndarray:
    """RMS를 TARGET_RMS로 맞추고, 최대 진폭이 0.99를 넘으면 0.99로 줄인다."""
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if len(wav) else 0.0
    if rms < RMS_EPS:
        return wav
    wav = wav * (config.TARGET_RMS / rms)
    peak = float(np.abs(wav).max())
    if peak > PEAK_LIMIT:
        wav = wav * (PEAK_LIMIT / peak)
    return wav


def load_audio(path) -> np.ndarray:
    """음성 파일 → 16kHz 모노 float32 1차원 배열 (무음 자르기 + 음량 맞추기 적용)."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    wav = wav.mean(axis=1)  # 모노
    if sr != config.SAMPLE_RATE:
        g = gcd(sr, config.SAMPLE_RATE)
        wav = resample_poly(wav, config.SAMPLE_RATE // g, sr // g)
    wav = trim_silence(wav, config.SAMPLE_RATE)
    wav = normalize_loudness(wav)
    return np.ascontiguousarray(wav, dtype=np.float32)
