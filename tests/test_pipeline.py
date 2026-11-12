"""ESD 없이 하는 검증 4가지. 실행: pytest -s tests/test_pipeline.py
(WavLM 가중치는 처음 한 번 Hugging Face에서 내려받음)
"""

import csv

import numpy as np
import pytest
import soundfile as sf
import torch

from emotion_scorer import config, extract_features, score, train_classifier
from emotion_scorer.audio import FRAME_SEC, load_audio

FIELDS = extract_features.CSV_COLUMNS


def sine(sr, seconds, freq, channels=1):
    t = np.arange(int(sr * seconds)) / sr
    wav = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([wav] * channels, axis=1) if channels > 1 else wav


@pytest.fixture(scope="module")
def wavs(tmp_path_factory):
    d = tmp_path_factory.mktemp("wav")
    files = {
        "sine16k": (d / "sine_16k.wav", 16000, sine(16000, 1.0, 220)),
        "sine24k": (d / "sine_24k.wav", 24000, sine(24000, 1.5, 440)),
        "stereo24k": (d / "stereo_24k.wav", 24000, sine(24000, 0.8, 330, channels=2)),
    }
    for path, sr, data in files.values():
        sf.write(path, data, sr)
    return {k: v[0] for k, v in files.items()}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def extracted(wavs, tmp_path_factory):
    d = tmp_path_factory.mktemp("feat")
    rows = [
        dict(utt_id=f"u{i}", speaker="0011", emotion=emo, sentence_id=str(i), text="hello",
             split="train", path=str(p), duration="1.0")
        for i, (p, emo) in enumerate(zip(wavs.values(), ["Neutral", "Happy", "Sad"]))
    ]
    write_csv(d / "utts.csv", rows)
    extract_features.main(["--csv", str(d / "utts.csv"), "--out", str(d / "f.npy"), "--index", str(d / "idx.txt")])
    return d, rows


@pytest.fixture(scope="module")
def fake_trained(tmp_path_factory):
    """무작위 가짜 특징 + 가짜 라벨로 학습한 체크포인트."""
    d = tmp_path_factory.mktemp("train")
    rng = np.random.default_rng(0)
    n = 300
    feats = rng.normal(size=(n, config.NUM_LAYERS, config.HIDDEN_SIZE)).astype(np.float16)
    np.save(d / "f.npy", feats)
    splits = ["train"] * 200 + ["eval"] * 50 + ["test"] * 50
    rows = [
        dict(utt_id=f"u{i:03d}", speaker="0011", emotion=config.LABELS[rng.integers(5)].capitalize(),
             sentence_id=str(i), text="x", split=s, path="none.wav", duration="1.0")
        for i, s in enumerate(splits)
    ]
    write_csv(d / "utts.csv", rows)
    (d / "idx.txt").write_text("".join(r["utt_id"] + "\n" for r in rows))
    history, m = train_classifier.train(d / "f.npy", d / "idx.txt", d / "utts.csv", d / "clf.pt")
    return d / "clf.pt", history, m


def test_1_extract_shape(extracted):
    d, rows = extracted
    feats = np.load(d / "f.npy")
    print(f"\n[1] features.npy shape={feats.shape} dtype={feats.dtype}")
    assert feats.shape == (len(rows), 13, 768)
    assert feats.dtype == np.float16
    assert np.isfinite(feats).all()
    assert (d / "idx.txt").read_text().split() == [r["utt_id"] for r in rows]


def test_2_train_loss_decreases(fake_trained):
    ckpt_path, history, m = fake_trained
    losses = [h["train_loss"] for h in history]
    print(f"\n[2] epochs={len(losses)} train_loss 처음 {losses[0]:.4f} → 마지막 {losses[-1]:.4f}")
    assert losses[-1] < losses[0]
    ckpt = torch.load(ckpt_path, weights_only=True)
    assert ckpt["mean"].shape == ckpt["std"].shape == (13, 768)
    assert ckpt["labels"] == config.LABELS


def test_3_score_24k(wavs, fake_trained, tmp_path):
    ckpt_path, _, _ = fake_trained
    wav24 = wavs["sine24k"]
    assert sf.info(wav24).samplerate == 24000
    resampled = load_audio(wav24)
    print(f"\n[3] 24kHz 1.5초 {sf.info(wav24).frames} 샘플 → 16kHz {len(resampled)} 샘플")
    assert len(resampled) == 24000

    out = tmp_path / "scores.csv"
    score.main([str(wav24), str(wavs["stereo24k"]), "--checkpoint", str(ckpt_path), "--out", str(out)])
    with open(out) as f:
        rows = list(csv.DictReader(f))
    print(open(out).read())
    assert list(rows[0].keys()) == ["file", "p_neutral", "p_happy", "p_angry", "p_sad", "p_surprise", "pred"]
    for r in rows:
        probs = [float(r[f"p_{n}"]) for n in config.LABELS]
        print(f"    합 = {sum(probs):.6f}")
        assert len(probs) == 5 and abs(sum(probs) - 1) < 1e-4
        assert r["pred"] == config.LABELS[int(np.argmax(probs))]


def test_4_same_features_both_paths(wavs, extracted, fake_trained):
    """extract_features 경로로 저장한 특징 == score 경로에서 계산한 특징 (정규화 전)."""
    d, rows = extracted
    saved = np.load(d / "f.npy")
    device = config.get_device()
    fe, wavlm = extract_features.load_wavlm(device)
    for i, r in enumerate(rows):
        live = score.features_for_file(r["path"], fe, wavlm, device)  # score가 쓰는 함수
        diff = np.abs(saved[i].astype(np.float32) - live.astype(np.float32)).max()
        print(f"\n[4] {r['path'].split('/')[-1]}: 최대 차이 {diff}")
        assert diff == 0.0

    # 정규화까지 포함한 score 결과 == 저장 특징에 체크포인트 정규화를 직접 적용한 결과
    ckpt_path, _, _ = fake_trained
    clf, mean, std = score.load_classifier(ckpt_path, device)
    with torch.no_grad():
        x = (torch.from_numpy(saved).float().to(device) - mean) / std
        expected = torch.softmax(clf(x), dim=1).cpu().numpy()
    got = score.score_files([r["path"] for r in rows], ckpt_path, device)
    got = np.array([[g[f"p_{n}"] for n in config.LABELS] for g in got])
    print(f"[4] score 확률 vs 저장 특징으로 계산한 확률 최대 차이 {np.abs(got - expected).max():.2e}")
    assert np.allclose(got, expected, atol=1e-5)


# ---- 전처리 (무음 자르기, 음량 맞추기) ----

SR = config.SAMPLE_RATE


def noise(seconds, rng, amp=1e-4):
    """실제 녹음처럼 완전한 0이 아닌 아주 작은 잡음."""
    return (amp * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def test_5_trim_silence(tmp_path):
    rng = np.random.default_rng(1)
    tone = sine(SR, 1.0, 300)
    long_sil = np.concatenate([noise(0.5, rng), tone, noise(1.0, rng)])
    short_sil = np.concatenate([noise(0.1, rng), tone, noise(0.1, rng)])
    sf.write(tmp_path / "long.wav", long_sil, SR)
    sf.write(tmp_path / "short.wav", short_sil, SR)
    a, b = load_audio(tmp_path / "long.wav"), load_audio(tmp_path / "short.wav")
    expected = 1.0 + 2 * config.TRIM_PAD_SEC
    print(f"\n[5] 입력 {len(long_sil) / SR:.3f}s / {len(short_sil) / SR:.3f}s → "
          f"출력 {len(a) / SR:.3f}s / {len(b) / SR:.3f}s (기대 약 {expected:.3f}s)")
    assert abs(len(a) - len(b)) <= 0.020 * SR
    for out in (a, b):
        # 소리 경계에 걸친 프레임(25ms)까지 남기므로 앞뒤로 최대 1프레임씩 더 길 수 있음
        assert expected - 0.01 <= len(out) / SR <= expected + 2 * FRAME_SEC


def test_6_loudness(tmp_path):
    for amp in (0.05, 0.5):
        sf.write(tmp_path / f"a{amp}.wav", (amp / 0.3) * sine(SR, 1.0, 300), SR)
    rms = [float(np.sqrt(np.mean(load_audio(tmp_path / f"a{amp}.wav").astype(np.float64) ** 2))) for amp in (0.05, 0.5)]
    print(f"\n[6] 진폭 0.05 → RMS {rms[0]:.5f}, 진폭 0.5 → RMS {rms[1]:.5f} (목표 {config.TARGET_RMS})")
    for r in rms:
        assert abs(r - config.TARGET_RMS) / config.TARGET_RMS < 0.01


def test_7_safety(tmp_path):
    rng = np.random.default_rng(2)
    cases = {
        "silent_noise": noise(1.0, rng),
        "digital_zero": np.zeros(SR, dtype=np.float32),
        "short_0.05s": sine(SR, 0.05, 300),
    }
    device = config.get_device()
    fe, wavlm = extract_features.load_wavlm(device)
    print()
    for name, data in cases.items():
        sf.write(tmp_path / f"{name}.wav", data, SR)
        out = load_audio(tmp_path / f"{name}.wav")
        feat = extract_features.features_for_file(tmp_path / f"{name}.wav", fe, wavlm, device)
        print(f"[7] {name}: 입력 {len(data)} → 출력 {len(out)} 샘플, 특징 {feat.shape}, 유한값 {np.isfinite(feat).all()}")
        assert np.isfinite(out).all() and len(out) > 0
        assert feat.shape == (13, 768) and np.isfinite(feat).all()
    # 짧은 파일은 자르지 않고 원본 길이 그대로
    assert len(load_audio(tmp_path / "short_0.05s.wav")) == len(cases["short_0.05s"])


def test_8_checkpoint_preprocess_mismatch(fake_trained, tmp_path):
    ckpt_path, _, _ = fake_trained
    ckpt = torch.load(ckpt_path, weights_only=True)
    assert ckpt["preprocess"] == config.preprocess_config()
    score.load_classifier(ckpt_path, "cpu")  # 원본은 통과

    ckpt["preprocess"]["target_rms"] = 0.1
    bad = tmp_path / "bad.pt"
    torch.save(ckpt, bad)
    with pytest.raises(ValueError, match="전처리") as e:
        score.load_classifier(bad, "cpu")
    print(f"\n[8] 예상대로 에러: {e.value}")
