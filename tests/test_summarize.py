import csv
import importlib.util
from pathlib import Path

import pytest

from emotion_scorer import config

SCRIPT = Path(__file__).resolve().parents[1] / "baselines" / "summarize.py"
spec = importlib.util.spec_from_file_location("summarize", SCRIPT)
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)


def write_scores(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file"] + [f"p_{label}" for label in config.LABELS] + ["pred"])
        w.writerows(rows)


def probs(pred):
    return [f"{0.6 if label == pred else 0.1:.6f}" for label in config.LABELS]


def test_summarize_table(tmp_path):
    csv_path = tmp_path / "scores.csv"
    write_scores(csv_path, [
        ["/x/wavs/0013/0013_happiness_001.wav", *probs("happy"), "happy"],
        ["/x/wavs/0013/0013_happiness_002.wav", *probs("surprise"), "surprise"],
        ["/x/wavs/0017/0017_anger_001.wav", *probs("angry"), "angry"],
    ])
    out = tmp_path / "summary.txt"
    summarize.main([str(csv_path), "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    row = next(line.split() for line in lines if line.startswith("0013     happy"))
    assert row[2:4] == ["2", "0.5000"]  # n, 정답률
    assert row[4:] == ["0.1000", "0.3500", "0.1000", "0.1000", "0.3500"]  # 평균 확률
    total = next(line.split() for line in lines if line.startswith("all      all"))
    assert total[2:4] == ["3", "0.6667"]


def test_summarize_rejects_unknown_file_name(tmp_path):
    csv_path = tmp_path / "scores.csv"
    write_scores(csv_path, [["/x/0013_000351.wav", *probs("happy"), "happy"]])
    with pytest.raises(ValueError, match="화자·감정"):
        summarize.main([str(csv_path), "--out", str(tmp_path / "s.txt")])
