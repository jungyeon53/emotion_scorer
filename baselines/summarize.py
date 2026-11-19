"""score 결과 CSV(실제 녹음) → 화자 × 감정별 정답률과 평균 확률 표.

정답 감정은 파일 이름(utt_id = {speaker}_{parquet 감정 이름}_{sentence_id:03d})에서 읽는다.

실행:
    python baselines/summarize.py baselines/holdout_real.csv --out baselines/holdout_real_summary.txt
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from emotion_scorer import config
from emotion_scorer.build_manifest import EMOTION_MAP

UTT_RE = re.compile(r"^(\d{4})_([a-z]+)_(\d{3})$")


def read_scores(csv_path) -> list[dict]:
    """[{speaker, emotion(정답), pred, probs: {label: p}}]. 파일 이름 형식이 다르면 에러."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stem = Path(r["file"]).stem
            m = UTT_RE.match(stem)
            if not m or m.group(2) not in EMOTION_MAP:
                raise ValueError(f"파일 이름에서 화자·감정을 읽을 수 없음: {r['file']}")
            if r["pred"] not in config.LABELS:
                raise ValueError(f"모르는 pred {r['pred']!r}: {r['file']}")
            rows.append({"speaker": m.group(1), "emotion": EMOTION_MAP[m.group(2)], "pred": r["pred"],
                         "probs": {label: float(r[f"p_{label}"]) for label in config.LABELS}})
    if not rows:
        raise ValueError(f"{csv_path}: 행이 없음")
    return rows


def summarize(rows) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["speaker"], r["emotion"])].append(r)
        groups[(r["speaker"], "all")].append(r)
        groups[("all", r["emotion"])].append(r)
        groups[("all", "all")].append(r)
    speakers = sorted({r["speaker"] for r in rows}) + ["all"]
    emotions = list(config.LABELS) + ["all"]

    head = f"{'speaker':<8} {'emotion':<9} {'n':>5} {'acc':>7}  " + " ".join(f"{'p_' + l[:6]:>9}" for l in config.LABELS)
    lines = [f"발화 {len(rows)}개, 화자 {', '.join(speakers[:-1])}",
             "acc = pred가 정답 감정과 같은 비율, p_* = 그 그룹의 평균 확률, all = 합계", "", head]
    for spk in speakers:
        for emo in emotions:
            g = groups.get((spk, emo))
            if not g:
                continue
            acc = sum(r["pred"] == r["emotion"] for r in g) / len(g)
            means = [sum(r["probs"][label] for r in g) / len(g) for label in config.LABELS]
            lines.append(f"{spk:<8} {emo:<9} {len(g):>5} {acc:>7.4f}  " + " ".join(f"{p:>9.4f}" for p in means))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="?", default=Path("baselines/holdout_real.csv"))
    ap.add_argument("--out", type=Path, default=Path("baselines/holdout_real_summary.txt"))
    args = ap.parse_args(argv)
    text = summarize(read_scores(args.csv))
    print(text)
    args.out.write_text(f"원본: {args.csv.name}\n" + text, encoding="utf-8")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
