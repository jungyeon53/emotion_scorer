"""무음 자르기 점검: 발화마다 원래 길이와 audio.load_audio를 거친 뒤 길이(16kHz 기준)를 비교한다.

실행:
    python -m emotion_scorer.check_trim --csv data/sample/utterances.csv
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config
from .audio import load_audio
from .extract_features import read_utterances, resolve_path

WARN_RATIO = 0.5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=config.UTT_CSV)
    args = ap.parse_args(argv)

    rows = read_utterances(args.csv)
    base = args.csv.resolve().parent
    results = []
    for r in rows:
        path = resolve_path(r["path"], base)
        info = sf.info(str(path))
        before = info.frames * config.SAMPLE_RATE / info.samplerate  # 16kHz 기준 샘플 수
        after = len(load_audio(path))
        results.append((r["utt_id"], before / config.SAMPLE_RATE, after / config.SAMPLE_RATE, after / before))

    ratios = np.array([x[3] for x in results])
    removed = np.array([x[1] - x[2] for x in results])
    print(f"발화 {len(results)}개")
    print(f"남은 길이 비율  최소 {ratios.min():.3f}  평균 {ratios.mean():.3f}  최대 {ratios.max():.3f}")
    print(f"줄어든 비율    최소 {1 - ratios.max():.3f}  평균 {1 - ratios.mean():.3f}  최대 {1 - ratios.min():.3f}")
    print(f"잘린 길이(초)  최소 {removed.min():.3f}  평균 {removed.mean():.3f}  최대 {removed.max():.3f}")

    short = [x for x in results if x[3] < WARN_RATIO]
    print(f"\n원래 길이의 {WARN_RATIO:.0%} 미만으로 줄어든 파일: {len(short)}개")
    for utt_id, b, a, ratio in sorted(short, key=lambda x: x[3]):
        print(f"  {utt_id}  {b:.3f}s → {a:.3f}s ({ratio:.1%})")


if __name__ == "__main__":
    main()
