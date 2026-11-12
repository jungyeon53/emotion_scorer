"""Hugging Face jspaulsen/esd parquet → wav 파일 + 발화 목록(utterances.csv).

출력: OUT/wavs/{speaker}/{utt_id}.wav, OUT/utterances.csv

실행:
    python -m emotion_scorer.build_manifest "data/train-*.parquet" --out data/esd
    python -m emotion_scorer.build_manifest data/train-00006-of-00007.parquet --out data/sample \
        --speakers 0019,0020 --per-split train=10,eval=4,test=4
"""


import argparse

import csv

import glob

import re

from collections import Counter

from pathlib import Path


import pyarrow.parquet as pq

import soundfile as sf


from . import config


EMOTION_MAP = {

    "neutral": "neutral",

    "happiness": "happy",

    "anger": "angry",

    "sadness": "sad",

    "surprise": "surprise",
}


BLOCK_ORDER = ["neutral", "angry", "happy", "sad", "surprise"]

SENTENCES_PER_EMOTION = 350

SPLITS = ["train", "eval", "test"]

UTT_RE = re.compile(r"^(\d{4})_(\d{6})$")

CSV_COLUMNS = ["utt_id", "speaker", "emotion", "sentence_id", "text", "split", "path", "duration", "gender"]

FULL_SPEAKERS = 10


def id_info(n: int) -> tuple:
    """ID 번호 → (감정 블록, sentence_id, split). ESD 공식 분할: 1~20 eval, 21~50 test, 51~350 train."""

    if not 1 <= n <= SENTENCES_PER_EMOTION * len(BLOCK_ORDER):

        raise ValueError(f"ID 번호 범위 밖: {n}")

    emotion = BLOCK_ORDER[(n - 1) // SENTENCES_PER_EMOTION]

    sentence_id = (n - 1) % SENTENCES_PER_EMOTION + 1

    split = "eval" if sentence_id <= 20 else "test" if sentence_id <= 50 else "train"

    return emotion, sentence_id, split


def make_row(audio_path: str, speaker_id: str, emotion_raw: str, transcript: str, gender: str) -> dict:
    """parquet 한 줄의 정보 → CSV 한 줄 (path, duration은 wav 저장 후 채움). 규칙에 어긋나면 에러."""

    utt_id = Path(audio_path).stem

    m = UTT_RE.match(utt_id)

    if not m:

        raise ValueError(f"utt_id 형식이 이상함: {audio_path!r}")

    if m.group(1) != speaker_id:

        raise ValueError(f"{utt_id}: 파일 이름의 화자 {m.group(1)} != speaker_id {speaker_id}")

    if emotion_raw not in EMOTION_MAP:

        raise ValueError(f"{utt_id}: 모르는 감정 {emotion_raw!r}")

    emotion = EMOTION_MAP[emotion_raw]

    block, sentence_id, split = id_info(int(m.group(2)))

    if block != emotion:

        raise ValueError(f"{utt_id}: ID 번호의 감정 블록 {block} != emotion 칸 {emotion} ({emotion_raw})")

    return {

        "utt_id": utt_id, "speaker": speaker_id, "emotion": emotion, "sentence_id": sentence_id,

        "text": transcript.strip(), "split": split, "gender": gender,
    }


def parse_per_split(s):

    if not s:

        return None

    limits = {}

    for part in s.split(","):

        k, v = part.split("=")

        if k.strip() not in SPLITS:

            raise ValueError(f"--per-split: 모르는 분할 {k!r} (train/eval/test)")

        limits[k.strip()] = int(v)

    return limits


def expand_inputs(patterns) -> list:

    files = sorted({f for p in patterns for f in (glob.glob(p) or [p])})

    for f in files:

        if not Path(f).is_file():

            raise FileNotFoundError(f)

    return files


def read_metadata(files, speakers) -> dict:
    """1차: 음성 바이트 없이 메타데이터만 읽고 검사. {utt_id: row}"""

    rows = {}

    cols = ["audio.path", "transcript", "emotion", "speaker_id", "gender", "language"]

    for f in files:

        for r in pq.read_table(f, columns=cols).to_pylist():

            if r["language"] != "en" or (speakers and r["speaker_id"] not in speakers):

                continue

            row = make_row(r["path"], r["speaker_id"], r["emotion"], r["transcript"], r["gender"])

            if row["utt_id"] in rows:

                raise ValueError(f"utt_id 중복: {row['utt_id']} ({f})")

            row["_file"] = f

            rows[row["utt_id"]] = row

    return rows


def select(rows: dict, limits) -> dict:
    """화자·감정·분할마다 sentence_id가 작은 순서로 N개."""

    if not limits:

        return rows

    picked, used = {}, Counter()

    for row in sorted(rows.values(), key=lambda r: (r["speaker"], r["emotion"], r["split"], r["sentence_id"])):

        key = (row["speaker"], row["emotion"], row["split"])

        if used[key] < limits.get(row["split"], 0):

            used[key] += 1

            picked[row["utt_id"]] = row

    return picked


def write_wavs(rows: dict, files, out_dir: Path):
    """2차: 고른 발화의 audio.bytes를 다시 인코딩하지 않고 그대로 저장."""

    for f in files:

        wanted = {u for u, r in rows.items() if r["_file"] == f}

        if not wanted:

            continue

        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["audio"]):

            for item in batch.column("audio").to_pylist():

                utt_id = Path(item["path"]).stem

                if utt_id not in wanted:

                    continue

                row = rows[utt_id]

                rel = Path("wavs") / row["speaker"] / f"{utt_id}.wav"

                (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)

                (out_dir / rel).write_bytes(item["bytes"])

                row["path"] = rel.as_posix()

                row["duration"] = f"{sf.info(str(out_dir / rel)).duration:.3f}"

    missing = [u for u, r in rows.items() if "path" not in r]

    if missing:

        raise RuntimeError(f"음성을 못 찾은 발화 {len(missing)}개, 예: {missing[:3]}")


def print_counts(rows: dict):

    counts = Counter((r["speaker"], r["emotion"], r["split"]) for r in rows.values())

    speakers = sorted({r["speaker"] for r in rows.values()})

    print(f"\n{'speaker':<8} {'emotion':<9} {'train':>6} {'eval':>5} {'test':>5} {'total':>6}")

    for spk in speakers:

        for emo in config.LABELS:

            c = [counts[(spk, emo, s)] for s in SPLITS]

            print(f"{spk:<8} {emo:<9} {c[0]:>6} {c[1]:>5} {c[2]:>5} {sum(c):>6}")

    tot = [sum(v for (_, _, s), v in counts.items() if s == split) for split in SPLITS]

    print(f"{'합계':<17}  {tot[0]:>6} {tot[1]:>5} {tot[2]:>5} {sum(tot):>6}")

    print(f"화자 {len(speakers)}명, 발화 {len(rows)}개")


def check_full(rows: dict):
    """옵션 없이 실행했을 때 ESD 영어 전체(10명 × 5감정 × 350 = 17,500)인지 확인. 다르면 경고만."""

    speakers = {r["speaker"] for r in rows.values()}

    per = Counter((r["speaker"], r["emotion"]) for r in rows.values())

    expected = FULL_SPEAKERS * len(config.LABELS) * SENTENCES_PER_EMOTION

    problems = []

    if len(speakers) != FULL_SPEAKERS:

        problems.append(f"영어 화자 {len(speakers)}명 (기대 {FULL_SPEAKERS}명)")

    short = sorted(f"{s}/{e}={per[(s, e)]}" for s in speakers for e in config.LABELS

                   if per[(s, e)] != SENTENCES_PER_EMOTION)

    if short:

        problems.append(f"350개가 아닌 화자·감정: {', '.join(short)}")

    if len(rows) != expected:

        problems.append(f"총 {len(rows)}개 (기대 {expected}개)")

    if problems:

        print("\n경고: ESD 영어 전체가 아님 (parquet 일부만 넣었으면 정상)")

        for p in problems:

            print(f"  - {p}")

    else:

        print(f"\n확인: 영어 화자 {FULL_SPEAKERS}명, 화자·감정마다 {SENTENCES_PER_EMOTION}개, 총 {expected}개")


def main(argv=None):

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("parquet", nargs="+", help="parquet 파일 (glob 가능, 따옴표로 감싸기)")

    ap.add_argument("--out", type=Path, required=True, help="출력 폴더")

    ap.add_argument("--speakers", default="", help="예: 0019,0020")

    ap.add_argument("--per-split", default="", help="예: train=10,eval=4,test=4")

    args = ap.parse_args(argv)


    files = expand_inputs(args.parquet)

    speakers = {s.strip() for s in args.speakers.split(",") if s.strip()}

    limits = parse_per_split(args.per_split)

    print(f"parquet {len(files)}개: {', '.join(Path(f).name for f in files)}")


    rows = select(read_metadata(files, speakers), limits)

    if not rows:

        raise SystemExit("조건에 맞는 발화가 없음")

    if speakers - {r["speaker"] for r in rows.values()}:

        print(f"경고: 데이터에 없는 화자 {sorted(speakers - {r['speaker'] for r in rows.values()})}")

    args.out.mkdir(parents=True, exist_ok=True)

    write_wavs(rows, files, args.out)


    csv_path = args.out / "utterances.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")

        w.writeheader()

        w.writerows(sorted(rows.values(), key=lambda r: r["utt_id"]))


    print_counts(rows)

    if not speakers and not limits:

        check_full(rows)

    print(f"\n저장: {csv_path}")


if __name__ == "__main__":

    main()
