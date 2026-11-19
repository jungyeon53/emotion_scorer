"""Hugging Face jspaulsen/esd parquet → wav 파일 + 발화 목록(utterances.csv).

출력: OUT/wavs/{speaker}/{utt_id}.wav, OUT/utterances.csv, (--revision을 주면) OUT/meta.json

실행 (parquet 파일은 항상 전부 넣는다. 아래 "sentence_id와 utt_id" 참고):
    python -m emotion_scorer.build_manifest "data/train-*.parquet" --out data/esd --revision <커밋 해시>
    python -m emotion_scorer.build_manifest "data/train-*.parquet" --out data/sample \
        --speakers 0019,0020 --per-split train=10,eval=4,test=4

화자 홀드아웃 (처음 보는 화자에 대한 성능 확인용):
    python -m emotion_scorer.build_manifest "data/train-*.parquet" --list-speakers
    python -m emotion_scorer.build_manifest "data/train-*.parquet" --out data/esd \
        --holdout-speakers 0011,0016

--holdout-speakers로 지정한 화자의 발화는 sentence_id와 상관없이 전부 split="holdout"이 되고,
나머지 화자는 sentence_id 분할(1~20 eval, 21~50 test, 51~350 train)을 그대로 따른다.

sentence_id와 utt_id:
    이 parquet에는 원본 파일 이름이 없다 (audio.path가 전부 비어 있음). 그래서 sentence_id는
    (화자, 감정)마다 350행씩 이어진 블록 안의 순서(1~350)로 정한다. ESD 공식 ID와 일치한다는 보장은 없다.
    utt_id도 공식 형식(0011_000351)과 헷갈리지 않도록 {speaker}_{parquet 감정 이름}_{sentence_id:03d}
    (예: 0011_anger_001)로 만든다.
    이 방식은 parquet의 행 순서에 기대므로, 매번 영어 화자 전체를 기준으로 행 순서를 검사하고
    하나라도 깨지면 에러를 낸다. --speakers나 --per-split로 일부만 뽑을 때도 뽑기 전에 전체를 검사한다.
    블록이 parquet 파일 경계에 걸쳐 있으므로 파일 일부만 넣으면 이 검사에서 에러가 난다.
"""


import argparse

import csv

import difflib

import glob

import json

import re

from collections import Counter

from datetime import datetime

from pathlib import Path


import pyarrow.parquet as pq

import soundfile as sf


from . import config


DATASET = "jspaulsen/esd"


EMOTION_MAP = {

    "neutral": "neutral",

    "happiness": "happy",

    "anger": "angry",

    "sadness": "sad",

    "surprise": "surprise",
}


SENTENCES_PER_EMOTION = 350

SPLITS = ["train", "eval", "test"]

HOLDOUT = "holdout"

ALL_SPLITS = SPLITS + [HOLDOUT]

SPEAKER_RE = re.compile(r"^\d{4}$")

# jspaulsen/esd parquet에서 실제로 확인한 값 (7개 파일, 35,000행)

LANGUAGES = {"en", "zh"}

GENDERS = {"male", "female"}

# 위치별 문장 검사: 다수결 문장과 표기가 다른 행의 유사도가 이보다 낮으면 다른 문장으로 본다.
# 실제 데이터에서 표기가 다른 행 267개의 유사도는 최소 0.75 (예: "piglet" / "piglit").

TEXT_SIMILARITY_MIN = 0.6

META_COLUMNS = ["transcript", "emotion", "speaker_id", "gender", "language"]

CSV_COLUMNS = ["utt_id", "speaker", "emotion", "sentence_id", "text", "split", "path", "duration", "gender"]

FULL_SPEAKERS = 10


def split_of(sentence_id: int) -> str:
    """sentence_id → split. 1~20 eval, 21~50 test, 51~350 train."""

    if not 1 <= sentence_id <= SENTENCES_PER_EMOTION:

        raise ValueError(f"sentence_id 범위 밖: {sentence_id}")

    return "eval" if sentence_id <= 20 else "test" if sentence_id <= 50 else "train"


def make_utt_id(speaker: str, emotion_raw: str, sentence_id: int) -> str:
    """자체 번호라는 게 드러나는 형식. 예: 0011_anger_001 (ESD 공식 0011_000351 형식이 아님)."""

    return f"{speaker}_{emotion_raw}_{sentence_id:03d}"


def normalize_text(text: str) -> str:
    """위치별 문장 비교용: 소문자, 영문·숫자 외는 공백, 공백 정리."""

    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def parse_per_split(s):

    if not s:

        return None

    limits = {}

    for part in s.split(","):

        k, v = part.split("=")

        if k.strip() not in ALL_SPLITS:

            raise ValueError(f"--per-split: 모르는 분할 {k!r} (train/eval/test/holdout)")

        limits[k.strip()] = int(v)

    return limits


def parse_speakers(s: str) -> set:

    return {x.strip() for x in s.split(",") if x.strip()}


def expand_inputs(patterns) -> list:

    files = sorted({f for p in patterns for f in (glob.glob(p) or [p])})

    for f in files:

        if not Path(f).is_file():

            raise FileNotFoundError(f)

    return files


def check_speaker_values(f, r):
    """language, speaker_id, gender 칸 검사 (영어 행만 speaker_id·gender까지). 예상과 다르면 에러."""

    if r["language"] not in LANGUAGES:

        raise ValueError(f"{f}: 모르는 language {r['language']!r} (기대 {sorted(LANGUAGES)})")

    if r["language"] != "en":

        return

    if not SPEAKER_RE.match(r["speaker_id"] or ""):

        raise ValueError(f"{f}: speaker_id 형식이 이상함 {r['speaker_id']!r} (기대 4자리 숫자)")

    if r["gender"] not in GENDERS:

        raise ValueError(f"{f}: 화자 {r['speaker_id']}의 모르는 gender {r['gender']!r}")


def read_blocks(files) -> list:
    """1차: 음성 바이트 없이 메타데이터를 parquet 순서대로 읽어 영어 (화자, 감정) 블록 목록을 만든다.

    블록 = 같은 (language, 화자, 감정)이 끊기지 않고 이어진 행들 (파일 경계를 넘어 이어질 수 있음).
    중간에 다른 행(중국어 행 포함)이 끼면 블록이 끊긴다. 칸 값이 예상과 다르면 에러.
    각 행에는 음성을 다시 찾기 위한 _file(parquet 경로), _index(파일 안 행 번호)를 붙인다.
    """

    blocks, prev = [], None

    for f in files:

        for i, r in enumerate(pq.read_table(f, columns=META_COLUMNS).to_pylist()):

            check_speaker_values(f, r)

            if r["language"] != "en":

                prev = None

                continue

            if r["emotion"] not in EMOTION_MAP:

                raise ValueError(f"{f} {i}번째 행: 모르는 감정 {r['emotion']!r} (기대 {sorted(EMOTION_MAP)})")

            if not isinstance(r["transcript"], str) or not r["transcript"].strip():

                raise ValueError(f"{f} {i}번째 행: transcript가 비어 있음 (화자 {r['speaker_id']}, {r['emotion']})")

            key = (r["speaker_id"], r["emotion"])

            if key != prev:

                blocks.append({"speaker": key[0], "emotion_raw": key[1], "rows": []})

                prev = key

            blocks[-1]["rows"].append({**r, "_file": f, "_index": i})

    return blocks


def check_blocks(blocks):
    """행 순서 검사 (영어 화자 전체 기준). 하나라도 깨지면 에러.

    1. 블록 안에서 화자·감정이 섞이지 않는지: 같은 (화자, 감정)이 두 번 이상 끊겨 나오면 에러
    2. (화자, 감정)마다 350행 연속 블록인지, 화자마다 5감정이 모두 있는지
    3. 위치별 문장이 다른 블록과 일치하는지: i번째 문장을 모든 블록의 i번째 다수결 문장과 비교
    """

    if not blocks:

        raise ValueError("영어 발화가 없음")

    seen = Counter((b["speaker"], b["emotion_raw"]) for b in blocks)

    broken = sorted(f"{s}/{e} {n}조각" for (s, e), n in seen.items() if n > 1)

    if broken:

        raise ValueError(f"블록이 연속되지 않음 (다른 화자·감정 행이 중간에 섞여 있음): {', '.join(broken)}")

    wrong = [f"{b['speaker']}/{b['emotion_raw']}={len(b['rows'])}" for b in blocks
             if len(b["rows"]) != SENTENCES_PER_EMOTION]

    if wrong:

        raise ValueError(f"{SENTENCES_PER_EMOTION}행이 아닌 블록: {', '.join(wrong)} "
                         f"(parquet 파일을 일부만 넣었으면 전부 넣어야 함)")

    emotions = {}

    for b in blocks:

        emotions.setdefault(b["speaker"], set()).add(b["emotion_raw"])

    lacking = {s: sorted(set(EMOTION_MAP) - e) for s, e in emotions.items() if e != set(EMOTION_MAP)}

    if lacking:

        raise ValueError(f"5감정 블록이 다 없는 화자: {lacking}")

    check_positions(blocks)


def check_positions(blocks):
    """위치별 문장 검사. 다수결과 표기만 다른 행(유사도 >= TEXT_SIMILARITY_MIN)은 통과, 그보다 다르면 에러."""

    bad, variants, min_sim = [], 0, 1.0

    for i in range(SENTENCES_PER_EMOTION):

        texts = [normalize_text(b["rows"][i]["transcript"]) for b in blocks]

        major = Counter(texts).most_common(1)[0][0]

        for b, t in zip(blocks, texts):

            if t == major:

                continue

            sim = difflib.SequenceMatcher(None, t, major).ratio()

            variants += 1

            min_sim = min(min_sim, sim)

            if sim < TEXT_SIMILARITY_MIN:

                bad.append(f"{b['speaker']}/{b['emotion_raw']} {i + 1}번째 (유사도 {sim:.2f}): "
                           f"{b['rows'][i]['transcript']!r} | 다수결 {major!r}")

    if bad:

        raise ValueError(f"위치별 문장이 다른 블록과 다름 {len(bad)}개 (기준 유사도 {TEXT_SIMILARITY_MIN}), 예:\n  "
                         + "\n  ".join(bad[:5]))

    print(f"행 순서 검사 통과: 블록 {len(blocks)}개 × {SENTENCES_PER_EMOTION}행, "
          f"위치별 문장이 표기만 다른 행 {variants}개" + (f" (최소 유사도 {min_sim:.2f})" if variants else ""))


def make_rows(blocks) -> dict:
    """블록 → {utt_id: row}. text는 각 행의 실제 transcript (다수결 문장으로 바꾸지 않음, 앞뒤 공백만 제거)."""

    rows = {}

    for b in blocks:

        for sentence_id, r in enumerate(b["rows"], start=1):

            utt_id = make_utt_id(b["speaker"], b["emotion_raw"], sentence_id)

            rows[utt_id] = {

                "utt_id": utt_id, "speaker": b["speaker"], "emotion": EMOTION_MAP[b["emotion_raw"]],

                "sentence_id": sentence_id, "text": r["transcript"].strip(), "split": split_of(sentence_id),

                "gender": r["gender"], "_emotion_raw": b["emotion_raw"], "_file": r["_file"], "_index": r["_index"],
            }

    return rows


def assign_holdout(rows: dict, holdout: set) -> dict:
    """홀드아웃 화자의 발화는 sentence_id와 상관없이 전부 split="holdout". 데이터에 없는 화자면 에러."""

    if not holdout:

        return rows

    missing = holdout - {r["speaker"] for r in rows.values()}

    if missing:

        raise ValueError(f"--holdout-speakers: 데이터에 없는 화자 {sorted(missing)}")

    for row in rows.values():

        if row["speaker"] in holdout:

            row["split"] = HOLDOUT

    return rows


def select(rows: dict, limits) -> dict:
    """화자·감정·분할마다 sentence_id가 작은 순서로 N개. holdout은 holdout=N이 없으면 전부."""

    if not limits:

        return rows

    picked, used = {}, Counter()

    for row in sorted(rows.values(), key=lambda r: (r["speaker"], r["emotion"], r["split"], r["sentence_id"])):

        key = (row["speaker"], row["emotion"], row["split"])

        if row["split"] == HOLDOUT and HOLDOUT not in limits:

            picked[row["utt_id"]] = row

        elif used[key] < limits.get(row["split"], 0):

            used[key] += 1

            picked[row["utt_id"]] = row

    return picked


def write_wavs(rows: dict, files, out_dir: Path):
    """2차: 고른 발화의 audio.bytes를 다시 인코딩하지 않고 그대로 저장. 행 번호(_index)로 찾고,
    같은 행의 speaker_id·emotion이 1차에서 읽은 값과 같은지 다시 확인한다."""

    for f in files:

        wanted = {r["_index"]: r for r in rows.values() if r["_file"] == f}

        if not wanted:

            continue

        i = 0

        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["audio", "speaker_id", "emotion"]):

            for item, spk, emo in zip(batch.column("audio").to_pylist(), batch.column("speaker_id").to_pylist(),
                                      batch.column("emotion").to_pylist()):

                row = wanted.get(i)

                i += 1

                if row is None:

                    continue

                if (spk, emo) != (row["speaker"], row["_emotion_raw"]):

                    raise RuntimeError(f"{f} {i - 1}번째 행: 음성의 화자·감정 {spk}/{emo} != "
                                       f"{row['utt_id']} (1차와 2차에서 읽은 행 순서가 다름)")

                rel = Path("wavs") / row["speaker"] / f"{row['utt_id']}.wav"

                (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)

                (out_dir / rel).write_bytes(item["bytes"])

                row["path"] = rel.as_posix()

                row["duration"] = f"{sf.info(str(out_dir / rel)).duration:.3f}"

    missing = [u for u, r in rows.items() if "path" not in r]

    if missing:

        raise RuntimeError(f"음성을 못 찾은 발화 {len(missing)}개, 예: {missing[:3]}")


def write_meta(out_dir: Path, revision: str, rows: dict, holdout: set):
    """--revision이 있으면 meta.json 저장. 없으면 경고만 (예전 meta.json은 이번 CSV와 안 맞으므로 지운다)."""

    meta_path = out_dir / "meta.json"

    if not revision:

        print("\n경고: --revision이 없어 데이터셋 버전을 기록하지 않음 (meta.json 없음)")

        if meta_path.exists():

            meta_path.unlink()

            print(f"  이전 실행의 {meta_path}는 이번 CSV와 맞지 않아 지움")

        return

    genders = {spk: g for spk, g, _ in speaker_table((r["speaker"], r["gender"]) for r in rows.values())}

    meta = {

        "dataset": DATASET, "revision": revision,

        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),

        "holdout_speakers": sorted(holdout), "speaker_genders": genders,
    }

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n저장: {meta_path} (revision {revision})")


def print_counts(rows: dict):

    counts = Counter((r["speaker"], r["emotion"], r["split"]) for r in rows.values())

    speakers = sorted({r["speaker"] for r in rows.values()})

    print(f"\n{'speaker':<8} {'emotion':<9} {'train':>6} {'eval':>5} {'test':>5} {'holdout':>8} {'total':>6}")

    for spk in speakers:

        for emo in config.LABELS:

            c = [counts[(spk, emo, s)] for s in ALL_SPLITS]

            print(f"{spk:<8} {emo:<9} {c[0]:>6} {c[1]:>5} {c[2]:>5} {c[3]:>8} {sum(c):>6}")

    tot = [sum(v for (_, _, s), v in counts.items() if s == split) for split in ALL_SPLITS]

    print(f"{'합계':<17}  {tot[0]:>6} {tot[1]:>5} {tot[2]:>5} {tot[3]:>8} {sum(tot):>6}")

    print(f"화자 {len(speakers)}명, 발화 {len(rows)}개")


def read_speaker_info(files, speakers) -> list:
    """--list-speakers용: speaker_id, gender, language 칸만 읽는다 (sentence_id·행 순서에 의존하지 않음).

    영어 발화마다 (speaker_id, gender)를 돌려준다. 칸 값이 예상과 다르면 에러.
    """

    pairs = []

    for f in files:

        for r in pq.read_table(f, columns=["speaker_id", "gender", "language"]).to_pylist():

            if r["language"] == "en" and speakers and r["speaker_id"] not in speakers:

                continue

            check_speaker_values(f, r)

            if r["language"] == "en":

                pairs.append((r["speaker_id"], r["gender"]))

    return pairs


def speaker_table(pairs) -> list:
    """(speaker_id, gender) 목록 → [(speaker_id, gender, 발화 수)] 화자 순서대로. 화자의 gender가 둘 이상이면 에러."""

    genders, counts = {}, Counter()

    for spk, gender in pairs:

        genders.setdefault(spk, set()).add(gender)

        counts[spk] += 1

    mixed = {spk: sorted(g) for spk, g in genders.items() if len(g) > 1}

    if mixed:

        raise ValueError(f"gender가 하나로 정해지지 않은 화자: {mixed}")

    return [(spk, genders[spk].pop(), counts[spk]) for spk in sorted(counts)]


def print_speakers(pairs):

    table = speaker_table(pairs)

    print(f"\n{'speaker':<8} {'gender':<8} {'발화 수':>7}")

    for spk, gender, n in table:

        print(f"{spk:<8} {gender:<8} {n:>8}")

    by_gender = Counter(g for _, g, _ in table)

    print(f"영어 화자 {len(table)}명 ({', '.join(f'{g} {n}명' for g, n in sorted(by_gender.items()))}), "
          f"발화 {sum(n for _, _, n in table)}개")


def print_holdout(rows: dict, holdout: set):
    """홀드아웃 화자와 gender 출력. 모두 같은 gender면 경고만 (막지는 않음)."""

    gender_of = {spk: g for spk, g, _ in speaker_table((r["speaker"], r["gender"]) for r in rows.values())}

    print("\n홀드아웃 화자 (split=holdout, 학습·검증·테스트에서 제외):")

    for spk in sorted(holdout):

        print(f"  {spk}  {gender_of.get(spk, '?')}")

    genders = {gender_of.get(spk) for spk in holdout}

    if len(holdout) >= 2 and len(genders) == 1:

        print(f"경고: 홀드아웃 화자의 gender가 모두 {genders.pop()} (남녀 1명씩 고르는 것을 권장)")


def check_full(rows: dict):
    """뽑기 전 영어 전체가 ESD 영어 전체(10명 × 5감정 × 350 = 17,500)인지 확인. 화자 수가 다르면 경고만.
    (화자·감정마다 350행인지는 check_blocks가 이미 에러로 막는다.)"""

    speakers = {r["speaker"] for r in rows.values()}

    expected = FULL_SPEAKERS * len(config.LABELS) * SENTENCES_PER_EMOTION

    if len(speakers) != FULL_SPEAKERS:

        print(f"\n경고: 영어 화자 {len(speakers)}명, 발화 {len(rows)}개 (ESD 영어 전체는 {FULL_SPEAKERS}명, {expected}개)")

    else:

        print(f"확인: 영어 화자 {FULL_SPEAKERS}명, 화자·감정마다 {SENTENCES_PER_EMOTION}개, 총 {expected}개")


def main(argv=None):

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("parquet", nargs="+", help="parquet 파일 (glob 가능, 따옴표로 감싸기). 항상 전부 넣는다")

    ap.add_argument("--out", type=Path, help="출력 폴더 (--list-speakers일 때만 생략 가능)")

    ap.add_argument("--speakers", default="", help="예: 0019,0020")

    ap.add_argument("--per-split", default="", help="예: train=10,eval=4,test=4 (holdout=N도 가능)")

    ap.add_argument("--holdout-speakers", default="", help="예: 0011,0016 (이 화자는 전부 split=holdout)")

    ap.add_argument("--revision", default="",
                    help=f"{DATASET} 데이터셋 버전(커밋 해시). 주면 OUT/meta.json에 기록, 없으면 경고만")

    ap.add_argument("--list-speakers", action="store_true",
                    help="메타데이터만 읽어 화자별 gender·발화 수를 출력하고 종료 (wav를 쓰지 않음)")

    args = ap.parse_args(argv)

    if args.out is None and not args.list_speakers:

        ap.error("--out이 필요함 (--list-speakers일 때만 생략 가능)")


    files = expand_inputs(args.parquet)

    speakers = parse_speakers(args.speakers)

    holdout = parse_speakers(args.holdout_speakers)

    limits = parse_per_split(args.per_split)

    if speakers and holdout - speakers:

        raise ValueError(f"--holdout-speakers {sorted(holdout - speakers)}가 --speakers에 없음")

    print(f"parquet {len(files)}개: {', '.join(Path(f).name for f in files)}")


    if args.list_speakers:

        print_speakers(read_speaker_info(files, speakers))

        return


    # 행 순서 검사는 --speakers/--per-split로 뽑기 전에 영어 화자 전체 기준으로
    blocks = read_blocks(files)

    check_blocks(blocks)

    all_rows = make_rows(blocks)

    speaker_table((r["speaker"], r["gender"]) for r in all_rows.values())  # 화자의 gender가 하나인지

    check_full(all_rows)


    rows = {u: r for u, r in all_rows.items() if not speakers or r["speaker"] in speakers}

    rows = select(assign_holdout(rows, holdout), limits)

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

    if holdout:

        print_holdout(rows, holdout)

    write_meta(args.out, args.revision, rows, holdout)

    print(f"\n저장: {csv_path}")


if __name__ == "__main__":

    main()
