import csv
import io
import json
import re
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from emotion_scorer import build_manifest
from emotion_scorer.build_manifest import (EMOTION_MAP, SENTENCES_PER_EMOTION, make_utt_id, normalize_text,
                                           parse_per_split, split_of)


@pytest.mark.parametrize("sentence_id, expected", [
    (1, "eval"), (20, "eval"), (21, "test"), (50, "test"), (51, "train"), (350, "train"),
])
def test_split_of(sentence_id, expected):
    assert split_of(sentence_id) == expected


@pytest.mark.parametrize("sentence_id", [0, 351])
def test_split_of_out_of_range(sentence_id):
    with pytest.raises(ValueError):
        split_of(sentence_id)


def test_utt_id_is_not_official_esd_format():
    utt_id = make_utt_id("0011", "anger", 1)
    assert utt_id == "0011_anger_001"
    assert not re.match(r"^\d{4}_\d{6}$", utt_id)  # 공식 ID(0011_000351)와 착각하지 않도록


def test_normalize_text():
    assert normalize_text("I'd shoot myself. ") == "i d shoot myself"
    assert normalize_text('I"ll be with you,Tom!') == "i ll be with you tom"


# ---- 실제 jspaulsen/esd와 같은 구조의 가짜 parquet ----
# audio.path는 전부 None, 중국어 행이 먼저, 영어는 화자 오름차순 → 감정 알파벳순으로 350행씩 이어진 블록,
# 블록 안 i번째 행은 모든 블록에서 같은 문장.

RAW_EMOTIONS = sorted(EMOTION_MAP)  # anger, happiness, neutral, sadness, surprise (실제 parquet 순서)
FAKE_SPEAKERS = {"0011": "male", "0012": "female", "0013": "male"}
N_FAKE = len(FAKE_SPEAKERS) * len(RAW_EMOTIONS) * SENTENCES_PER_EMOTION  # 5250
VARIANT = ("0012", "sadness", 5, "Sentense number 5 is here!")  # 표기만 다른 행 (실제 데이터의 "piglit"처럼)
WAV_SR = 1000  # 음성 길이 = sentence_id 샘플 → duration = sentence_id / 1000초 (음성과 행이 맞는지 확인용)


def sentence(sentence_id: int) -> str:
    return f"Sentence number {sentence_id} is here."


def wav_bytes(n_samples: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(n_samples, dtype=np.float32), WAV_SR, format="WAV")
    return buf.getvalue()


def fake_records() -> list[dict]:
    wavs = {sid: wav_bytes(sid) for sid in range(1, SENTENCES_PER_EMOTION + 1)}
    records = [{"audio": {"bytes": wavs[1], "path": None}, "transcript": "中文", "emotion": "neutral",
                "speaker_id": "0001", "gender": "female", "language": "zh"} for _ in range(3)]
    for spk, gender in FAKE_SPEAKERS.items():
        for emo in RAW_EMOTIONS:
            for sid in range(1, SENTENCES_PER_EMOTION + 1):
                text = VARIANT[3] if (spk, emo, sid) == VARIANT[:3] else sentence(sid)
                records.append({"audio": {"bytes": wavs[sid], "path": None}, "transcript": text, "emotion": emo,
                                "speaker_id": spk, "gender": gender, "language": "en"})
    return records


def index_of(records, spk, emo, sid) -> int:
    for i, r in enumerate(records):
        if (r["speaker_id"], r["emotion"], r["language"]) == (spk, emo, "en"):
            return i + sid - 1
    raise KeyError((spk, emo))


def write_parquet(path, records):
    schema = pa.schema([("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                        ("transcript", pa.string()), ("emotion", pa.string()), ("speaker_id", pa.string()),
                        ("gender", pa.string()), ("language", pa.string())])  # 실제 parquet과 같은 스키마
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path)
    return path


@pytest.fixture(scope="module")
def records():
    return fake_records()


@pytest.fixture(scope="module")
def fake_parquet(records, tmp_path_factory):
    return write_parquet(tmp_path_factory.mktemp("pq") / "fake.parquet", records)


@pytest.fixture
def broken(records, tmp_path):
    """records를 복사해 고친 뒤 parquet으로 쓰는 함수."""

    def make(edit):
        recs = [dict(r) for r in records]
        edit(recs)
        return write_parquet(tmp_path / "broken.parquet", recs)

    return make


def run_manifest(parquet, out, *extra) -> list[dict]:
    build_manifest.main([str(parquet), "--out", str(out), *extra])
    with open(out / "utterances.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---- 기본 manifest ----

def test_manifest_from_block_order(fake_parquet, tmp_path, capsys):
    rows = run_manifest(fake_parquet, tmp_path / "out")
    assert len(rows) == N_FAKE
    assert list(rows[0].keys()) == build_manifest.CSV_COLUMNS
    assert [r["utt_id"] for r in rows] == sorted(r["utt_id"] for r in rows)
    out = capsys.readouterr().out
    assert "행 순서 검사 통과: 블록 15개 × 350행, 위치별 문장이 표기만 다른 행 1개" in out
    for r in rows:
        sid = int(r["sentence_id"])
        assert r["utt_id"] == f"{r['speaker']}_{[k for k, v in EMOTION_MAP.items() if v == r['emotion']][0]}_{sid:03d}"
        assert r["split"] == split_of(sid)
        assert r["gender"] == FAKE_SPEAKERS[r["speaker"]]
        # 음성이 같은 행에서 왔는지: 가짜 음성 길이 = sentence_id 샘플
        assert r["duration"] == f"{sid / WAV_SR:.3f}"
        assert (tmp_path / "out" / r["path"]).is_file()
    assert Counter(r["split"] for r in rows) == {"eval": 20 * 15, "test": 30 * 15, "train": 300 * 15}
    assert "0001" not in {r["speaker"] for r in rows}  # 중국어 화자 제외


def test_text_is_each_rows_own_transcript(fake_parquet, tmp_path):
    """다수결 문장으로 바꾸지 않고, 음성과 실제로 짝인 그 행의 transcript를 그대로 쓴다."""
    rows = {r["utt_id"]: r for r in run_manifest(fake_parquet, tmp_path / "out", "--speakers", "0012",
                                                 "--per-split", "train=1,eval=5,test=1")}
    assert rows["0012_sadness_005"]["text"] == VARIANT[3]
    assert rows["0012_happiness_005"]["text"] == sentence(5)


def test_no_holdout_option_is_byte_identical(fake_parquet, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    run_manifest(fake_parquet, a, "--per-split", "train=2,eval=2,test=2")
    run_manifest(fake_parquet, b, "--per-split", "train=2,eval=2,test=2", "--holdout-speakers", "")
    assert (a / "utterances.csv").read_bytes() == (b / "utterances.csv").read_bytes()


# ---- 행 순서 검사 (하나라도 깨지면 에러) ----

def test_block_not_350_rows(broken, tmp_path):
    path = broken(lambda recs: recs.pop(index_of(recs, "0012", "anger", 100)))
    with pytest.raises(ValueError, match="350행이 아닌 블록: 0012/anger=349"):
        run_manifest(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_block_interrupted_by_other_row(broken, tmp_path):
    zh = {"audio": {"bytes": b"", "path": None}, "transcript": "中文", "emotion": "neutral",
          "speaker_id": "0001", "gender": "female", "language": "zh"}
    path = broken(lambda recs: recs.insert(index_of(recs, "0012", "anger", 100), zh))
    with pytest.raises(ValueError, match="연속되지 않음.*0012/anger 2조각"):
        run_manifest(path, tmp_path / "out")


def test_block_mixed_emotion(broken, tmp_path):
    def edit(recs):
        i = index_of(recs, "0011", "anger", 10)
        recs[i] = {**recs[i], "emotion": "happiness"}
    with pytest.raises(ValueError, match="연속되지 않음.*0011/anger 2조각"):
        run_manifest(broken(edit), tmp_path / "out")


def test_position_sentence_mismatch(broken, tmp_path):
    def edit(recs):
        i = index_of(recs, "0013", "surprise", 100)
        recs[i] = {**recs[i], "transcript": "Completely unrelated words."}
    with pytest.raises(ValueError, match=r"(?s)위치별 문장이 다른 블록과 다름 1개.*0013/surprise 100번째"):
        run_manifest(broken(edit), tmp_path / "out")


def test_check_runs_on_all_speakers_before_selection(broken, tmp_path):
    """--speakers/--per-split로 0011만 뽑아도 0013의 블록이 깨져 있으면 에러."""
    path = broken(lambda recs: recs.pop(index_of(recs, "0013", "neutral", 1)))
    with pytest.raises(ValueError, match="0013/neutral=349"):
        run_manifest(path, tmp_path / "out", "--speakers", "0011", "--per-split", "train=1,eval=1,test=1")


def test_unknown_emotion(broken, tmp_path):
    def edit(recs):
        i = index_of(recs, "0011", "anger", 1)
        recs[i] = {**recs[i], "emotion": "fear"}
    with pytest.raises(ValueError, match="모르는 감정 'fear'"):
        run_manifest(broken(edit), tmp_path / "out")


def test_partial_parquet_files_fail(records, tmp_path):
    """블록이 파일 경계에 걸쳐 있을 때 파일 하나만 넣으면 에러 (실제 데이터의 0018/anger처럼)."""
    cut = index_of(records, "0012", "neutral", 200)
    write_parquet(tmp_path / "train-00000.parquet", records[:cut])
    second = write_parquet(tmp_path / "train-00001.parquet", records[cut:])
    rows = run_manifest(str(tmp_path / "train-*.parquet"), tmp_path / "all",
                        "--per-split", "train=1,eval=1,test=1")  # 전부 넣으면 파일을 넘어 이어짐
    assert len(rows) == 15 * 3
    with pytest.raises(ValueError, match="350행이 아닌 블록"):
        run_manifest(second, tmp_path / "part")


# ---- 화자 홀드아웃 ----

def test_holdout_speaker_all_utterances(fake_parquet, tmp_path, capsys):
    rows = run_manifest(fake_parquet, tmp_path / "out", "--holdout-speakers", "0012")
    held = [r for r in rows if r["speaker"] == "0012"]
    # sentence_id 1~350 (eval·test·train 구간 모두) 전부 holdout
    assert len(held) == 5 * SENTENCES_PER_EMOTION
    assert {int(r["sentence_id"]) for r in held} == set(range(1, SENTENCES_PER_EMOTION + 1))
    assert all(r["split"] == "holdout" for r in held)
    # 나머지 화자는 sentence_id 분할 그대로
    assert all(r["split"] == split_of(int(r["sentence_id"])) for r in rows if r["speaker"] != "0012")
    out = capsys.readouterr().out
    assert any("holdout" in line and line.startswith("speaker") for line in out.splitlines())  # print_counts 머리줄
    assert "0012  female" in out
    assert "경고: 홀드아웃 화자의 gender" not in out


def test_holdout_same_gender_warns(fake_parquet, tmp_path, capsys):
    run_manifest(fake_parquet, tmp_path / "out", "--holdout-speakers", "0011,0013",
                 "--per-split", "train=1,eval=1,test=1,holdout=1")
    assert "경고: 홀드아웃 화자의 gender가 모두 male" in capsys.readouterr().out


def test_holdout_missing_speaker_raises(fake_parquet, tmp_path):
    with pytest.raises(ValueError, match="데이터에 없는 화자"):
        run_manifest(fake_parquet, tmp_path / "out", "--holdout-speakers", "0012,0099")
    assert not (tmp_path / "out" / "utterances.csv").exists()


def test_holdout_not_in_speakers_raises(fake_parquet, tmp_path):
    with pytest.raises(ValueError, match="--speakers에 없음"):
        run_manifest(fake_parquet, tmp_path / "out", "--speakers", "0011,0012", "--holdout-speakers", "0013")


def test_per_split_holdout_limit(fake_parquet, tmp_path):
    rows = run_manifest(fake_parquet, tmp_path / "out", "--holdout-speakers", "0012",
                        "--per-split", "train=1,eval=1,test=1,holdout=2")
    counts = Counter((r["speaker"], r["emotion"], r["split"]) for r in rows)
    for emo in EMOTION_MAP.values():
        assert counts[("0012", emo, "holdout")] == 2
        for spk in ("0011", "0013"):
            assert [counts[(spk, emo, s)] for s in ("train", "eval", "test")] == [1, 1, 1]
    # holdout도 sentence_id가 작은 순서로 고른다
    assert {r["sentence_id"] for r in rows if r["split"] == "holdout"} == {"1", "2"}
    assert {r["sentence_id"] for r in rows if r["split"] != "holdout"} == {"1", "21", "51"}


def test_per_split_without_holdout_key_uses_all_holdout(fake_parquet, tmp_path):
    rows = run_manifest(fake_parquet, tmp_path / "out", "--holdout-speakers", "0012",
                        "--per-split", "train=1,eval=1,test=1")
    assert sum(r["split"] == "holdout" for r in rows) == 5 * SENTENCES_PER_EMOTION


def test_parse_per_split_holdout():
    assert parse_per_split("train=10,holdout=3") == {"train": 10, "holdout": 3}
    with pytest.raises(ValueError, match="모르는 분할"):
        parse_per_split("dev=3")


# ---- 데이터셋 버전 (meta.json) ----

def test_revision_writes_meta(fake_parquet, tmp_path):
    out = tmp_path / "out"
    run_manifest(fake_parquet, out, "--revision", "abc123", "--holdout-speakers", "0012",
                 "--per-split", "train=1,eval=1,test=1,holdout=1")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert set(meta) == {"dataset", "revision", "created_at", "holdout_speakers", "speaker_genders"}
    assert meta["dataset"] == "jspaulsen/esd"
    assert meta["revision"] == "abc123"
    assert meta["holdout_speakers"] == ["0012"]
    assert meta["speaker_genders"] == FAKE_SPEAKERS
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", meta["created_at"])


def test_no_revision_warns_and_removes_stale_meta(fake_parquet, tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "meta.json").write_text('{"revision": "old"}', encoding="utf-8")  # 이전 실행이 남긴 것
    run_manifest(fake_parquet, out, "--per-split", "train=1,eval=1,test=1")
    assert "경고: --revision이 없어 데이터셋 버전을 기록하지 않음" in capsys.readouterr().out
    assert not (out / "meta.json").exists()


# ---- --list-speakers ----

def test_list_speakers_without_out(fake_parquet, tmp_path, monkeypatch, capsys):
    """sentence_id·행 순서와 상관없이 speaker_id·gender만으로 화자 표가 나와야 한다."""
    monkeypatch.chdir(tmp_path)
    build_manifest.main([str(fake_parquet), "--list-speakers"])
    out = capsys.readouterr().out
    assert "0011     male" in out and "0012     female" in out
    assert "0001" not in out  # 영어가 아닌 화자는 제외
    assert f"영어 화자 3명 (female 1명, male 2명), 발화 {N_FAKE}개" in out
    assert not list(tmp_path.rglob("*.wav")) and not list(tmp_path.rglob("*.csv"))


def test_list_speakers_ignores_block_order(broken, capsys):
    path = broken(lambda recs: recs.pop(index_of(recs, "0012", "anger", 100)))
    build_manifest.main([str(path), "--list-speakers"])
    assert f"발화 {N_FAKE - 1}개" in capsys.readouterr().out


def test_out_required_without_list_speakers(fake_parquet):
    with pytest.raises(SystemExit):
        build_manifest.main([str(fake_parquet)])


@pytest.mark.parametrize("field, value, match", [
    ("gender", "unknown", "모르는 gender"),
    ("gender", None, "모르는 gender"),
    ("language", "fr", "모르는 language"),
    ("speaker_id", "11", "speaker_id 형식"),
])
@pytest.mark.parametrize("mode", ["list", "manifest"])
def test_rejects_unexpected_values(broken, tmp_path, field, value, match, mode):
    def edit(recs):
        i = index_of(recs, "0011", "anger", 1)
        recs[i] = {**recs[i], field: value}
    path = broken(edit)
    args = [str(path), "--list-speakers"] if mode == "list" else [str(path), "--out", str(tmp_path / "out")]
    with pytest.raises(ValueError, match=match):
        build_manifest.main(args)


def test_list_speakers_rejects_mixed_gender(broken):
    def edit(recs):
        i = index_of(recs, "0011", "anger", 1)
        recs[i] = {**recs[i], "gender": "female"}  # 0011은 male인데 한 줄만 female
    with pytest.raises(ValueError, match="gender가 하나로"):
        build_manifest.main([str(broken(edit)), "--list-speakers"])
