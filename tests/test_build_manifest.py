import pytest

from emotion_scorer.build_manifest import id_info, make_row


@pytest.mark.parametrize("n, expected", [
    (1, ("neutral", 1, "eval")),
    (21, ("neutral", 21, "test")),
    (51, ("neutral", 51, "train")),
    (351, ("angry", 1, "eval")),
    (601, ("angry", 251, "train")),
    (1750, ("surprise", 350, "train")),
])
def test_id_info(n, expected):
    assert id_info(n) == expected


@pytest.mark.parametrize("n", [0, 1751])
def test_id_info_out_of_range(n):
    with pytest.raises(ValueError):
        id_info(n)


def test_make_row_ok():
    row = make_row("0018_000601.wav", "0018", "anger", "  Hello there.  ", "female")
    assert row == {"utt_id": "0018_000601", "speaker": "0018", "emotion": "angry", "sentence_id": 251,
                   "text": "Hello there.", "split": "train", "gender": "female"}


def test_make_row_emotion_block_mismatch():
    # 601번은 angry 블록인데 emotion 칸이 happiness
    with pytest.raises(ValueError, match="감정 블록"):
        make_row("0018_000601.wav", "0018", "happiness", "x", "female")


def test_make_row_speaker_mismatch():
    with pytest.raises(ValueError, match="화자"):
        make_row("0018_000601.wav", "0019", "anger", "x", "female")
