# emotion_scorer

ESD 감정 데이터로 학습한, TTS 음성 감정 채점용 분류기.

- 바닥 모델 `microsoft/wavlm-base-plus` (고정, 학습 안 함)
- 13개 hidden state를 시간 축 평균 → 발화당 `[13, 768]`
- 층 가중합(softmax) → `Linear(768 → 5)`
- 라벨 순서: `neutral, happy, angry, sad, surprise`

## 파일

| 파일 | 하는 일 |
|---|---|
| `config.py` | 경로, 모델 이름, 라벨 순서, 학습 설정, 디바이스 선택(cuda → mps → cpu) |
| `audio.py` | `load_audio`: 읽기 → 모노 → 16kHz. 모든 전처리는 이 함수만 사용 |
| `extract_features.py` | 발화 CSV → `features.npy` `[N,13,768]` float16 + `features_index.txt` |
| `train_classifier.py` | 특징 + CSV → 분류기 체크포인트(정규화 값 포함) + 리포트 |
| `score.py` | 음성 파일/폴더 → 확률 CSV |

발화 CSV 칸: `utt_id, speaker, emotion, sentence_id, text, split, path, duration`
(`split`은 `train`/`eval`/`test`, `path`가 상대 경로면 CSV 파일이 있는 폴더 기준)

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

CUDA PC에서는 먼저 맞는 CUDA 버전의 torch를 설치한 뒤 위 명령을 실행.

## 실행 순서

```bash
python -m emotion_scorer.extract_features --csv data/utterances.csv
python -m emotion_scorer.train_classifier --csv data/utterances.csv
python -m emotion_scorer.score path/to/tts_outputs/ --out scores.csv
```

기본 경로는 `config.py` 참고 (`features/`, `checkpoints/classifier.pt`).

## 검증 (데이터 없이)

```bash
pytest -s tests/test_pipeline.py
```
