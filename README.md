# emotion_scorer

ESD 감정 데이터로 학습한, TTS 음성 감정 채점용 분류기.

- 바닥 모델 `microsoft/wavlm-base-plus` (고정, 학습 안 함)
- 13개 hidden state를 시간 축 평균 → 발화당 `[13, 768]`
- 층 가중합(softmax) → `Linear(768 → 5)`
- 라벨 순서: `neutral, happy, angry, sad, surprise`

## 파일

| 파일 | 하는 일 |
|---|---|
| `build_manifest.py` | ESD parquet → wav + 발화 CSV (`utterances.csv`) + `meta.json`. 화자 목록 확인, 화자 홀드아웃 분할 |
| `config.py` | 경로, 모델 이름, 라벨 순서, 학습 설정, 디바이스 선택(cuda → mps → cpu) |
| `audio.py` | `load_audio`: 읽기 → 모노 → 16kHz. 모든 전처리는 이 함수만 사용 |
| `extract_features.py` | 발화 CSV → `features.npy` `[N,13,768]` float16 + `features_index.txt` |
| `train_classifier.py` | 특징 + CSV → 분류기 체크포인트(정규화 값 포함) + 리포트 |
| `score.py` | 음성 파일/폴더 → 확률 CSV |
| `baselines/summarize.py` | 채점 CSV(실제 녹음) → 화자 × 감정별 정답률·평균 확률 표 |

발화 CSV 칸: `utt_id, speaker, emotion, sentence_id, text, split, path, duration`
(`build_manifest`는 `gender`도 함께 기록. `split`은 `train`/`eval`/`test`/`holdout`,
`path`가 상대 경로면 CSV 파일이 있는 폴더 기준)

`sentence_id`와 `utt_id`:

- 사용하는 Hugging Face `jspaulsen/esd` parquet에는 원본 파일 이름이 없다(`audio.path`가 전부 비어 있음).
  그래서 **sentence_id는 parquet 블록 안 순서이며 ESD 공식 ID와 일치한다는 보장은 없다.**
  parquet은 (화자, 감정)마다 350행씩 이어진 블록으로 되어 있고, 블록 안 i번째 행은 모든 블록에서
  같은 문장이다. 그 i를 sentence_id(1–350)로 쓴다.
- 공식 ID와 착각하지 않도록 `utt_id`는 `{speaker}_{parquet 감정 이름}_{sentence_id:03d}` 형식이다
  (예: `0011_anger_001`). ESD 원본의 `0011_000351` 형식과 대조하면 안 된다.
- 공식 분할과 정확히 같지는 않을 수 있지만, 같은 위치는 모든 화자·감정에서 같은 문장이므로
  **같은 문장이 train과 eval/test에 동시에 들어가지는 않는다.**
- 이 방식은 행 순서에 기대므로 `build_manifest`는 매번 영어 화자 전체를 기준으로 다음을 검사하고,
  하나라도 깨지면 에러를 낸다. `--speakers`나 `--per-split`로 일부만 뽑을 때도 뽑기 전에 전체를 검사한다.
  1. 같은 (화자, 감정)이 끊기지 않고 한 블록으로 이어지는지 (다른 행이 섞이지 않았는지)
  2. 블록마다 350행인지, 화자마다 5감정이 다 있는지
  3. 블록 안 i번째 문장이 모든 블록의 i번째 다수결 문장과 같은지. 소문자로 바꾸고 문장부호를 지운 뒤
     비교하고, 표기만 다른 경우(유사도 0.6 이상, 실제 데이터는 최소 0.75)는 통과시킨다.
- 블록이 parquet 파일 경계에 걸쳐 있으므로 **parquet 파일은 항상 전부 넣는다** (일부만 넣으면 2번 검사에서 에러).
- `text` 칸은 각 행의 실제 transcript다 (앞뒤 공백만 제거). 표기가 다른 행도 다수결 문장으로 바꾸지 않는다.
  음성과 실제로 짝인 대사여야 TTS 학습에 그대로 쓸 수 있기 때문이다.

분할 규칙:

- 기본은 sentence_id 분할(1–20 eval, 21–50 test, 51–350 train)이라 영어 화자 10명이
  모두 train과 test에 들어간다. 이때 test 정확도는 "이미 들어본 화자의 새 문장"에 대한 성능이다.
- `--holdout-speakers`로 지정한 화자는 sentence_id와 상관없이 전부 `holdout`이 된다.
  holdout은 학습, early stopping, 최고 모델 선택, 정규화 값 계산에 쓰지 않고 마지막에 한 번만 평가한다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

CUDA PC에서는 먼저 맞는 CUDA 버전의 torch를 설치한 뒤 위 명령을 실행.

## 실행 순서

1. 화자·성별 확인 (메타데이터만 읽고 wav는 쓰지 않음, `--out` 불필요)

   ```bash
   python -m emotion_scorer.build_manifest "data/train-*.parquet" --list-speakers
   ```

2. 남녀 1명씩 골라 홀드아웃으로 manifest 생성 (나머지 8명은 공식 분할로 train/eval/test)

   ```bash
   python -m emotion_scorer.build_manifest "data/train-*.parquet" --out data \
       --holdout-speakers 0011,0016 --revision <jspaulsen/esd 커밋 해시>
   ```

   예시의 `0011,0016`은 자리표시자다. 1단계 표에서 확인한 화자 ID로 바꾼다.
   두 화자의 gender가 같으면 경고만 출력한다.
   작은 샘플은 `--per-split train=10,eval=4,test=4,holdout=4`처럼 holdout 개수도 정할 수 있다
   (`holdout=N`이 없으면 홀드아웃 화자의 발화를 전부 사용).

   `--revision`을 주면 `data/meta.json`에 데이터셋 버전을 기록한다
   (`dataset`, `revision`, 만든 시각 `created_at`, `holdout_speakers`, 화자별 gender `speaker_genders`).
   커밋 해시는 Hugging Face 데이터셋 페이지의 "Files and versions"에서 확인한다.
   없으면 경고만 출력하고, 같은 폴더에 이전 실행의 `meta.json`이 남아 있으면 이번 CSV와 맞지 않으므로 지운다.

3. 특징 추출 → 학습 → 채점

   ```bash
   python -m emotion_scorer.extract_features --csv data/utterances.csv
   python -m emotion_scorer.train_classifier --csv data/utterances.csv   # 기존 체크포인트가 있으면 멈춤
   python -m emotion_scorer.score path/to/tts_outputs/ --out scores.csv
   ```

기본 경로는 `config.py` 참고 (`features/`, `checkpoints/classifier.pt`).
`train_classifier`는 `--out`에 체크포인트가 이미 있으면 경고하고 학습 전에 멈춘다.
덮어쓰려면 `--overwrite`를 준다 (채점 기준을 실수로 바꾸지 않기 위함).
체크포인트에는 `train_speakers`와 `holdout_speakers`가 저장되고, CSV 옆에 `meta.json`이 있으면
그 내용이 `dataset_meta`로 함께 저장된다 (없으면 `None`, 학습 시작 때 경고). `score`는 시작할 때
"이 분류기가 학습에서 보지 않은 화자"를 한 줄 출력한다 (화자 정보가 없는 예전 체크포인트도 읽힘).

**홀드아웃 화자는 이후 TTS 평가에서 화자 참조로 쓸 화자이므로, TTS 학습 데이터에도 넣지 않는다.**

## 리포트 읽는 법

`checkpoints/classifier.report.txt`에는 기존 test 리포트(정확도, macro F1, 감정별 정답률, 혼동 행렬,
층 가중치) 뒤에 holdout 리포트(같은 형식), `[요약]` 한 줄, 화자별 정확도가 붙는다.

- test 정확도: 학습에서 들어본 화자의 새 문장에 대한 성능
- holdout 정확도: 처음 보는 화자에 대한 성능. TTS 합성음 채점에 더 가까운 값
- 차이(test − holdout): 클수록 분류기가 감정 자체보다 학습 화자의 고유한 목소리 단서에 기대고 있다는 뜻이다.
  화자별 정확도에서 특정 홀드아웃 화자만 낮다면 그 화자의 특성(성별, 발성 습관)도 함께 확인한다.

## 학습 결과

채점 기준 체크포인트: **`checkpoints/v1/classifier.pt`** (리포트 `checkpoints/v1/classifier.report.txt`).
TTS 채점에는 항상 이 파일을 쓴다 (`score --checkpoint checkpoints/v1/classifier.pt`).

- 데이터: `jspaulsen/esd` revision `dc57273e5872666a110e68dd9e0eee11d5efad60`
- 홀드아웃 화자: 0013 (male), 0017 (female) / 학습 화자 8명 (0011, 0012, 0014, 0015, 0016, 0018, 0019, 0020)

| | 정확도 |
|---|---|
| test (들어본 화자의 새 문장) | 0.9150 |
| holdout (처음 보는 화자) | 0.7549 |
| 차이 (test − holdout) | 0.16 |

화자별 holdout 정확도: 0013 0.652, 0017 0.858

holdout 감정별 정답률:

| neutral | happy | angry | sad | surprise |
|---|---|---|---|---|
| 0.82 | 0.34 | 0.92 | 0.73 | 0.96 |

층 가중치는 3~6층에 집중되어 있다 (0.13, 0.16, 0.14, 0.12. 나머지 층은 0.03~0.08).

## 한계와 사용 규칙

- 처음 보는 화자에서는 흥분도(들뜸/차분함)는 잘 구분하지만, 같은 흥분도 안의 감정
  (happy ↔ surprise ↔ angry, neutral ↔ sad)은 헷갈린다. holdout 혼동 행렬에서 happy는 surprise(334개)와
  angry(98개)로, neutral과 sad는 서로(124개, 184개) 가장 많이 틀린다.
- happy는 처음 보는 화자에서 약하다 (0.34). 틀린 음성을 직접 들어 보니 사람이 들어도 애매한 경우가 많아,
  분류기보다 연기 자체의 한계로 판단했다.
- 그래서 TTS 평가에서는 다음 규칙을 따른다.
  - **같은 화자·같은 문장의 상대 비교를 기본으로 한다** (예: 감정 조건을 바꿨을 때 확률이 어떻게 움직이는지).
    서로 다른 화자나 문장 사이의 절대 점수 비교는 하지 않는다.
  - **happy는 `p_happy` 하나가 아니라 확률 분포 전체의 변화로 해석한다**
    (예: neutral 대비 `p_surprise`, `p_angry` 쪽으로 함께 이동했는지).

## 실제 녹음 기준선

`baselines/holdout_real.csv`는 홀드아웃 화자(0013, 0017) **실제 녹음** 3,500개를 `checkpoints/v1/classifier.pt`로
채점한 결과다. 같은 화자로 합성한 TTS 점수의 **상한선 비교용**이다 (실제 녹음보다 높게 나오기는 어렵다).

화자 × 감정별 정답률과 평균 확률 표는 `baselines/holdout_real_summary.txt`에 있다. 다시 만들려면:

```bash
python baselines/summarize.py baselines/holdout_real.csv --out baselines/holdout_real_summary.txt
```

정답 감정은 파일 이름(`{speaker}_{감정}_{sentence_id:03d}`)에서 읽는다. 전체 정답률 0.7549와
화자별·감정별 값은 `checkpoints/v1/classifier.report.txt`의 holdout 값과 같다.

## 검증 (데이터 없이)

```bash
pytest -s tests/test_pipeline.py
pytest tests/test_build_manifest.py   # 작은 가짜 parquet으로 홀드아웃 분할 검증
pytest tests/test_summarize.py        # 기준선 요약 표
```
