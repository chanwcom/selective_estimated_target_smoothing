# 전통 LS 스윕 — 인하대 (`inha_4090_00`) 작업 지시

2026-09-28 개정. PAS 실험(`PAS_SPEC.md`, 같은 디렉터리)의 **비교 기준**이 되는
전통 LS 를 맡아 주십시오.

## 이 개정에서 바뀐 것

| 항목 | 이전 | 지금 |
|---|---|---|
| 배치 | 언급 없음 | **동적 배칭 `--max_batch_audio_len 1600000`** |
| learning rate | 1e-4 | **5e-5** |
| eval 배치 | 언급 없음 | **훈련과 따로 4** |
| GPU 상한 | 2장 | **3장** |
| RNN-T 인코더 인자 | 불명 | **`--model_name`** |
| 실행 예 | 없음 | 아래 §실행 예 |

---

## 맡을 것

**교과서 uniform label smoothing**, PAS 와 **동일한 조건·동일한 격자**로.

```
9 alpha x 2 손실(CTC, RNN-T) x 5 seed = 90 런
baseline                     x 5 seed = 10 런
```

2단계를 권합니다 — **seed 0 으로 9점을 먼저** 돌려 곡선을 보고, 최적 부근
2~3점만 5시드로 채우면 40런 안쪽입니다.

---

## 중요 — 전통 LS 는 **진짜** 전통 LS 입니다

PAS 와 헷갈리지 마십시오. PAS 는 blank 를 빼고 정렬 평면에서 하지만,
**여기서 맡으시는 것은 그런 변형이 아닙니다.**

```
y_ls = (1 - alpha) * y + alpha / C        C = 전체 클래스 수 (=32)
```

- **클래스 축**에서 한다 (`--smoothing_space class`)
- **blank 를 빼지 않는다.** 전 클래스에 균등
- 활성 집합 개념이 없다 — 모든 클래스가 `alpha/C` 를 받는다

기존 저장소에서 이것은 `--alpha_mode floored_active_support --beta 1.0
--smoothing_space class` 로 나옵니다. `CHECKPOINTS.md` 에 *"FAS + beta=1.0 이
textbook LS 와 bit-exact"* 로 기록돼 있습니다.

**`pas_h` / `pas_u` 를 쓰지 마십시오.** 그건 5090·4090 이 맡습니다.

---

## 훈련 설정 — PAS 와 동일하게

| 항목 | 값 |
|---|---|
| 모델 | `facebook/wav2vec2-large-lv60` (SSL 전용) |
| 파인튜닝 셋 | LibriSpeech train-clean-100 (100h) |
| 총 스텝 | 15,000 |
| 스케줄 | WSD warmup 1,000 / stable 11,000 / decay 3,000 |
| optimizer | AdamW, lr **5e-5** (CTC · RNN-T 동일) — 아래 참조 |
| alpha 격자 | **0.00125 0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32** |
| beta | **1.0** (전통 LS 를 만드는 값) |
| smoothing_space | **class** |

**`wav2vec2-large-960h-lv60-self` 를 쓰면 안 됩니다** — 이미 960h 로 ASR
파인튜닝된 모델이라 100h 연구에 쓸 수 없습니다. `large-lv60` 은
`Wav2Vec2ForPreTraining`, SSL 만 된 것입니다. RNN-T 인코더도 large 로
통일합니다 — 인자는 **`--model_name`** 입니다 (`wav2vec2_rnnt.py:201`, 이번에
추가된 것이라 `git pull` 이 필요합니다. 이전에는 `wav2vec2-base` 가
하드코딩돼 있어서 바꿀 수가 없었습니다).

⚠ **`--finetune_profile` 만 주면 스케줄이 1000/10000/4000 이 됩니다.**
프로파일 기본값이 그렇습니다. 사양의 1000/11000/3000 을 쓰려면 **네 값을
명시로 덮어쓰셔야** 합니다. 명시 인자가 프로파일을 이기고, 세 값이
`max_steps` 와 안 맞으면 assert 로 죽으므로 조용히 틀릴 일은 없습니다.

기존 AWS 레시피(1000/10000/4000)와 달라지는 것은 맞습니다. 어차피 그쪽
AWS 는 `wav2vec2-base` 라 large 기반 이번 실험과 같은 표에 못 놓습니다.
LS · PAS · baseline 이 **자기들끼리** 같은 조건이면 됩니다.

### learning rate — **5e-5. 1e-4 가 아니다**

배치가 기존 대비 1/4 이 되었으므로 그에 맞춰 낮춘다.

| | 기존 AWS (base) | 이번 PAS (large) |
|---|---|---|
| `max_batch_audio_len` | 6,400,000 | **1,600,000** |
| 실제 배치 B (중앙값) | **27** 발화 | **6.7** 발화 |
| `grad_accum` | 2 | 2 |
| 유효 배치 | ~54 발화 | **~13 발화** |
| learning rate | 1e-4 | **5e-5** |

RNN-T 는 `max_dynamic_batch_size` 가 `None` 이라 개수 상한이 걸리지 않는다.
오디오 예산만 binding 이므로 예산 1/4 이 배치 1/4 로 그대로 간다.

**왜 정확히 절반인가**: AdamW 계열의 sqrt 스케일링(`lr ∝ √batch`)으로
`1/√4 = 1/2`. 그리고 5e-5 는 wav2vec2-large 파인튜닝의 표준 범위라, 배치
문제를 빼고 보더라도 large 에 1e-4 를 쓰는 것보다 안전한 쪽이다.

⚠ **15,000 스텝이 보는 데이터도 1/4 이 된다** — 기존 약 28 에폭에서
**약 7 에폭**으로 줄어든다. 기울기 누적을 8 로 올리면 메모리 증가 없이
기존 유효 배치와 데이터량을 그대로 복원할 수 있지만 스텝당 4배가 걸려
런당 24~33 시간이 된다. 런이 100개 가까우므로 택하지 않았다.

**그래서 baseline 을 가장 먼저 돌린다.** baseline WER 이 100h large 의
통상 범위에서 크게 벗어나면 7 에폭이 모자란다는 신호이고, 그때는 스윕
전체를 시작하기 전에 이 결정을 다시 본다.

### 배치 — **반드시 동적 배칭. 값은 실측 고정입니다**

```
--dynamic_batching --max_batch_audio_len 1600000 --max_sample_audio_len 480000
CTC   추가:  --per_device_eval_batch_size 4
RNN-T 추가:  --eval_batch_size 4
```

**CTC · RNN-T 가 같은 값을 씁니다. 바꾸지 마십시오.**

`--dynamic_batching` 을 켜지 않으면 `--gpu_profile` 의
`per_device_train_batch_size=24` 가 들어갑니다. 그 값은 **base 인코더 기준**
이라 large 에서 OOM 합니다 — 4090 에서 실제로 났습니다. 기존 100h 실험도
전부 동적 배칭이었습니다.

**실측** (5090 에 `--gpu_memory_fraction 0.46` = 14.4 GiB 상한을 걸어 24GB
카드 조건을 재현한 값. 실물 4090 과 0.1 GiB 이내로 일치 확인):

| | 피크 |
|---|---|
| RNN-T @ 1,600,000 (eval 포함) | **13.68 alloc / 14.42 GiB reserved — 통과** |
| RNN-T @ 2,400,000 | 21.5 GiB |
| CTC @ 1,600,000 | 여유 |
| 고정배치 24 (동적배칭 꺼짐) | **20.2 GiB 에서 OOM** |

**카드당 동시 실행: 24GB 4090 은 1개.** (5090 은 2개)
**GPU 는 최대 3장** = 동시 3런으로 부탁드립니다.

**eval 배치를 따로 낮추는 이유**: eval 이 메모리 피크입니다. RNN-T 는 eval
DataLoader 가 `batch_size=32` 로 하드코딩돼 있어 large 에서 한 번에 4.07 GiB
를 요구했고, 훈련이 13.9 GiB 로 잘 돌던 런을 eval 에서 죽였습니다.
`--eval_batch_size` 인자를 새로 냈습니다 (기본 32 로 기존 레시피 불변).
eval 배치 크기는 WER 에 영향이 없습니다.

---

## 경로 — 인하대는 NAS 를 못 쓰므로 로컬에 저장

```
체크포인트  /mnt/data/home/chanwcom/models/ls_{ctc,rnnt}_libri100hr_alpha_{A}_s{SEED}
baseline    /mnt/data/home/chanwcom/models/baseline_{ctc,rnnt}_libri100hr_s{SEED}
로그        /mnt/data/home/chanwcom/logs/<체크포인트 디렉터리명>.log
결과        /mnt/data/home/chanwcom/results/RESULT_LS.md
```

`logs/` 와 `results/` 는 없으면 만들어 주십시오.

- `{A}` 는 소수점을 `p` 로: `0.00125 -> 0p00125`, `0.32 -> 0p32`
- **`/tmp` 금지.** 처음부터 영구 저장소에 쓸 것
- `run_args.json` 은 스크립트가 자동으로 남깁니다
- 이름 형식은 한 가지만


### 로그 머리에 **론치 파라미터 전체**를 기록한다

**모든 머신 공통 의무다.** 나중에 어떤 런이 무슨 설정으로 돌았는지를
로그 하나만 열어서 알 수 있어야 한다.

`run_args.json` 만으로는 부족하다. 그건 학습 스크립트가 **파싱한 것**만
담으며, 어느 git 리비전이었는지, 어느 카드였는지, `--dynamic_batching` 을
실제로 넘겼는지 기본값이었는지를 보여주지 못한다.

훈련 시작 전에 로그 맨 앞에 다음을 남긴다.

| 항목 | 예 |
|---|---|
| 런 이름 | `pas_h_rnnt_libri100hr_alpha_0p01_s3` |
| method / loss | `PAS-H / rnnt` |
| alpha / seed | `0.01 / 3` |
| 파인튜닝 셋 | `LibriSpeech train-clean-100 (100h)` |
| 시작 시각 | `date -Is` |
| host / GPU | 호스트명, `CUDA_VISIBLE_DEVICES`, 카드 모델·메모리 |
| conda env / python / torch | |
| **git 리비전** | `selective_estimated_target_smoothing` 와 `cognitive_workflow_kit` **둘 다**, `--dirty` 포함 |
| 체크포인트 경로 / 결과 파일 경로 | |
| **전달한 인자 전문** | 스케줄·배칭·공통·방법별·손실별 전부 |
| 사용한 학습 스크립트 / 평가 스크립트 | |
| `launcher argv` | 런처를 부른 명령 그대로 |
| `PYTORCH_CUDA_ALLOC_CONF` | |

conda 활성화 **뒤에** 기록해야 env 이름과 torch 버전이 찍힌다.

5090 쪽 구현은 `run_pas.sh` 의 `LAUNCH` 블록이다. 그대로 가져다 쓰거나
같은 항목을 남기면 된다.

---

## 평가 — 길이 필터를 걸지 마십시오

⚠ **이 버그는 지금도 코드에 있습니다.** 고친 것이 아니라 우회하는 것입니다.
`wav2vec2_finetuning_pas.py` 의 **1393 / 1403 / 1410 행**이 `eval_datasets`
에도 훈련용 길이 상한을 넘깁니다. 그 결과 dev 가 **2703/2864 -> 2694/2857**
로 줄었습니다. 그쪽 120셀이 `run_inference_sweep.sh` 의 `BATCH=${BATCH:-8}`
때문에 못 쓰게 된 일도 있었습니다.

| 시점 | 대상 | 방법 |
|---|---|---|
| 중간 | dev | 훈련 루프 내장 (진행 확인용, **결과에 넣지 않음** — 필터 걸린 값) |
| 훈련 종료 후 | dev-clean, dev-other, test-clean, test-other **전체** | `wav2vec2_inference.py` **배치 추론** |

- 최종 평가는 **`wav2vec2_inference.py` 로만**. 길이 필터가 없습니다
- **`--batch_size` 는 키워도 됩니다.** `large-lv60` 은
  `feat_extract_norm="layer"` 라 base 의 group-norm 문제가 없습니다.
  이번에 large 를 쓰는 이유가 그것입니다
- **평가 발화 수 `n` 을 반드시 기록.** 전체 dev = 2703/2864,
  전체 test = 2620/2939. 이 수가 아니면 필터가 걸린 것입니다
- **훈련 직후 같은 큐에서** 평가까지 끝내 주십시오. 별도 프로세스여도
  됩니다 — 요지는 나중으로 미루지 않는 것입니다


### RNN-T 평가는 스크립트가 다르다

⚠ **`wav2vec2_inference.py` 는 CTC 전용이다.** HF `pipeline()` 으로 CTC
체크포인트를 디코드하며 RNN-T 체크포인트는 읽지 못한다. RNN-T 는
**`rnnt_decode_test.py`** 를 쓴다 — 길이 필터가 없고 `n` 을 기록한다.

```bash
# CTC
python wav2vec2_inference.py --checkpoint_dir <ckpt> --vocab_size 32 \
    --test_split dev-clean --batch_size 8

# RNN-T
python rnnt_decode_test.py --device cuda --batch_size 8 \
    --splits dev-clean,dev-other,test-clean,test-other \
    --ckpt_glob '<모델디렉터리>/<이름>/rnnt.pt' --out <이름>_decode.jsonl
```

**`rnnt_decode_test.py` 를 이번에 두 군데 고쳤으니 반드시 pull 할 것.**

1. 모델 재구성 때 `encoder_name` 을 안 넘겨 **항상 `wav2vec2-base` 로
   만들었다.** large 체크포인트는 shape 불일치로 못 읽는다
2. **CPU 전용이었다.** `--device` 를 냈다 (기본 `cpu`, 기존 동작 불변).
   ⚠ **CPU 와 GPU 디코드는 같은 체크포인트에서 0.74% 상대 차이가 측정됐다**
   — 보고하려는 효과와 같은 자릿수다. **한 표의 모든 셀이 같은 `--device`
   를 써야 한다. 이번 실험은 전부 `cuda` 로 통일한다**

**RNN-T 훈련 루프 내장 eval 도 길이 필터가 걸린다** (`wav2vec2_rnnt.py:486-490`).
게다가 dev 만 돈다. 결과 파일에 넣지 말 것.

---

## 결과 형식

`/mnt/data/home/chanwcom/results/RESULT_LS.md` 에 **런이 끝날 때마다 한 줄씩**.

| 열 |
|---|
| method (`LS` 또는 `baseline`) |
| loss (CTC / RNN-T) |
| alpha, seed |
| **파인튜닝 셋 = `LibriSpeech 100h`** (반드시 명시) |
| 체크포인트 절대경로 |
| dev-clean WER + n, dev-other WER + n |
| test-clean WER + n, test-other WER + n |

파일 머리에 **"배치 추론으로 측정"** 과 **모델 이름**을 적어 주십시오.

NAS 를 못 쓰시므로 결과 MD 내용을 주기적으로 보내 주시면 저희가 중앙 표에
합치겠습니다. 체크포인트는 그쪽에 두셔도 됩니다.

---

## 실행 예

**인자 이름이 두 스크립트가 다릅니다.**

| | CTC | RNN-T |
|---|---|---|
| 출력 | `--checkpoint_top_dir` + `--run_name` | `--output_dir` (전체 경로) |
| eval 크기 | `--per_device_eval_batch_size` | `--eval_batch_size` |

### CTC

```bash
NAME=ls_ctc_libri100hr_alpha_0p01_s0
python wav2vec2_finetuning_pas.py \
    --model_name facebook/wav2vec2-large-lv60 \
    --alpha_mode floored_active_support --alpha 0.01 --beta 1.0 \
    --smoothing_space class --fas_eps 1e-10 --vocab_size 32 --seed 0 \
    --finetune_profile libri_speech_clean_100hr_wsd \
    --max_steps 15000 --warmup_steps 1000 --num_stable_steps 11000 --num_decay_steps 3000 \
    --learning_rate 5e-5 \
    --dynamic_batching --max_batch_audio_len 1600000 --max_sample_audio_len 480000 \
    --per_device_eval_batch_size 4 --dataloader_num_workers 4 \
    --checkpoint_top_dir /mnt/data/home/chanwcom/models --run_name $NAME \
  2>&1 | tee /mnt/data/home/chanwcom/logs/$NAME.log
```

### RNN-T

```bash
NAME=ls_rnnt_libri100hr_alpha_0p01_s0
python wav2vec2_rnnt.py \
    --model_name facebook/wav2vec2-large-lv60 \
    --alpha_mode floored_active_support --alpha 0.01 --beta 1.0 \
    --fas_eps 1e-10 --vocab_size 32 --seed 0 \
    --finetune_profile libri_speech_clean_100hr_wsd \
    --max_steps 15000 --warmup_steps 1000 --num_stable_steps 11000 --num_decay_steps 3000 \
    --learning_rate 5e-5 \
    --dynamic_batching --max_batch_audio_len 1600000 --max_sample_audio_len 480000 \
    --max_label_len 450 --eval_batch_size 4 --dataloader_num_workers 4 --grad_accum 2 \
    --output_dir /mnt/data/home/chanwcom/models/$NAME \
  2>&1 | tee /mnt/data/home/chanwcom/logs/$NAME.log
```

**baseline** 은 `--alpha_mode fixed --alpha 0.0 --beta 0.0`, 이름은
`baseline_{ctc,rnnt}_libri100hr_s{SEED}`.

---

## 왜 이 실험이 필요한가

PAS-H / PAS-U 가 얼마나 좋은지는 **전통 LS 대비**로만 말할 수 있습니다.
지금 있는 LS 결과는 **wav2vec2-base** 로 뽑은 것이라 large 기반 PAS 와
같은 표에 둘 수 없습니다. 같은 모델·같은 스케줄·같은 격자·같은 평가로
뽑은 LS 가 있어야 비교가 성립합니다.

참고로 기존 base 결과에서 CTC 100h 는 LS 최적이 alpha=0.25 근처였고
FAS(활성 제한)가 alpha=0.2 에서 그것을 이겼습니다. large 에서도 같은지가
이번 실험의 질문 중 하나입니다.

---

## 지금 할 것

1. `git pull` (`cwk`, `selective_estimated_target_smoothing` 둘 다)
2. `/mnt/data/home/chanwcom/{logs,results}` 생성
3. **baseline 2런** (CTC/RNN-T, seed 0) 으로 경로·평가 파이프라인 점검
4. 점검이 끝나면 seed 0 으로 9점 스윕

돌던 AWS 큐는 끊지 마십시오. 그 뒤에 들어가면 됩니다.
막히거나 인자 이름이 다르면 알려 주십시오.
