# PAS 스윕 — 4090 (`slpl_4090_00`) 작업 지시

2026-09-28 개정. 전체 사양은 `PAS_SPEC.md` (같은 디렉터리). 이 문서는 4090
몫만 추린 것이다.

## 이 개정에서 바뀐 것

| 항목 | 이전 | 지금 |
|---|---|---|
| 배치 | `--gpu_profile 4090` 기본값 | **동적 배칭 `--max_batch_audio_len 1600000`** |
| learning rate | 1e-4 | **5e-5** |
| `--gpu_memory_fraction` | 언급 없음 | **넘기지 않는다** (24GB 는 1런) |
| eval 배치 | 언급 없음 | **훈련과 따로 4** |
| GPU 상한 | 2장 | **3장** |
| 실행 예 | `--output_dir` (CTC 에 없는 인자) | 아래 §실행 예 |
| RNN-T | "스크립트 없음" | **`wav2vec2_rnnt.py` 에 이미 있다** |

---

## 맡을 것

**seed 3, 4 전량.** PAS-H · PAS-U · baseline, CTC · RNN-T 양쪽.

```
PAS   9 alpha x 2 방법 x 2 손실 x 2 seed = 72 런
base  1       x 1      x 2 손실 x 2 seed =  4 런
                                    합계  76 런
```

**단, 2단계로 나눈다.** seed 0 파일럿(5090 담당)이 끝나 최적 구간이 정해지면
그 구간만 채운다. **파일럿 결과가 나오기 전에는 baseline 4 런만** 돌린다.

---

## 코드

`cwk` 와 `selective_estimated_target_smoothing` 둘 다 최신을 받는다.
**기존 AWS/FAS 함수는 바이트 무변경**이다 (과거 런 재현성 유지).

| | |
|---|---|
| CTC 손실 | `cwk/loss/pytorch/shc_loss.py: apply_pas_smoothing` |
| RNN-T 손실 | `cwk/loss/pytorch/rnnt_shc_loss.py: apply_pas_diagonal_smoothing` |
| CTC 학습 | `wav2vec2_finetuning_pas.py` |
| **RNN-T 학습** | **`wav2vec2_rnnt.py`** — 별도 PAS 파일이 아니다 |
| 테스트 | `cwk/loss/pytorch/pas_smoothing_test.py` |

**`wav2vec2_rnnt_pas.py` 같은 두 번째 파일을 만들지 말 것.** PAS 는 기존
`wav2vec2_rnnt.py` 에 직접 들어가 있다 (`--alpha_mode pas_h|pas_u`,
`--model_name`, `--eval_batch_size`). 파일이 두 벌 생기면 이번 재실험이
고치려는 바로 그 분산 문제가 된다.

**받은 직후 테스트를 돌릴 것.**

```bash
cd <cwk>/cwk/loss/pytorch
python pas_smoothing_test.py      # 7/7 이어야 함. pytest 아님
```

7/7 이 아니면 **훈련을 시작하지 말고** 알릴 것.

---

## 알고리즘 요약

기존 AWS 와 두 군데가 다르다. 이름이 비슷하다고 같은 것으로 취급하면 안 된다.

1. **CTC 는 blank 를 지지집합에서 뺀다.** 확장 레이블열이
   `c0 b c1 b ... c_{L-1}` (`boundary_blanks=False`) 이므로 **짝수 위치만**
   지지집합이다. 기존 AWS 는 blank 를 포함했다.
2. **기준 폭이 고정이다.** CTC 는 `N = L`(비-blank 수, `2L-1` 아님),
   RNN-T 는 `N = T + L`(대각선 길이 아님). 기존 RNN-T AWS 는
   `alpha / n_valid` 로 대각선마다 높이가 달랐다.

```
PAS-H : (1 - alpha*N_a/N) * r + (alpha/N) * 1_active     활성만
PAS-U : (1 - alpha*N_v/N) * r + (alpha/N) * 1_valid      비-blank 전체
```

**축소는 지지집합이 아니라 벡터 전체에 건다.** blank 를 뺀 뒤로는 이게
중요하다 — blank 가 질량의 90% 를 쥔 프레임에서 지지집합만 줄이면
alpha=0.16 부터 계수가 음수가 되어 합이 1 을 넘는다 (실측 1.75).

`--beta` 는 읽지 않는다. 넘기면 assert 로 막힌다.

---

## 훈련 설정

| 항목 | 값 |
|---|---|
| 모델 | `facebook/wav2vec2-large-lv60` (SSL 전용) |
| 파인튜닝 셋 | LibriSpeech train-clean-100 (100h) |
| 총 스텝 | 15,000 |
| 스케줄 | WSD warmup 1,000 / stable 11,000 / decay 3,000 |
| optimizer | AdamW, lr **5e-5** (CTC · RNN-T 동일) — 아래 참조 |
| `--fas_eps` | **1e-10** (기본값이지만 명시할 것) |

**`wav2vec2-large-960h-lv60-self` 를 쓰면 안 된다** — 이미 960h 로 ASR
파인튜닝된 모델이라 100h 연구에 쓸 수 없다. RNN-T 인코더도 large 로 통일한다.

⚠ **`--finetune_profile` 만 주면 스케줄이 1000/10000/4000 이 된다.**
프로파일 기본값이 그렇다. 사양의 1000/11000/3000 을 쓰려면 **네 값을 명시로
덮어쓸 것.** 명시 인자가 프로파일을 이기고, 세 값이 `max_steps` 와 안 맞으면
assert 로 죽으므로 조용히 틀릴 일은 없다. 프로파일 자체는 계속 쓴다 —
`train_subdir` 을 거기서 받는다.


### ⚠ `--gpu_memory_fraction` — **24GB 카드에는 넘기지 않는다**

| 카드 | 동시 | 플래그 |
|---|---|---|
| 5090 (31.4 GiB) | 2런 | `--gpu_memory_fraction 0.46` |
| **4090 (23 GiB)** | **1런** | **넘기지 않는다. 카드 전체를 쓴다** |

이 플래그는 **한 카드에 두 런을 올릴 때**만 쓴다. 24GB 카드는 애초에 한 런
밖에 못 올리므로, 거기에 0.46 을 주면 그 한 런을 10.6 GiB 로 묶어 버린다 —
14.4 GiB 가 필요한 런이 **카드에 12 GiB 가 남은 채로** 첫 backward 에서
죽는다.

**두 스크립트의 동작이 달랐다.** CTC 는 30 GiB 미만이면 조용히 무시했고
(`_MEMORY_FRACTION_MIN_GIB`), RNN-T 는 **무조건 적용**했다. 같은 명령줄이
한쪽에서는 안전하고 다른 쪽에서는 치명적이었다는 뜻이다. RNN-T 에도 같은
가드를 넣었으니 이제 양쪽 다 무시하지만, **애초에 넘기지 않는 것이 맞다.**

`run_pas.sh` 는 머신이 아니라 **카드 용량을 보고** 판단한다
(`nvidia-smi --query-gpu=memory.total`), 그래서 5090 과 4090 에서 같은
스크립트가 그대로 맞는다.

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

### 배치 — **반드시 동적 배칭. 값은 실측 고정이다**

```
--dynamic_batching --max_batch_audio_len 1600000 --max_sample_audio_len 480000
CTC   추가:  --per_device_eval_batch_size 4
RNN-T 추가:  --eval_batch_size 4
```

**CTC · RNN-T 가 같은 값을 쓴다. 바꾸지 말 것.**

`--dynamic_batching` 을 켜지 않으면 `--gpu_profile` 의
`per_device_train_batch_size=24` 가 들어간다. 그 값은 **base 인코더 기준**
이라 large 에서 OOM 한다 — 그쪽이 겪은 것이 이것이다. 기존 100h 실험도
전부 동적 배칭이었다.

**실측** (5090 에 `--gpu_memory_fraction 0.46` = 14.4 GiB 상한을 걸어
24GB 카드 조건을 재현한 값):

| | 피크 |
|---|---|
| RNN-T @ 1,600,000 (eval 포함) | **13.68 alloc / 14.42 GiB reserved — 통과** |
| RNN-T @ 2,400,000 | 21.5 GiB |
| CTC @ 1,600,000 | 여유 |
| 고정배치 24 (동적배칭 꺼짐) | **20.2 GiB 에서 OOM** |

**카드당 동시 실행**

| 카드 | 동시 |
|---|---|
| 4090 (23 GiB) | **1개** — 14.42 < 23, 여유 8.6 GiB |
| 5090 (31.4 GiB) | 2개 |

**그쪽은 카드당 1개, 최대 3장 = 동시 3런.** 나머지 GPU 는 다른 사람 몫으로
비워 둘 것.

**eval 배치를 따로 낮추는 이유**: eval 이 메모리 피크다. RNN-T 는 eval
DataLoader 가 `batch_size=32` 로 하드코딩돼 있어 large 에서 한 번에 4.07 GiB
를 요구했고, 훈련이 13.9 GiB 로 잘 돌던 런을 eval 에서 죽였다.
`--eval_batch_size` 인자를 새로 냈다 (기본 32 로 기존 레시피 불변). eval
배치 크기는 WER 에 영향이 없다.

---

## 산출물 — 이름 규칙을 반드시 지킬 것

이번 재실험의 목적이 알고리즘보다 **산출물 규율**이다. 현재 867 런 중
test 가 있는 것이 347 개뿐이고, 같은 방법이 네 군데에 흩어져 있으며 태그가
두 형식으로 공존한다.

```
체크포인트  /mnt/synology_nas_00/chanwcom/models/pas_{h,u}_{ctc,rnnt}_libri100hr_alpha_{A}_s{SEED}
baseline    /mnt/synology_nas_00/chanwcom/models/baseline_{ctc,rnnt}_libri100hr_s{SEED}
로그        /mnt/synology_nas_00/chanwcom/logs/<체크포인트 디렉터리명>.log
```

- `{A}` 는 소수점을 `p` 로: `0.00125 -> 0p00125`, `0.32 -> 0p32`
- **`/tmp` 금지.** 처음부터 NAS 에 쓴다
- `run_args.json` 은 스크립트가 훈련 시작 전에 자동으로 남긴다 (CTC·RNN-T 양쪽)
- 이름 형식은 **한 가지만**


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

## 평가 — 길이 필터를 걸지 말 것

⚠ **이 버그는 지금도 코드에 그대로 있다.** 고친 것이 아니라 우회하는 것이다.
`wav2vec2_finetuning_pas.py` 의 **1393 / 1403 / 1410 행**이 `eval_datasets`
에도 `max_sample_length=args.max_sample_audio_len` 을 넘긴다. 그 결과 dev 가
**2703/2864 -> 2694/2857** 로 줄었다.

| 시점 | 대상 | 방법 |
|---|---|---|
| 중간 | dev | 훈련 루프 내장 (진행 확인용, **결과 파일에 넣지 않는다** — 필터 걸린 값) |
| 훈련 종료 후 | dev-clean, dev-other, test-clean, test-other **전체** | `wav2vec2_inference.py` **배치 추론** |

- 최종 평가는 **`wav2vec2_inference.py` 로만**. 이 스크립트에는 길이 필터가 없다
- **`--batch_size` 는 키워도 된다.** `large-lv60` 은
  `feat_extract_norm="layer"` 라 base 의 group-norm 문제가 없다
- **평가 발화 수 `n` 을 반드시 기록.** 전체 dev = 2703/2864,
  전체 test = 2620/2939. 이 수가 아니면 필터가 걸린 것이다
- **훈련 직후 같은 잡(또는 같은 큐)에서** 평가까지 끝낼 것


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

## 결과

```
/mnt/synology_nas_00/chanwcom/results/RESULT_PAS.md
```

**런 하나가 끝날 때마다 한 줄씩** 추가한다. 몰아서 하면 빠진다.

| 열 |
|---|
| method (PAS-H / PAS-U / baseline) |
| loss (CTC / RNN-T) |
| alpha, seed |
| **파인튜닝 셋 = `LibriSpeech 100h`** (반드시 명시) |
| 체크포인트 절대경로 |
| dev-clean WER + n, dev-other WER + n |
| test-clean WER + n, test-other WER + n |

`verified.jsonl` 에는 **넣지 말 것.** 그 파일은 `RESULTS.md` 의 소스이고
bs=1 값만 받는다. PAS 는 배치 추론이라 절대값 스케일이 다르다.

---

## 실행 예

**인자 이름이 두 스크립트가 다르다.**

| | CTC | RNN-T |
|---|---|---|
| 출력 | `--checkpoint_top_dir` + `--run_name` | `--output_dir` (전체 경로) |
| 배치 | `--per_device_train_batch_size` | `--batch_size` |
| 누적 | `--gradient_accumulation_steps` | `--grad_accum` |
| 스무딩 공간 | `--smoothing_space label` | (없음) |
| eval 크기 | `--per_device_eval_batch_size` | `--eval_batch_size` |

### CTC

```bash
NAME=pas_h_ctc_libri100hr_alpha_0p01_s3
python wav2vec2_finetuning_pas.py \
    --model_name facebook/wav2vec2-large-lv60 \
    --alpha_mode pas_h --alpha 0.01 --smoothing_space label --fas_eps 1e-10 \
    --vocab_size 32 --seed 3 \
    --finetune_profile libri_speech_clean_100hr_wsd \
    --max_steps 15000 --warmup_steps 1000 --num_stable_steps 11000 --num_decay_steps 3000 \
    --learning_rate 5e-5 \
    --dynamic_batching --max_batch_audio_len 1600000 --max_sample_audio_len 480000 \
    --per_device_eval_batch_size 4 --dataloader_num_workers 4 \
    --checkpoint_top_dir /mnt/synology_nas_00/chanwcom/models --run_name $NAME \
  2>&1 | tee /mnt/synology_nas_00/chanwcom/logs/$NAME.log
```

### RNN-T

```bash
NAME=pas_h_rnnt_libri100hr_alpha_0p01_s3
python wav2vec2_rnnt.py \
    --model_name facebook/wav2vec2-large-lv60 \
    --alpha_mode pas_h --alpha 0.01 --fas_eps 1e-10 --vocab_size 32 --seed 3 \
    --finetune_profile libri_speech_clean_100hr_wsd \
    --max_steps 15000 --warmup_steps 1000 --num_stable_steps 11000 --num_decay_steps 3000 \
    --learning_rate 5e-5 \
    --dynamic_batching --max_batch_audio_len 1600000 --max_sample_audio_len 480000 \
    --max_label_len 450 --eval_batch_size 4 --dataloader_num_workers 4 --grad_accum 2 \
    --output_dir /mnt/synology_nas_00/chanwcom/models/$NAME \
  2>&1 | tee /mnt/synology_nas_00/chanwcom/logs/$NAME.log
```

**baseline** 은 `--alpha_mode fixed --alpha 0.0`, 이름은
`baseline_{ctc,rnnt}_libri100hr_s{SEED}`.

---

## 지금 할 것

1. `git pull` 후 `pas_smoothing_test.py` 7/7 확인
2. **baseline 4 런** (CTC/RNN-T x seed 3,4) 시작. 카드당 1개
3. 5090 의 seed 0 파일럿 결과를 기다렸다가 최적 구간만 채우기

테스트가 7/7 이 아니면 반드시 먼저 알릴 것.
