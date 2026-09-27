# PAS (Post-Alignment Smoothing) — 실험 사양서

2026-09-28 개정본. 코드를 직접 읽어 대조했고 메모리 값은 실측이다.

> **관련 문서**
>
> | 문서 | 경로 | 받는 곳 |
> |---|---|---|
> | 이 사양서 | `selective_estimated_target_smoothing/PAS_SPEC.md` | 전체 |
> | 4090 지시문 | `selective_estimated_target_smoothing/PAS_TASK_4090.md` | `slpl_4090_00` |
> | 인하대 지시문 | `selective_estimated_target_smoothing/PAS_TASK_INHA.md` | `inha_4090_00` |
>
> 세 파일 모두 저장소 최상위에 있고 git 에 올라간다. 인하대는 NAS 접근이
> 안 되므로 **git 으로만** 받는다.

## 0. 이 개정본에서 바뀐 것

| 항목 | 이전 | 지금 | 사유 |
|---|---|---|---|
| 배치 | `--gpu_profile 4090` 기본값 | **동적 배칭 `--max_batch_audio_len 1600000`** (§4.2) | 고정배치 24 가 large 에서 OOM (실물 4090 확인). 값은 실측으로 고정 |
| eval 배치 | 훈련과 동일 | **훈련과 따로, 4** (§4.2) | RNN-T eval 이 `batch_size=32` 하드코딩이라 4.07 GiB 를 한 번에 요구 |
| GPU 상한 | 머신당 2장 | **머신당 3장** | 사용자 지시 |
| learning rate | 1e-4 | **5e-5** (§4.2a) | 배치가 1/4 이 되어 sqrt 스케일링으로 절반 |
| 구현 상태 | "둘 다 새로 짜야 한다" | **완료, 테스트 7/7** (§3) | |
| 평가 필터 | "사고가 났다" (과거) | **현재도 코드에 있다. CTC·RNN-T 양쪽** (§7.0) | 고친 게 아니라 우회하는 것 |
| RNN-T 최종 평가 | `wav2vec2_inference.py` | **`rnnt_decode_test.py`** (§7) | 전자는 CTC 전용이라 RNN-T 를 못 읽는다 |
| 실행 인자 | 없음 | **§4.3 신설** | CTC 와 RNN-T 의 인자 이름이 다르다 |

---

## 1. 목적

정렬 사후확률을 `[T, L]` 평면에서 스무딩하는 **두 방법**(PAS-H, PAS-U)을
LibriSpeech 100h · wav2vec2-large 로 CTC · RNN-T 양쪽에서 스윕한다.

### 1.1 이번 재실험의 진짜 목적 — 산출물 규율

알고리즘 자체보다 **결과가 정리되지 않은 것**이 재실험의 이유다. 현 상태:

- 런 867개 중 test 가 있는 것은 **347개뿐** (`results/INVENTORY.md`)
- CTC 10h 113런, RNN-T 10h 88런은 완주했는데 **test 0건**
- 체크포인트가 `/tmp` 에만 있던 런 5개 (구조됨)
- 같은 방법이 `u24` / `u22` / `inha_4090` / scratchpad 네 곳에 흩어짐
- 태그가 두 형식 공존 (`100hr_LS_a0.05_s0` vs `rnnt_100h_LS_a0.005_s0`)
- 색인(`checkpoints.jsonl`)이 실물·평가표와 어긋남

**절대 규칙**

| 규칙 | 이유 |
|---|---|
| 체크포인트는 **처음부터 NAS**. `/tmp` 금지 | scratchpad 런이 소실될 뻔했다 |
| 이름에 **method·loss·alpha·seed 전부** | seed 없는 이름은 서로 덮어쓴다 |
| 이름 형식은 **한 가지만** | 두 형식이 공존하면 조회가 한쪽을 놓친다 |
| 로그명 = 체크포인트 디렉터리명 + `.log` | 이름만으로 대응을 찾는다 |
| 훈련 직후 **같은 잡 안에서** 최종 평가까지 | 평가를 미루면 안 돈다 (현 347/867) |
| 런 하나 끝날 때마다 결과 파일에 **한 줄 추가** | 몰아서 하면 빠진다 |
| `run_args.json` 을 체크포인트 옆에 남긴다 | 이름만으로 방법을 추정하면 판별불가가 생긴다 |

한 줄로: **런이 끝나면 체크포인트·로그·결과 한 줄이 제자리에 있고, 셋을
이름만으로 서로 찾을 수 있어야 한다.**

---

## 2. 알고리즘

### 2.1 스무딩 대상 벡터

| 손실 | 정의 | 묶는 축 |
|---|---|---|
| CTC | `γ_{t,l} = P(q_t = l \| X, θ)` | **프레임 t 고정**<br>`r_t = [γ_{t,0}, …, γ_{t,L'-1}]` |
| RNN-T | `γ_{t,l} = P(r_s = t, q_s = l \| X, θ)`, `s = t+l` | **반대각선 s 고정**<br>`r_s = [γ_{s,0}, …, γ_{0,s}]` |

### 2.2 활성 집합 — **blank 제외 (CTC)**

`ShcLoss` 는 `to_blank_augmented_labels(inputs, 0, False, False)` 를 부르므로
`boundary_blanks=False` 이고, 확장 레이블열은

```
c₀  ∅  c₁  ∅  c₂  …  c_{L-1}          폭 L' = 2L−1
짝수 위치 = 레이블,  홀수 위치 = blank
```

**활성 정의**

```
CTC    : active(t) = { l : l 이 짝수(비-blank) AND γ_{t,l} > eps }
RNN-T  : active(s) = { (t,l) : t+l = s, 유효 노드, γ_{t,l} > eps }
```

`eps = 1e-10`.

**왜 blank 를 빼는가**: 확장 레이블열은 위치의 약 절반이 blank 이고, CTC 는
대부분의 프레임에서 blank 가 지배적이다. 활성 집합에 blank 를 넣으면
스무딩 질량의 절반 가까이가 blank 로 간다 — 스무딩이 "어느 토큰을 낼지"가
아니라 "얼마나 blank 를 낼지"를 건드리게 된다.

### 2.3 기준 폭 `N` — **고정 상수**

```
CTC    : N = L          (비-blank 위치 수 = 전사 길이).  2L−1 이 아니다
RNN-T  : N = T + L      (발화 전체 길이. 대각선 길이가 아니다)
```

CTC 에서 `2L−1`(전체 폭)이 아니라 `L` 인 이유: 스무딩이 **비-blank 위치에만**
작용하므로 기준 폭도 그 지지집합의 크기여야 한다. `2L−1` 을 쓰면 blank 에
줄 몫까지 분모에 넣는 셈이 되어 실효 높이가 절반으로 줄어든다.

RNN-T 에서 반대각선 `s` 의 길이는 `1 → min(T, L+1) → 1` 로 **발화 안에서
변한다.** 대각선 길이를 쓰면 높이가 대각선마다 달라져 "높이 보존"이 성립하지
않는다. `T + L` 은 발화당 상수이므로 **모든 대각선에서 높이가 같다.**

### 2.4 두 방법

두 방법은 **높이가 같고 지지집합만 다르다.** 그래서 비교가 정확히
"활성만 vs 전체" 한 축이 된다.

| | 혼합 | 지지집합 | 활성 높이 |
|---|---|---|---|
| **PAS-H** | `(1 − α·N_a/N)·r + (α/N)·1_active` | **활성만** | `α/N` |
| **PAS-U** | `(1 − α·N_v/N)·r + (α/N)·1_valid` | **유효 전체** | `α/N` |

- `N_a` = 활성 개수, `N_v` = 유효 개수
  - CTC: `N_v` = 비-blank 위치 수 = `L`, 따라서 PAS-U 는 `(1−α)·r + (α/L)·1_nonblank`
  - RNN-T: `N_v` = 그 대각선의 유효 노드 수
- 주입 질량은 `α·N_a/N` (PAS-H), `α·N_v/N` (PAS-U) 로 변한다.
  **높이를 고정하면 질량이 변하는 것이 정의상 당연하다.**

**축소는 지지집합이 아니라 벡터 전체에 건다.** `(1 − α·N_·/N)` 이 `r` 의
**모든** 성분에 곱해진다. blank 를 지지집합에서 뺀 뒤로는 이게 중요하다 —
blank 가 질량의 90% 를 쥔 프레임에서 지지집합만 축소하면 α=0.16 부터
계수가 음수가 되어 합이 1 을 넘는다 (α=1 에서 1.75 로 실측).

**정규화 확인**: `r` 이 1로 합해지므로 결과도
`(1 − α·N_·/N) + α·N_·/N = 1`. `N_a ≤ N_v ≤ N` 이고 `α ≤ 1` 이므로 계수는
항상 비음수다.

**blank 는 어느 방법에서도 질량을 받지 않는다** (CTC). 비-blank 위치끼리만
재분배되며, blank 의 원래 `γ` 는 `(1 − α·N_·/N)` 로 축소되기만 한다.

### 2.5 PAS-W 는 하지 않는다

`(1−α)·r + α·u_a` (질량 고정, 높이 `α/N_a`) 는 이번 스윕에서 제외한다.

---

## 3. 구현 — **완료**

| 방법 | 파일 | 함수 |
|---|---|---|
| CTC | `cwk/loss/pytorch/shc_loss.py` | `apply_pas_smoothing` |
| RNN-T | `cwk/loss/pytorch/rnnt_shc_loss.py` | `apply_pas_diagonal_smoothing` |
| 테스트 | `cwk/loss/pytorch/pas_smoothing_test.py` | 7개 |

`support ∈ {"active", "valid"}` 로 PAS-H / PAS-U 를 가른다.
`--alpha_mode pas_h` / `pas_u` 로 노출된다. `--beta` 는 읽지 않으며 넘기면
assert 로 막힌다.

**기존 AWS / FAS 함수는 바이트 무변경이다.** 과거 런의 재현성을 유지해야
하므로 수정하지 않고 새 함수를 추가했다.

### 3.1 착수 전 필수 — 검증 테스트

pytest 미설치 환경이므로 **`python pas_smoothing_test.py`** 로 실행한다.
**7/7 이 아니면 훈련을 시작하지 않는다.**

1. **정규화** — CTC 모든 `(b,t)`, RNN-T 모든 `(b,s)` 에서 합이 1
2. **높이 불변** — `N_a` 를 바꿔도 활성 증분이 `α/N` 으로 일정
3. **RNN-T 대각선 간 높이 동일** — `T+L` 로 바꾼 이유이므로 반드시
4. **blank 불수령** — CTC 홀수 위치는 증가하지 않는다
5. **α=0 항등** — 비트 단위로 동일
6. **경계** — `N_a = 0` 인 프레임, 길이 1 대각선, α=1 에서 NaN/음수 없음

---

## 4. 훈련 설정

### 4.1 공통

| 항목 | 값 |
|---|---|
| 모델 | **`facebook/wav2vec2-large-lv60`** (SSL 전용, `Wav2Vec2ForPreTraining`) |
| 파인튜닝 셋 | **LibriSpeech train-clean-100 (100h)** |
| 총 스텝 | 15,000 |
| 스케줄 | **WSD** warmup 1,000 / stable 11,000 / decay 3,000 |
| 옵티마이저 | AdamW |
| learning rate | **5e-5** (CTC · RNN-T 동일) — §4.2a |
| eps | 1e-10 |

**모델 선택 근거**: `large-960h-lv60-self` 는 이미 960h ASR 파인튜닝된
모델이라 100h 연구에 쓸 수 없다. `large-lv60` 은 `Wav2Vec2ForPreTraining`
— SSL 만 된 것이다.

**배치 추론 가능 여부**: `large-lv60` 은 `feat_extract_norm="layer"` 라
시간축 통계를 쓰지 않는다. `base` 의 `"group"` 은 패딩이 유효 프레임 출력을
바꿔 bs=1 이 필수였지만 **large 는 해당 없다.**

RNN-T 인코더도 large 로 통일한다.

### 4.2 배치 — **동적 배칭. 값은 실측으로 고정했다**

```
--dynamic_batching --max_batch_audio_len 1600000 --max_sample_audio_len 480000
CTC   추가:  --per_device_eval_batch_size 4
RNN-T 추가:  --eval_batch_size 4
```

**CTC · RNN-T 가 같은 값을 쓴다. 이 값은 고정 상수이며 바꾸지 않는다.**

**반드시 `--dynamic_batching` 을 켠다.** 켜지 않으면 `--gpu_profile` 의
`per_device_train_batch_size=24` 가 그대로 들어가고, 이 값은 **base 인코더
기준**이라 `large-lv60` 에서 실물 4090 의 20.2 GiB 에서 OOM 한다. 기존 100h
실험도 전부 동적 배칭이었다.

동적 배칭은 배치당 **총 오디오 길이**를 상한으로 잡으므로 피크가 구성상
고정이다. 고정 개수 배칭은 피크가 배치 운에 달려 있어 긴 발화가 몰리면
열 시간 뒤에도 죽을 수 있다.

**실측** (5090 에 `--gpu_memory_fraction 0.46` = 14.4 GiB 상한을 걸어 잰
값. 실물 4090 과 CTC 3.2M 에서 13.5 vs 13.4 로 일치 확인):

| `--max_batch_audio_len` | CTC | RNN-T (eval 포함) |
|---|---|---|
| **1,600,000 (100초)** | 여유 | **13.68 alloc / 14.42 GiB reserved — 통과** |
| 2,400,000 (150초) | 약 11 GiB | 21.5 GiB |
| 3,200,000 (200초) | 13.5 GiB | 21.9 GiB |
| 4,800,000 (300초) | 17.4 GiB | — |
| 6,400,000 (400초) | 21.3 GiB | — |
| 고정배치 24 (동적배칭 꺼짐) | **20.2 GiB 에서 OOM** | — |

**왜 1,600,000 인가**

| 카드 | 동시 실행 | 근거 |
|---|---|---|
| 5090 (31.4 GiB) | **2개** | 14.42 x 2 = 28.8 < 31.4. `--gpu_memory_fraction 0.46` 으로 검증 |
| 4090 (23 GiB) | **1개** | 14.42 < 23, 여유 8.6 GiB |

**제약은 RNN-T 다.** 같은 예산에서 RNN-T 가 CTC 의 약 1.7배를 쓴다. 같은
배치를 쓰라는 지시이므로 RNN-T 기준으로 정하고 CTC 는 여유를 갖는다 —
손실별로 배치를 따로 잡으면 두 표를 같은 조건으로 읽을 수 없다.

**왜 기존 AWS 의 6,400,000 을 못 쓰는가**: 기존 AWS 100h 런은
`wav2vec2-base` 다 (`run_args.json` 에 `model_name` 이 없다 — 당시 하드코딩).
large 는 파라미터가 3.3배다. 그 AWS 런들도 `gpu_memory_fraction 0.46` 으로
32GB 카드에 두 개씩 올려 하나당 약 14 GiB 를 썼다 — **카드당 2개라는 운용은
그대로 유지하고, 같은 14 GiB 에 들어가도록 예산만 낮춘 것이다.**

**eval 배치는 훈련과 따로 잡는다.** eval 이 메모리 피크다.

- **RNN-T**: eval DataLoader 가 `batch_size=32` 로 **하드코딩**돼 있었다
  (`wav2vec2_rnnt.py:669`). large 에서 32 x 30초는 한 번에 4.07 GiB 를
  요구해, 훈련 스텝이 13.9 GiB 로 잘 돌던 런을 eval 에서 죽였다.
  **`--eval_batch_size` 인자를 새로 냈다** (기본값 32 로 기존 레시피 불변).
- **CTC**: 프로파일 기본값 12 는 12 x 30초 = 360초 분량으로 훈련 예산
  100초의 3.6배다.

eval 배치 크기는 WER 에 영향이 없다 — 한 번에 몇 발화를 묶는지만 바꾸고
훈련 경로를 건드리지 않는다. `--max_batch_audio_len` 변경과 달리 서로 다른
값으로 잰 셀도 비교 가능하다.

### 4.2a learning rate — **5e-5. 1e-4 가 아니다**

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

### 4.3 실행 인자 — **두 스크립트의 이름이 다르다**

| | CTC (`wav2vec2_finetuning_pas.py`) | RNN-T (`wav2vec2_rnnt.py`) |
|---|---|---|
| 출력 | `--checkpoint_top_dir` + `--run_name` | `--output_dir` (전체 경로) |
| 배치 | `--per_device_train_batch_size` | `--batch_size` |
| 누적 | `--gradient_accumulation_steps` | `--grad_accum` |
| 스무딩 공간 | `--smoothing_space label` | (없음 — 격자뿐) |
| eval 배치 | `--per_device_eval_batch_size` | `--eval_batch_size` (이번에 신설) |
| eval 발화 수 | (없음 — dev 전체를 돈다) | `--eval_examples` |

**RNN-T 전용 스크립트를 새로 만들지 않는다.** PAS 는 기존
`wav2vec2_rnnt.py` 에 직접 들어가 있다 (`--alpha_mode pas_h|pas_u`,
`--model_name`). `wav2vec2_rnnt_pas.py` 같은 두 번째 파일을 만들면 이번
재실험이 고치려는 바로 그 분산 문제가 된다.

⚠ **`--finetune_profile` 만 주면 스케줄이 1000/10000/4000 이 된다.**
프로파일 기본값이 그렇다 (`wav2vec2_finetuning_pas.py:294-305`). 사양의
1000/11000/3000 을 쓰려면 **네 값을 명시로 덮어써야 한다.** 명시 인자가
프로파일을 이기고(`:1275-1288`, `if getattr(args, key) is None` 일 때만
채운다), 세 값이 `max_steps` 와 안 맞으면 assert 로 죽으므로 조용히 틀릴
일은 없다.

프로파일 자체는 계속 쓴다 — `train_subdir` 을 거기서 받는다.

---

## 5. 스윕

### 5.1 α 격자 — 로그-2, 9점, 공용

```
0.00125  0.0025  0.005  0.01  0.02  0.04  0.08  0.16  0.32
```

방법·손실 공용 (논문 표를 위해 단일 격자). 범위 256배.

**참고 — 기존 최적점** (공간·정의·모델이 달라 직접 비교는 안 되나 범위 근거):

| | 최적 α | 공간 | 모델 |
|---|---|---|---|
| RNN-T 100h AWS | 0.01 (−1.62%, t=−2.90, 5시드 test) | 레이블, 활성만 | base |
| CTC 100h FAS | 0.2 (−2.65%, 5시드 dev) | 클래스, 활성만 | base |
| CTC 100h LS | 0.25 | 클래스, 전체 | base |
| CTC 10h LS | 0.05 | 클래스, 전체 | base |

⚠ **최적점이 격자 끝(0.00125 또는 0.32)에 붙으면 결론을 내리지 말고 그
방향으로 확장한다.** "범위 밖이라 나쁜 것"과 "방법이 나쁜 것"은 결과만
보고 구분할 수 없다.

⚠ **blank 제외로 `N` 이 `2L−1` → `L` 로 바뀌어 실효 높이가 약 2배가
된다.** 기존 AWS 최적점보다 α 가 낮은 쪽으로 이동할 가능성이 있다.

### 5.2 2단계 진행

**0단계 — baseline**
```
baseline × 2 손실 × 5 seed = 10 런
```
Δ 의 기준이므로 가장 먼저 돌린다.

**1단계 — seed 0 파일럿**
```
9 α × 2 방법 × 2 손실 = 36 런
```
곡선을 보고 방법·손실별 최적 부근 **2–3 점**을 고른다.

**2단계 — 5시드 확정**
```
선택 α × 2 방법 × 2 손실 × seed 1–4 = 32~48 런
```

총 **78–94 런** (baseline 10 포함).

**왜 2단계인가**: 기존 AWS 에서 3시드→5시드 때 값이 크게 움직였다
(RNN-T 100h α=0.05 가 +0.57% → +0.01%). 결론은 5시드로 내되 **어느 α 를
채울지는 파일럿 뒤에 정한다.**

### 5.3 머신 배분

| 머신 | 담당 | seed | 쓸 GPU |
|---|---|---|---|
| 5090 (u24, 이 머신) | PAS-H · PAS-U · baseline | 0, 1, 2 | **최대 3장** |
| 4090 (`slpl_4090_00`) | PAS-H · PAS-U · baseline | 3, 4 | **최대 3장** |
| 인하대 (`inha_4090_00`) | **전통 LS** (비교 기준) | 0–4 | **최대 3장** |

**GPU 는 각 머신 3장을 넘기지 않는다.** 나머지는 다른 사람 몫으로 비워 둔다.

파일럿(seed 0)은 전부 5090.

**baseline (α=0) 도 이 스윕 안에서 돌린다.** 다른 담당의 baseline 은
모델·스케줄·평가 조건이 달라 Δ 계산에 쓸 수 없다. CTC · RNN-T 각각
**5시드**, 이름은 `baseline` 으로 따로 둔다 (§6.1).

---

## 6. 산출물

### 6.1 체크포인트

```
/mnt/synology_nas_00/chanwcom/models/pas_{h,u}_{ctc,rnnt}_libri100hr_alpha_{A}_s{SEED}
/mnt/synology_nas_00/chanwcom/models/baseline_{ctc,rnnt}_libri100hr_s{SEED}
```

`{A}` 는 소수점을 `p` 로: `0.00125 → 0p00125`, `0.32 → 0p32`.
예: `pas_h_rnnt_libri100hr_alpha_0p01_s3`

baseline 은 α 를 이름에 넣지 않는다. `run_args.json` 을 같은 디렉터리에 남긴다
(스크립트가 자동으로 한다).

인하대는 NAS 를 못 쓰므로 `/mnt/data/home/chanwcom/models/` 아래 같은 규칙,
이름은 `ls_{ctc,rnnt}_libri100hr_alpha_{A}_s{SEED}`.

### 6.2 로그

```
/mnt/synology_nas_00/chanwcom/logs/<체크포인트 디렉터리명>.log
```

인하대는 `/mnt/data/home/chanwcom/logs/`.

### 6.3 결과

```
/mnt/synology_nas_00/chanwcom/results/RESULT_PAS.md     (5090, 4090)
/mnt/data/home/chanwcom/results/RESULT_LS.md            (인하대)
```

런 하나가 끝날 때마다 **한 줄씩** 추가한다. 몰아서 하면 빠진다.

| 열 |
|---|
| method (PAS-H / PAS-U / LS / baseline) |
| loss (CTC / RNN-T) |
| alpha, seed |
| **파인튜닝 셋 = `LibriSpeech 100h`** (반드시 명시) |
| 체크포인트 절대경로 |
| dev-clean WER + **n**, dev-other WER + **n** |
| test-clean WER + **n**, test-other WER + **n** |

파일 머리에 **"배치 추론으로 측정"** 과 **모델 이름**을 적는다.

---

## 7. 평가

| 시점 | 대상 | 방법 | 길이 필터 |
|---|---|---|---|
| 중간 | dev | 훈련 루프 내장 | **걸림 — 결과에 쓰지 않는다** |
| 훈련 종료 후 (CTC) | dev-clean, dev-other, test-clean, test-other **전체** | `wav2vec2_inference.py` | 없음 |
| 훈련 종료 후 (RNN-T) | 위와 동일 | **`rnnt_decode_test.py`** | 없음 |

⚠ **`wav2vec2_inference.py` 는 CTC 전용이다.** HF `pipeline()` 으로 CTC
체크포인트를 디코드하며 RNN-T 체크포인트는 읽지 못한다. RNN-T 는
`rnnt_decode_test.py` 가 대응물이고, `greedy_decode` 를 학습 스크립트에서
그대로 가져다 쓴다.

```bash
# CTC
python wav2vec2_inference.py --checkpoint_dir <ckpt> --vocab_size 32 \
    --test_split dev-clean --batch_size 8

# RNN-T
python rnnt_decode_test.py --device cuda --batch_size 8 \
    --splits dev-clean,dev-other,test-clean,test-other \
    --ckpt_glob '/mnt/synology_nas_00/chanwcom/models/<name>/rnnt.pt' \
    --out <name>_decode.jsonl
```

**`rnnt_decode_test.py` 를 이번에 두 군데 고쳤다.**

1. 모델을 재구성할 때 `encoder_name` 을 넘기지 않아 **항상
   `wav2vec2-base` 로 만들었다.** large 체크포인트는 shape 불일치로 못
   읽는다. 이제 체크포인트의 `args["model_name"]` 에서 읽는다
2. **CPU 전용이었다.** `--device` 를 냈다 (기본 `cpu` 로 기존 동작 불변).
   large 인코더로 90런 x 4split 을 CPU 로 도는 것은 현실적이지 않다.
   ⚠ **CPU 와 GPU 디코드는 같은 체크포인트에서 0.74% 상대 차이가 측정됐다**
   — 보고하려는 효과(-2~-4%)와 같은 자릿수다. **한 표 안의 모든 셀은 같은
   `--device` 를 써야 한다.** PAS 는 전부 `cuda` 로 통일한다

### 7.0 ⚠ 최종 평가에서 길이 필터를 걸지 않는다

**이 버그는 지금도 코드에 그대로 있다.** 고친 것이 아니라 우회하는 것이다.
`wav2vec2_finetuning_pas.py` 가 `eval_datasets` 에도 훈련용 길이 상한을
넘긴다 — **1393 / 1403 / 1410 행**:

```python
eval_datasets = {
    name: sample_util.make_dataset(
        top_dir, True, spm_model_path,
        max_sample_length=args.max_sample_audio_len)   # <-- 1410행
    for name, top_dir in eval_top_dirs.items()}
```

그 결과 dev 가 **2703/2864 → 2694/2857** 로 줄었다. 긴 발화가 빠진
집합의 WER 은 낮게 나오며, 필터를 안 건 값과 **같은 표에 둘 수 없다.**

**규칙**

- 최종 평가는 **CTC 는 `wav2vec2_inference.py`, RNN-T 는
  `rnnt_decode_test.py` 로만** 한다. 두 스크립트 다 길이 필터가 없다
  (확인함 — `wav2vec2_inference.py` 의 `filtered_ids` 는 토큰 디코딩용이고,
  `rnnt_decode_test.py` 는 `make_dataset` 에 `max_sample_length` 를 넘기지
  않는다)
- **훈련 루프의 중간 eval 값은 진행 확인용이며 결과 파일에 넣지 않는다.**
  그 값은 필터가 걸린 값이다. **RNN-T 도 마찬가지다** —
  `wav2vec2_rnnt.py:486-490` 의 `dev_sets` 도 `max_sample_length` 를 받는다.
  RNN-T 는 dev 만 돌기도 해서 test 가 아예 없다
- 결과 파일에 **평가 발화 수 `n` 을 반드시 함께 적는다.**
  전체 dev = 2703 / 2864, 전체 test = 2620 / 2939. 이 수가 아니면 필터가
  걸린 것이다
- **훈련 직후 같은 잡(또는 같은 큐)에서** 평가까지 끝낸다. 별도 프로세스여도
  되지만 나중으로 미루면 안 된다 — 그래서 지금 867런 중 347런만 test 가 있다

**`verified.jsonl` 에는 넣지 않는다.** 그 파일은 `RESULTS.md` 의 소스이고
**bs=1 값만** 받는다. PAS 는 배치 추론이라 절대값 스케일이 다르다 — 과거
bs=8 값을 `DISCARDED_bs8/` 로 분리한 것과 같은 이유다.

### 7.1 진단 로깅

두 방법의 차이가 지지집합에서 나오므로 훈련 중 다음을 남긴다.

- **`N_a / N_v` 평균** (스텝별). PAS-H 와 PAS-U 의 실효 질량 차이가 여기서
  결정된다. `_LAST_ACTIVE_SUPPORT_STATS["sum_n_a"]` 가 이미 있다
- **RNN-T 대각선 길이 분포** — `T+L` 기준이 의도대로 작동하는지 확인
- **blank 가 받는 질량** — 정의상 0 이어야 한다. 0 이 아니면 blank 마스크
  버그다

---

## 8. 선행 작업

**CTC 100h FAS 25셀 test 디코드.**

FAS(클래스 축 활성만)는 CTC 100h **dev 에서 α=0.2, −2.65%** 로 LS(−1.69%)를
이긴다. 그런데 `RESULTS.md` 에 test 가 **0건**이다 (100h 49런, 10h 70런 전부
dev 만). 훈련은 이미 끝나 있으므로 디코드만 하면 된다.

이건 `base` 결과라 PAS(large)와 같은 표에 들어가지 않는다. 기존 논문 표를
채우는 별도 작업이다.

---

## 9. 예상 (검증 대상)

100h 기준. **틀리면 기록으로 남겨 반증한다.**

| | 1위 | 2위 |
|---|---|---|
| CTC | PAS-H | PAS-U |
| RNN-T | PAS-H | PAS-U |

**근거**: 100h 에서 CTC·RNN-T 모두 활성 제한이 이겼다 (CTC FAS −2.65%,
RNN-T AWS −1.62%). 단 CTC **10h** 에서는 FAS 가 전 구간 +6.8~9.7% 로
해로웠다 — 데이터 양에 따라 뒤집힌다.

**유보**: CTC 근거가 전부 **클래스 축**에서 온 유추다. 레이블 축은 성격이
다르고, 이번에는 blank 까지 뺐으므로 기존 결과의 외삽이 더 약해진다.
PAS-U 가 CTC 에서 1위로 갈 여지가 있다.

**전통 LS 와의 비교가 본 결론이다.** 지금 있는 LS 결과는 base 로 뽑은
것이라 large 기반 PAS 와 같은 표에 둘 수 없다. 그래서 인하대가 같은
모델·스케줄·격자·평가로 LS 를 다시 돌린다 (`PAS_TASK_INHA.md`).

---

## 10. 미해결

없음. 다음이 확정되었다.

- CTC 의 `N` = `L` (비-blank 위치 수) — §2.3
- baseline 은 이 스윕 안에서 CTC · RNN-T 각 5시드 — §5.2, §6.1
- 배치 = 동적 배칭 `--max_batch_audio_len 1600000`, CTC·RNN-T 공통 — §4.2
- learning rate = **5e-5** (배치 1/4 에 맞춘 sqrt 스케일링) — §4.2a
- GPU 상한 = 머신당 3장 — §5.3
