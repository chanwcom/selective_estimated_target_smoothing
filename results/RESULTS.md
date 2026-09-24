# 검증된 test 결과 — 단일 출처

**이 파일만 믿으십시오.** 훈련 로그에서 긁은 값, dev 값, beam search 값은
들어오지 않습니다. 사람이 손으로 고치지 마십시오 — `make_results.py` 가
다시 씁니다.

    python /mnt/synology_nas_00/chanwcom/results/make_results.py

마지막 생성: 2026-09-24 13:51 UTC

체크포인트가 어디 있는지, 디렉터리 이름이 어느 방법을 뜻하는지는
**[CHECKPOINTS.md](CHECKPOINTS.md)** 를 보십시오. 특히 **전통 LS 는
`fas` 라는 이름의 디렉터리에 들어 있습니다** (FAS + beta=1.0 이 textbook
LS 와 bit-exact). 거기서 헷갈리면 이 표를 잘못 읽게 됩니다.

## 들어올 수 있는 줄의 조건

1. **`wav2vec2_inference.py`** (CTC) 또는 **`rnnt/eval_ckpt.py`** (RNN-T) 의
   출력일 것. 훈련 루프 eval 값은 제외 — 같은 체크포인트 60셀 실측으로
   훈련 로그가 평균 **+0.39%** 높습니다(HF Trainer 가 평가에도 bf16
   autocast 를 걸기 때문). 100h LS 효과(−1%)와 같은 크기라 섞으면 비교가
   무너집니다.
2. **greedy** (`--decoder pipeline`). beam_search 는 test-clean −0.67% /
   test-other −0.72% 로 계통적으로 좋아서 같은 표에 못 들어갑니다
   (같은 체크포인트 144쌍 실측, test-other 는 71/71 에서 beam 이 이김).
3. **test-clean / test-other 전체**. dev 는 안 받습니다.
4. **방법은 체크포인트 디렉터리 접미사로 판정**. 태그를 믿지 않습니다.

   | 접미사 / 조건 | 방법 |
   |---|---|
   | `alpha_0p0` | baseline |
   | `_classspace` (fas 없음), beta=0 | LS (전통, class space) |
   | `_classspace_fas_eps1eneg10`, **beta=1.0** | LS (textbook 과 bit-exact) |
   | `_classspace_fas_eps1eneg10`, beta=0.0 | FAS |
   | `_aws_eps1eneg10_hmatch` | AWS |

   `watch_decode_cpu.sh:72` 는 beta 만 보고 `LS/FAS` 를 붙이므로 **AWS 런이
   전부 `FAS_` 태그로 기록돼 있습니다.** decode.jsonl 의 태그를 그대로
   읽으면 안 됩니다.

5. 장치는 무관합니다. `wav2vec2_inference.py` 는 autocast/bf16 이 없어
   항상 fp32 이고, 같은 체크포인트에서 CPU 와 CUDA 가 소수점 17자리까지
   같았습니다. GPU 가 9배 빠릅니다.

⚠ 표시는 5시드가 안 찬 칸입니다.


## CTC — 1h

| 방법 | α | n | test-clean | test-other | 기하평균 | Δ vs baseline | t |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 5 | 0.21503 | 0.32715 | **0.26522** | — | — |
| LS | 0.05 | 5 | 0.20149 | 0.30840 | **0.24928** | -6.01% | -5.16 |
| LS | 0.1 | 5 | 0.19909 | 0.30526 | **0.24652** | -7.05% | -6.49 |
| LS | 0.15 | 5 | 0.19844 | 0.30394 | **0.24559** | -7.40% | -6.58 |
| LS | 0.2 | 4 ⚠ | 0.20051 | 0.30829 | **0.24863** | -6.26% | -6.29 |

## CTC — 100h

| 방법 | α | n | test-clean | test-other | 기하평균 | Δ vs baseline | t |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 5 | 0.05969 | 0.15535 | **0.09630** | — | — |
| LS | 0.05 | 5 | 0.05918 | 0.15365 | **0.09536** | -0.97% | -1.77 |
| LS | 0.1 | 5 | 0.05904 | 0.15332 | **0.09514** | -1.20% | -2.12 |
| LS | 0.15 | 5 | 0.05906 | 0.15329 | **0.09515** | -1.19% | -2.25 |
| LS | 0.2 | 5 | 0.05918 | 0.15369 | **0.09536** | -0.97% | -1.76 |
| LS | 0.25 | 5 | 0.05954 | 0.15541 | **0.09620** | -0.10% | -0.16 |
| LS | 0.3 | 5 | 0.06086 | 0.15782 | **0.09801** | +1.78% | 1.54 |
| FAS | 0.05 | 4 ⚠ | 0.05828 | 0.15133 | **0.09391** | -2.48% | -4.46 |
| FAS | 0.1 | 5 | 0.05873 | 0.15175 | **0.09440** | -1.97% | -3.42 |
| FAS | 0.15 | 2 ⚠ | 0.05877 | 0.15054 | **0.09406** | -2.32% | -2.40 |
| FAS | 0.2 | 5 | 0.05846 | 0.15055 | **0.09381** | -2.58% | -4.79 |
| FAS | 0.25 | 5 | 0.05895 | 0.15203 | **0.09467** | -1.69% | -2.47 |
| FAS | 0.3 | 5 | 0.05976 | 0.15398 | **0.09593** | -0.38% | -0.61 |
| AWS | 0.005 | 3 ⚠ | 0.05979 | 0.15589 | **0.09654** | +0.26% | 0.35 |
| AWS | 0.01 | 2 ⚠ | 0.05915 | 0.15833 | **0.09678** | +0.50% | 1.00 |
| AWS | 0.02 | 2 ⚠ | 0.06050 | 0.15519 | **0.09690** | +0.63% | 0.76 |
| AWS | 0.04 | 1 ⚠ | 0.06014 | 0.15291 | **0.09590** | -0.41% | — |

## RNN-T — 1h

_아직 없음._


## RNN-T — 100h

| 방법 | α | n | test-clean | test-other | 기하평균 | Δ vs baseline | t |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 1 ⚠ | 0.06555 | 0.17373 | **0.10671** | — | — |
| AWS | 0.005 | 4 ⚠ | 0.06087 | 0.16346 | **0.09975** | -6.53% | — |
| AWS | 0.01 | 2 ⚠ | 0.06128 | 0.16568 | **0.10077** | -5.57% | — |

## 한쪽 split 만 있는 칸 (미완)

- `CTC 1h LS a=0.2 s4` — test-clean 만 있음
- `CTC 1h LS a=0.25 s0` — test-clean 만 있음

## 제외된 줄

- beam_search: 146 줄
- 태그 충돌 — 어느 체크포인트인지 특정 불가 (libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed0_aws_eps1eneg10_hmatch / libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed0_classspace_fas_eps1eneg10): 2 줄
- 태그 충돌 — 어느 체크포인트인지 특정 불가 (libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed0_classspace_fas_eps1eneg10 / libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed0_classspace_fas_eps1eneg10_label_only): 2 줄
- 태그 충돌 — 어느 체크포인트인지 특정 불가 (libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed1_classspace_fas_eps1eneg10 / libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed1_classspace_fas_eps1eneg10_label_only): 2 줄
- 태그 충돌 — 어느 체크포인트인지 특정 불가 (libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed2_classspace_fas_eps1eneg10 / libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed2_classspace_fas_eps1eneg10_label_only): 2 줄
