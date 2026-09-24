# 체크포인트가 어디 있나

**이 파일은 자동 생성됩니다. 손으로 고치지 마십시오.**

    python /mnt/synology_nas_00/chanwcom/results/make_checkpoints_md.py

마지막 생성: 2026-09-24 22:45 KST   ·   결과 수치는 [RESULTS.md](RESULTS.md)

## 방법 이름 대응표 — 이것부터 보십시오

디렉터리 이름과 논문 용어가 다릅니다. **특히 전통 LS 가 `fas` 라는 이름의
디렉터리에 들어 있습니다.**

| 논문 용어 | 실제 조건 | CTC 디렉터리 이름 | RNN-T `rnnt.pt` 의 `args` |
|---|---|---|---|
| **baseline** | `alpha = 0` | `..._alpha_0p0_...` | `alpha=0.0` |
| **LS** (전통 label smoothing) | **FAS + `beta = 1.0`** | `..._classspace_fas_eps1eneg10` (beta_1p0) | `alpha_mode=floored_active_support`, `beta=1.0` |
| | 또는 class-space fixed | `..._classspace` (beta_0p0) | — |
| **FAS** | FAS + `beta = 0.0` | `..._classspace_fas_eps1eneg10` (beta_0p0) | `alpha_mode=floored_active_support`, `beta=0.0` |
| **AWS** | `alpha_mode=aws`, `eps=1e-10` | `..._aws_eps1eneg10_hmatch` | `alpha_mode=aws`, `fas_eps=1e-10` |

`beta = 1.0` 인 FAS 는 textbook LS 와 **bit-exact** 입니다
(`apply_floored_active_support_smoothing` docstring: "beta = 1 -> textbook
uniform LS, (1 - alpha) y + alpha/K").

AWS 이름 끝의 `_hmatch` 는 **의미가 없습니다** -- `_default_run_name()` 의
catch-all 분기가 남긴 잔재이고 엔트로피 매칭과 무관합니다.

## 증거가 어디 있나 -- 두 태스크가 다릅니다

| | 방법·alpha·eps 를 무엇으로 아는가 |
|---|---|
| **CTC** | **디렉터리 이름뿐.** `training_args.bin` 에는 `alpha`/`beta`/`alpha_mode`/`fas_eps` 가 없습니다(HF `TrainingArguments` 라 `seed`·`max_steps`·`output_dir` 만). 2026-09-24 이후 시작한 런은 `run_args.json` 에 전체 인자가 남습니다. |
| **RNN-T** | **체크포인트 자체.** `rnnt.pt` 안에 `vars(args)` 가 통째로 들어 있어 이름과 무관하게 읽힙니다. 352개 전수 복원 성공, 실패 0. |

완주 판정: CTC 는 `checkpoint-*/trainer_state.json` 의 `global_step >= max_steps`
(또는 `save_pretrained` 최종본), RNN-T 는 `rnnt.pt` 존재.


## CTC — 1h

| 방법 | step | α | β | seeds | 완주 | 위치 |
|---|---:|---:|---:|---|---:|---|
| baseline | 120 | 0.0 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_120steps_alpha_0p0_beta_0p0_unigram_32_dynbatch1600000_seed*` |
| baseline | 3000 | 0.0 | 0.0 | [0, 1, 2, 3, 4] | 11/11 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p0_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 3000 | 0.05 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 3000 | 0.1 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 3000 | 0.15 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 3000 | 0.2 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 3000 | 0.25 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 3000 | 0.3 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| FAS | 2000 | 0.15 | 0.0 | [0, 1] | 2/2 | `u24/libri_light_1hr_shc_2000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 2000 | 0.2 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 2000 | 0.25 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 2000 | 0.3 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 2000 | 0.35 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p35_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 2000 | 0.4 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 3000 | 0.05 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 3000 | 0.1 | 0.0 | [0, 1, 2, 3, 4] | 5/6 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 3000 | 0.15 | 0.0 | [0, 1, 2, 3, 4] | 5/6 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 3000 | 0.2 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 3000 | 0.25 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 3000 | 0.3 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_1hr_shc_3000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| AWS | 3000 | 0.005 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `u24/libri_light_1hr_shc_3000steps_alpha_0p005_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 3000 | 0.01 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `u24/libri_light_1hr_shc_3000steps_alpha_0p01_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 3000 | 0.02 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `u24/libri_light_1hr_shc_3000steps_alpha_0p02_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 3000 | 0.03 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `u24/libri_light_1hr_shc_3000steps_alpha_0p03_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 3000 | 0.04 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `u24/libri_light_1hr_shc_3000steps_alpha_0p04_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 3000 | 0.05 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `u24/libri_light_1hr_shc_3000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AS | 2000 | 0.05 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.1 | 0.0 | [0, 1] | 2/2 | `u24/libri_light_1hr_shc_2000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.15 | 0.0 | [0] | 2/2 | `slpl_4090_00/libri_light_1hr_shc_2000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.2 | 0.0 | [0, 1] | 3/3 | `slpl_4090_00/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.25 | 0.0 | [0, 1, 2] | 4/4 | `slpl_4090_00/libri_light_1hr_shc_2000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.3 | 0.0 | [0, 1, 2] | 3/3 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.35 | 0.0 | [0, 1, 2] | 5/5 | `slpl_4090_00/libri_light_1hr_shc_2000steps_alpha_0p35_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.4 | 0.0 | [0, 1, 2] | 3/3 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.45 | 0.0 | [0, 1, 2] | 3/3 | `u24/libri_light_1hr_shc_2000steps_alpha_0p45_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.5 | 0.0 | [0, 1, 2] | 3/3 | `u24/libri_light_1hr_shc_2000steps_alpha_0p5_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| AS | 2000 | 0.55 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p55_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_actsup` |
| FAS(beta=0.025) | 2000 | 0.15 | 0.025 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p15_beta_0p025_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.025) | 2000 | 0.2 | 0.025 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p025_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.025) | 2000 | 0.25 | 0.025 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p25_beta_0p025_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.025) | 2000 | 0.3 | 0.025 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p025_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.025) | 2000 | 0.35 | 0.025 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p35_beta_0p025_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.025) | 2000 | 0.4 | 0.025 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p025_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.05) | 2000 | 0.15 | 0.05 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p15_beta_0p05_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.05) | 2000 | 0.2 | 0.05 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p05_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.05) | 2000 | 0.25 | 0.05 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p25_beta_0p05_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.05) | 2000 | 0.3 | 0.05 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p05_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.05) | 2000 | 0.35 | 0.05 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p35_beta_0p05_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.05) | 2000 | 0.4 | 0.05 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p05_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.075) | 2000 | 0.15 | 0.075 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p15_beta_0p075_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.075) | 2000 | 0.2 | 0.075 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p075_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.075) | 2000 | 0.25 | 0.075 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p25_beta_0p075_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.075) | 2000 | 0.3 | 0.075 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p075_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.075) | 2000 | 0.4 | 0.075 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p075_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.1) | 2000 | 0.15 | 0.1 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p15_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.1) | 2000 | 0.2 | 0.1 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.1) | 2000 | 0.25 | 0.1 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p25_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.1) | 2000 | 0.3 | 0.1 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.1) | 2000 | 0.35 | 0.1 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p35_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.1) | 2000 | 0.4 | 0.1 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.2) | 2000 | 0.3 | 0.2 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p2_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS(beta=0.2) | 2000 | 0.4 | 0.2 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p2_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS-thr | 2000 | 0.2 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.2 | 0.1 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.2 | 0.2 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p2_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.2 | 0.3 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p2_beta_0p3_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.3 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.3 | 0.1 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.3 | 0.2 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p2_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.3 | 0.3 | [0, 1] | 2/2 | `u24/libri_light_1hr_shc_2000steps_alpha_0p3_beta_0p3_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.4 | 0.0 | [0, 1, 2] | 3/3 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.4 | 0.1 | [0, 1, 2] | 3/3 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.4 | 0.2 | [0, 1] | 2/2 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p2_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| FAS-thr | 2000 | 0.4 | 0.3 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p4_beta_0p3_unigram_32_dynbatch6400000_seed*_classspace_fas_thr0p1` |
| SETS(beta=0.025) | 2000 | 0.1 | 0.025 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p1_beta_0p025_unigram_32_dynbatch6400000_seed*_classspace` |
| SETS(beta=0.05) | 2000 | 0.1 | 0.05 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p1_beta_0p05_unigram_32_dynbatch6400000_seed*_classspace` |
| SETS(beta=0.075) | 2000 | 0.1 | 0.075 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p1_beta_0p075_unigram_32_dynbatch6400000_seed*_classspace` |
| SETS(beta=0.1) | 2000 | 0.1 | 0.1 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p1_beta_0p1_unigram_32_dynbatch6400000_seed*_classspace` |
| SETS(beta=0.25) | 2000 | 0.05 | 0.25 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p05_beta_0p25_unigram_32_dynbatch6400000_seed*_classspace` |
| SETS(beta=0.5) | 2000 | 0.05 | 0.5 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p05_beta_0p5_unigram_32_dynbatch6400000_seed*_classspace` |
| SETS(beta=0.75) | 2000 | 0.05 | 0.75 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p05_beta_0p75_unigram_32_dynbatch6400000_seed*_classspace` |
| SETS(beta=1) | 2000 | 0.05 | 1.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_2000steps_alpha_0p05_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace` |
| blank-gate | 3000 | 0.1 | 0.0 | [0] | 2/2 | `u24/libri_light_1hr_shc_3000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10_blank_only` |
| blank-gate | 3000 | 0.15 | 0.0 | [0, 1, 2] | 4/4 | `u24/libri_light_1hr_shc_3000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10_blank_only` |
| blank-gate | 3000 | 0.2 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_3000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10_label_only` |
| 판별불가 | 3000 | 0.01 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_3000steps_alpha_0p01_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_hmatch` |
| 판별불가 | 3000 | 0.05 | 0.0 | [0] | 2/2 | `u24/libri_light_1hr_shc_3000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_hmatch` |
| 판별불가 | 3000 | 0.1 | 0.0 | [0] | 0/1 | `u24/libri_light_1hr_shc_3000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_hmatch` |
| 판별불가 | 3000 | 0.15 | 0.0 | [0] | 1/1 | `u24/libri_light_1hr_shc_3000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_hmatch` |

호스트: inha_4090, slpl_4090_00, u24

## CTC — 10h

| 방법 | step | α | β | seeds | 완주 | 위치 |
|---|---:|---:|---:|---|---:|---|
| baseline | 6000 | 0.0 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p0_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 6000 | 0.05 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 6000 | 0.1 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 6000 | 0.15 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 6000 | 0.2 | 0.0 | [0, 1, 2, 3, 4] | 4/5 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 6000 | 0.25 | 0.0 | [0, 1, 2, 3, 4] | 4/5 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| LS | 6000 | 0.3 | 0.0 | [0, 1, 2, 3] | 4/4 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace` |
| FAS | 4000 | 0.1 | 0.0 | [0, 1, 2, 3, 4] | 7/7 | `slpl_4090_00/libri_light_10hr_shc_4000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 4000 | 0.15 | 0.0 | [0, 1, 2, 3, 4] | 6/6 | `slpl_4090_00/libri_light_10hr_shc_4000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 4000 | 0.2 | 0.0 | [0, 1, 2, 3, 4] | 6/6 | `slpl_4090_00/libri_light_10hr_shc_4000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 4000 | 0.25 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_light_10hr_shc_4000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 4000 | 0.3 | 0.0 | [0, 1, 2, 3, 4] | 7/7 | `slpl_4090_00/libri_light_10hr_shc_4000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 4000 | 0.35 | 0.0 | [0, 1, 2, 3, 4] | 6/6 | `slpl_4090_00/libri_light_10hr_shc_4000steps_alpha_0p35_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 4000 | 0.4 | 0.0 | [0, 1, 2, 3, 4] | 6/6 | `slpl_4090_00/libri_light_10hr_shc_4000steps_alpha_0p4_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 6000 | 0.05 | 0.0 | [0, 1, 2, 3] | 4/4 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 6000 | 0.1 | 0.0 | [0, 1, 2, 3] | 7/7 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 6000 | 0.15 | 0.0 | [0, 1, 2, 3] | 4/4 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 6000 | 0.2 | 0.0 | [0, 1, 2, 3] | 4/4 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 6000 | 0.25 | 0.0 | [0, 1, 2, 3] | 4/4 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 6000 | 0.3 | 0.0 | [0, 1, 2, 3] | 4/4 | `inha_4090/libri_light_10hr_shc_6000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| AWS | 6000 | 0.01 | 0.0 | [0] | 1/1 | `u24/libri_light_10hr_shc_6000steps_alpha_0p01_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| ASAP | 4000 | 0.1 | 0.0 | [0] | 1/1 | `u24/libri_light_10hr_shc_4000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_asap_eps0p03125` |
| ASAP | 4000 | 0.15 | 0.0 | [0] | 1/1 | `u24/libri_light_10hr_shc_4000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_asap_eps0p03125` |
| ASAP | 4000 | 0.2 | 0.0 | [0] | 1/1 | `u24/libri_light_10hr_shc_4000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_asap_eps0p03125` |
| ASAP | 4000 | 0.3 | 0.0 | [0] | 2/2 | `u24/libri_light_10hr_shc_4000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_asap_eps0p03125` |
| ASAP | 4000 | 0.35 | 0.0 | [0] | 1/1 | `u24/libri_light_10hr_shc_4000steps_alpha_0p35_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_asap_eps0p03125` |
| ASAP | 4000 | 0.4 | 0.0 | [0] | 1/1 | `u24/libri_light_10hr_shc_4000steps_alpha_0p4_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_asap_eps0p03125` |
| blank-gate | 6000 | 0.15 | 0.0 | [0] | 0/1 | `u24/libri_light_10hr_shc_6000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10_label_only` |

호스트: inha_4090, slpl_4090_00, u24

## CTC — 100h

| 방법 | step | α | β | seeds | 완주 | 위치 |
|---|---:|---:|---:|---|---:|---|
| baseline | 8000 | 0.0 | 0.0 | [0, 1, 2, 3, 4] | 5/6 | `slpl_4090_00/libri_speech_clean_100hr_shc_8000steps_alpha_0p0_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| baseline | 10000 | 0.0 | 0.0 | [0, 1] | 0/2 | `slpl_4090_00/libri_speech_clean_100hr_shc_10000steps_alpha_0p0_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| baseline | 12000 | 0.0 | 0.0 | [0, 1, 2, 3, 4] | 4/5 | `slpl_4090_00/libri_speech_clean_100hr_shc_12000steps_alpha_0p0_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| baseline | 15000 | 0.0 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `u24/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p0_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 8000 | 0.05 | 1.0 | [0, 1] | 2/2 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p05_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 8000 | 0.1 | 1.0 | [0, 1] | 1/2 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p1_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 8000 | 0.15 | 1.0 | [0] | 1/1 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p15_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 8000 | 0.2 | 1.0 | [2] | 1/1 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p2_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 8000 | 0.25 | 1.0 | [2] | 1/1 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p25_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 10000 | 0.05 | 1.0 | [0] | 0/1 | `u24/libri_speech_clean_100hr_shc_10000steps_alpha_0p05_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 10000 | 0.2 | 1.0 | [0] | 0/1 | `u24/libri_speech_clean_100hr_shc_10000steps_alpha_0p2_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 12000 | 0.05 | 1.0 | [0] | 1/1 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p05_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 12000 | 0.1 | 1.0 | [0] | 1/1 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p1_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 12000 | 0.15 | 1.0 | [0] | 0/1 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p15_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 12000 | 0.2 | 1.0 | [0, 3] | 1/2 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p2_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 12000 | 0.25 | 1.0 | [0] | 1/1 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p25_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 12000 | 0.3 | 1.0 | [0] | 0/1 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p3_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 15000 | 0.05 | 1.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p05_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 15000 | 0.1 | 1.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p1_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 15000 | 0.15 | 1.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p15_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 15000 | 0.2 | 1.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p2_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 15000 | 0.25 | 1.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p25_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| LS | 15000 | 0.3 | 1.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p3_beta_1p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 8000 | 0.05 | 0.0 | [0, 1] | 1/2 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 8000 | 0.15 | 0.0 | [0, 1] | 2/2 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 8000 | 0.2 | 0.0 | [0, 1] | 2/2 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 8000 | 0.25 | 0.0 | [0, 1] | 1/2 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 8000 | 0.3 | 0.0 | [0] | 0/1 | `u24/libri_speech_clean_100hr_shc_8000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 10000 | 0.05 | 0.0 | [0] | 0/1 | `u24/libri_speech_clean_100hr_shc_10000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 10000 | 0.2 | 0.0 | [0] | 0/1 | `u24/libri_speech_clean_100hr_shc_10000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 12000 | 0.05 | 0.0 | [0, 3] | 2/2 | `slpl_4090_00/libri_speech_clean_100hr_shc_12000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 12000 | 0.1 | 0.0 | [0] | 1/1 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 12000 | 0.15 | 0.0 | [0] | 0/1 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 12000 | 0.2 | 0.0 | [0, 3] | 1/2 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 12000 | 0.25 | 0.0 | [0] | 1/1 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 12000 | 0.3 | 0.0 | [0] | 0/1 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 15000 | 0.05 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 15000 | 0.1 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p1_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 15000 | 0.15 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 15000 | 0.2 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p2_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 15000 | 0.25 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p25_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| FAS | 15000 | 0.3 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `slpl_4090_00/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p3_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10` |
| AWS | 15000 | 0.005 | 0.0 | [0, 1, 2, 3, 4] | 3/5 | `u24/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p005_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 15000 | 0.01 | 0.0 | [0, 2, 3, 4] | 2/4 | `u24/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p01_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 15000 | 0.02 | 0.0 | [0, 2, 3] | 2/3 | `u24/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p02_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 15000 | 0.03 | 0.0 | [0, 1, 2, 3] | 0/4 | `u24/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p03_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 15000 | 0.04 | 0.0 | [0, 1, 2, 3] | 1/4 | `u24/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p04_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| AWS | 15000 | 0.05 | 0.0 | [0, 1, 2, 3] | 1/4 | `u24/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p05_beta_0p0_unigram_32_dynbatch6400000_seed*_aws_eps1eneg10_hmatch` |
| blank-gate | 12000 | 0.15 | 0.0 | [0, 1, 2] | 0/3 | `u24/libri_speech_clean_100hr_shc_12000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10_label_only` |
| blank-gate | 15000 | 0.15 | 0.0 | [0, 1, 2] | 3/3 | `u24/libri_speech_clean_100hr_wsd_shc_15000steps_alpha_0p15_beta_0p0_unigram_32_dynbatch6400000_seed*_classspace_fas_eps1eneg10_label_only` |

호스트: slpl_4090_00, u24

## RNN-T — 1h

| 방법 | step | α | β | seeds | 완주 | 위치 |
|---|---:|---:|---:|---|---:|---|
| baseline | 3000 | 0.0 | 0.0 | [0, 1, 2, 3, 4] | 62/62 | `slpl_4090_00/rnnt1hr/1hr_base_a0.0_s*` |
| LS | 3000 | 0.05 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt1hr/1hr_LS_a0.05_s*` |
| LS | 3000 | 0.1 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt1hr/1hr_LS_a0.1_s*` |
| LS | 3000 | 0.15 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt1hr/1hr_LS_a0.15_s*` |
| LS | 3000 | 0.2 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt1hr/1hr_LS_a0.2_s*` |
| LS | 3000 | 0.25 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt1hr/1hr_LS_a0.25_s*` |
| LS | 3000 | 0.3 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt1hr/1hr_LS_a0.3_s*` |
| FAS | 3000 | 0.005 | 0.0 | [0] | 1/1 | `scratchpad/fas1hr/ckpt/rnnt1hr_fas_a0.005_s*` |
| FAS | 3000 | 0.01 | 0.0 | [0] | 1/1 | `scratchpad/fas1hr/ckpt/rnnt1hr_fas_a0.01_s*` |
| FAS | 3000 | 0.02 | 0.0 | [0] | 1/1 | `scratchpad/fas1hr/ckpt/rnnt1hr_fas_a0.02_s*` |
| FAS | 3000 | 0.03 | 0.0 | [0] | 3/3 | `scratchpad/fas1hr/ckpt/rnnt1hr_fas_a0.03_gatehigh_s*` |
| FAS | 3000 | 0.05 | 0.0 | [0] | 2/2 | `scratchpad/fas1hr/ckpt/rnnt1hr_fas_a0.05_gateblank_only_s*` |
| AWS | 3000 | 0.005 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `scratchpad/fas1hr/ckpt/rnnt1hr_aws_a0.005_eps1e-10_s*` |
| AWS | 3000 | 0.01 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `scratchpad/fas1hr/ckpt/rnnt1hr_aws_a0.01_eps1e-10_s*` |
| AWS | 3000 | 0.02 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `scratchpad/fas1hr/ckpt/rnnt1hr_aws_a0.02_eps1e-10_s*` |
| AWS | 3000 | 0.03 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `scratchpad/fas1hr/ckpt/rnnt1hr_aws_a0.03_eps1e-10_s*` |
| AWS | 3000 | 0.05 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `scratchpad/fas1hr/ckpt/rnnt1hr_aws_a0.05_eps1e-10_s*` |
| AWS | 3000 | 0.1 | 0.0 | [0] | 1/1 | `scratchpad/fas1hr/ckpt/rnnt1hr_aws_a0.10_s*` |
| AWS(eps=0.01) | 3000 | 0.01 | 0.0 | [0] | 1/1 | `scratchpad/fas1hr/ckpt/rnnt1hr_aws_a0.01_eps1e-2_s*` |
| AWS(eps=0.01) | 3000 | 0.05 | 0.0 | [0] | 1/1 | `scratchpad/fas1hr/ckpt/rnnt1hr_aws_a0.05_eps1e-2_s*` |
| alignment_biased | 3000 | 0.01 | 0.0 | [0] | 1/1 | `scratchpad/abs/ckpt/rnnt1hr_abs_a0.01_s*` |
| aws_alpha_beta | 3000 | 0.01 | 0.0 | [0] | 1/1 | `scratchpad/ab1hr/ckpt/rnnt1hr_ab_a0.01_s*` |
| aws_alpha_beta | 3000 | 0.02 | 0.0 | [0] | 1/1 | `scratchpad/ab1hr/ckpt/rnnt1hr_ab_a0.02_s*` |
| diagonal_active_support | 60 | 0.05 | 0.0 | [0] | 1/1 | `scratchpad/rnnt/smoke_das` |
| diagonal_projected | 3000 | 0.001 | 0.0 | [0] | 1/1 | `scratchpad/dps1hr/ckpt/rnnt1hr_dps_a0.001_s*` |
| diagonal_projected | 3000 | 0.002 | 0.0 | [0] | 1/1 | `scratchpad/dps1hr/ckpt/rnnt1hr_dps_a0.002_s*` |
| diagonal_projected | 3000 | 0.005 | 0.0 | [0] | 1/1 | `scratchpad/dps1hr/ckpt/rnnt1hr_dps_a0.005_s*` |
| diagonal_projected | 3000 | 0.05 | 0.0 | [0] | 2/2 | `scratchpad/dps1hr_INCREMENT/ckpt/rnnt1hr_dps_a0.05_s*` |
| diagonal_projected | 3000 | 0.1 | 0.0 | [0] | 2/2 | `scratchpad/dps1hr_INCREMENT/ckpt/rnnt1hr_dps_a0.10_s*` |
| mos | 3000 | 0.01 | 0.0 | [0] | 1/1 | `scratchpad/fas1hr/ckpt/rnnt1hr_mos_a0.01_eps1e-2_s*` |
| mos | 3000 | 0.05 | 0.0 | [0] | 3/3 | `scratchpad/fas1hr/ckpt/rnnt1hr_mos_a0.05_eps1e-10_s*` |

호스트: NAS/slpl_4090_00, NAS/u22, scratchpad

## RNN-T — 10h

| 방법 | step | α | β | seeds | 완주 | 위치 |
|---|---:|---:|---:|---|---:|---|
| baseline | 6000 | 0.0 | 0.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt10hr/10hr_base_a0.0_s*` |
| LS | 6000 | 0.05 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt10hr/10hr_LS_a0.05_s*` |
| LS | 6000 | 0.1 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt10hr/10hr_LS_a0.1_s*` |
| LS | 6000 | 0.15 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt10hr/10hr_LS_a0.15_s*` |
| LS | 6000 | 0.2 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt10hr/10hr_LS_a0.2_s*` |
| LS | 6000 | 0.25 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt10hr/10hr_LS_a0.25_s*` |
| LS | 6000 | 0.3 | 1.0 | [0, 1, 2, 3, 4] | 10/10 | `slpl_4090_00/rnnt10hr/10hr_LS_a0.3_s*` |
| FAS | 6000 | 0.005 | 0.0 | [0] | 1/1 | `scratchpad/bigeps/ckpt/rnnt10hr_fas_a0.005_eps0.1_s*` |
| FAS | 6000 | 0.01 | 0.0 | [0] | 2/2 | `scratchpad/bigeps/ckpt/rnnt10hr_fas_a0.01_eps0.1_s*` |
| FAS | 6000 | 0.02 | 0.0 | [0] | 1/1 | `scratchpad/bigeps/ckpt/rnnt10hr_fas_a0.02_eps0.1_s*` |
| FAS | 6000 | 0.05 | 0.0 | [0] | 5/5 | `scratchpad/bigeps/ckpt/rnnt10hr_fas_a0.05_eps0.1_s*` |
| FAS | 6000 | 0.1 | 0.0 | [0] | 5/5 | `scratchpad/bigeps/ckpt/rnnt10hr_fas_a0.1_eps0.1_s*` |
| AWS | 6000 | 0.005 | 0.0 | [0] | 1/1 | `scratchpad/aws10hr/ckpt/rnnt10hr_aws_a0.005_eps1e-10_b0.0_s*` |
| AWS | 6000 | 0.01 | 0.0 | [0] | 3/3 | `slpl_4090_00/aws10hr/rnnt10hr_aws_a0.01_eps1e-10_s*` |
| AWS | 6000 | 0.01 | 1.0 | [0] | 1/1 | `scratchpad/aws10hr/ckpt/rnnt10hr_aws_a0.01_eps1e-10_b1.0_s*` |
| AWS | 6000 | 0.02 | 0.0 | [0, 1, 2, 3, 4] | 5/5 | `scratchpad/aws10hr/ckpt/rnnt10hr_aws_a0.02_eps1e-10_s*` |
| AWS(eps=0.01) | 6000 | 0.01 | 0.0 | [0] | 1/1 | `scratchpad/aws10hr/ckpt/rnnt10hr_aws_a0.01_eps1e-2_b0.0_s*` |
| AWS(eps=0.01) | 6000 | 0.02 | 0.0 | [0] | 1/1 | `scratchpad/aws10hr/ckpt/rnnt10hr_aws_a0.02_eps1e-2_b0.0_s*` |
| AWS(eps=0.1) | 6000 | 0.02 | 0.0 | [0] | 1/1 | `scratchpad/bigeps/ckpt/rnnt10hr_aws_a0.02_eps0.1_s*` |
| AWS(eps=0.1) | 6000 | 0.1 | 0.0 | [0] | 1/1 | `scratchpad/bigeps/ckpt/rnnt10hr_aws_a0.1_eps0.1_s*` |
| diagonal_active_support | 6000 | 0.05 | 0.0 | [0] | 1/1 | `scratchpad/rnnt/ckpt10hr/10hr_DAS_a0.05_s*` |
| diagonal_active_support | 6000 | 0.1 | 0.0 | [0] | 1/1 | `scratchpad/rnnt/ckpt10hr/10hr_DAS_a0.1_s*` |
| diagonal_active_support | 6000 | 0.15 | 0.0 | [0] | 1/1 | `scratchpad/rnnt/ckpt10hr/10hr_DAS_a0.15_s*` |

호스트: NAS/slpl_4090_00, NAS/u22, scratchpad

## RNN-T — 100h

| 방법 | step | α | β | seeds | 완주 | 위치 |
|---|---:|---:|---:|---|---:|---|
| baseline | 15000 | 0.0 | 0.0 | [0, 1, 2, 3, 4] | 7/7 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.0_s*` |
| LS | 15000 | 0.05 | 1.0 | [0, 1, 2, 3, 4] | 5/5 | `scratchpad/rnnt/ckpt/100hr_LS_a0.05_s*` |
| LS | 15000 | 0.1 | 1.0 | [0, 1, 2, 3, 4] | 5/5 | `scratchpad/rnnt/ckpt/100hr_LS_a0.1_s*` |
| LS | 15000 | 0.15 | 1.0 | [0, 1, 2, 3, 4] | 5/5 | `scratchpad/rnnt/ckpt/100hr_LS_a0.15_s*` |
| FAS | 15000 | 0.05 | 0.0 | [0] | 1/1 | `scratchpad/rnnt/ckpt/100hr_FAS_a0.05_s*` |
| FAS | 15000 | 0.1 | 0.0 | [0] | 1/1 | `scratchpad/rnnt/ckpt/100hr_FAS_a0.1_s*` |
| FAS | 15000 | 0.15 | 0.0 | [0] | 1/1 | `scratchpad/rnnt/ckpt/100hr_FAS_a0.15_s*` |
| AWS | 15000 | 0.005 | 0.0 | [0, 1, 2, 3] | 9/9 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.005_s*` |
| AWS | 15000 | 0.01 | 0.0 | [0, 1, 2, 3] | 9/9 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.01_s*` |
| AWS | 15000 | 0.02 | 0.0 | [0, 1, 2, 3] | 7/7 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.02_s*` |
| AWS | 15000 | 0.03 | 0.0 | [0, 1, 2, 3] | 12/12 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.030_s*` |
| AWS | 15000 | 0.04 | 0.0 | [0, 1, 2] | 8/8 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.04_s*` |
| AWS | 15000 | 0.05 | 0.0 | [0, 1, 2] | 6/6 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.05_s*` |
| AWS | 15000 | 0.06 | 0.0 | [0] | 2/2 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.06_s*` |
| AWS | 15000 | 0.08 | 0.0 | [0] | 2/2 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.08_s*` |
| AWS | 15000 | 0.1 | 0.0 | [0] | 2/2 | `slpl_4090_00/rnnt100hr_aws/rnnt100hr_aws_a0.10_s*` |
| frame_label_support | 15000 | 0.05 | 0.0 | [0] | 1/1 | `scratchpad/rnnt/ckpt/100hr_FLS_a0.05_s*` |
| frame_label_support | 15000 | 0.1 | 0.0 | [0] | 1/1 | `scratchpad/rnnt/ckpt/100hr_FLS_a0.1_s*` |

호스트: NAS/slpl_4090_00, NAS/u22, scratchpad
