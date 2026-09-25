# 평가 규약 — 숫자를 만들기 전에 읽으십시오

이 문서는 **어떤 숫자가 논문 표에 들어갈 수 있는가**를 정합니다. 2026-09-25
에 하루 동안 네 종류의 오염을 발견하고 정한 규약입니다. 각 항목에 실측
근거가 붙어 있습니다.

## 채택 조건 — 넷이 모두 맞아야 합니다

| 축 | CTC / RNN-T | SlimDucer |
|---|---|---|
| **평가 집합** | test-clean **2620** / test-other **2939** (필터 없음) | test-clean **2611** / test-other **2932** (30초 초과 제외) |
| **배치** | `batch_size = 1` | 해당 없음 |
| **dtype** | fp32 | fp32 |
| **디코더** | greedy | greedy |

생산 경로: CTC 는 `wav2vec2_inference.py --batch_size 1 --decoder pipeline`,
RNN-T 는 `rnnt/eval_ckpt.py --batch_size 1`, SlimDucer 는
`slimducer_eval_ckpt.py --device cpu --no_bf16`.

**훈련 루프 eval 값은 어떤 경우에도 넣지 않습니다.**

---

## 1. 배치 크기가 WER 을 바꿉니다 (CTC / RNN-T)

`facebook/wav2vec2-base` 는 `feat_extract_norm="group"` 이라 conv 특징
추출기의 GroupNorm 이 **시간축 전체**를 정규화합니다. 0 패딩이 그 통계에
들어가면 유효 프레임의 출력까지 바뀌고, `attention_mask` 는 트랜스포머
층에서만 작동해 되돌릴 수 없습니다.

같은 체크포인트 · 같은 192발화 · 배치 설정만 변경:

    CTC    bs=1   0.04922      <- 배치와 무관한 유일한 값
           bs=8   0.05202      +5.7%
           bs=12  0.05318      +8.1%   (훈련 루프 eval 이 이 값)
    RNN-T  bs=1   0.04712
           bs=8   0.04852      +3.0%
           bs=8   0.04992      +5.9%   (마스크 없이 = HF 권고안)
           bs=32  0.06018     +27.7%   (훈련 루프 eval 이 이 값)

**HF 권고안(마스크 미전달)은 답이 아닙니다** -- 저희 모델은 마스크를 넘기는
조건에서 훈련됐습니다 (`wav2vec2_rnnt.py:162`,
`wav2vec2_finetuning_sets.py:510` 이 `return_attention_mask=True` 로
기본값을 뒤집습니다).

**bs=1 이 더 빠릅니다.** `padding="longest"` 가 배치 안 최장 발화에 맞추는데
LibriSpeech 발화가 1~35초라 패딩 계산이 낭비됩니다. 128발화 실측, 4스레드,
번갈아 2라운드: bs=1 109/106초 vs bs=8 217/220초 -- **2.03배**.

CPU 효율은 **워커당 2스레드**가 최고입니다 (코어·초/발화: CTC 2.77 / 3.83 /
4.71, RNN-T 3.04 / -- / 6.83 for 2 / 4 / 8 스레드). 스레드를 줄이고 워커를
늘리는 편이 총 처리량이 2.25배입니다. bs=1 에서는 GPU 이점이 1.6배뿐이라
(0.370 vs CPU 8스레드 0.854 초/발화) CPU 로 충분합니다.

**SlimDucer 는 무관합니다**: AuT 는 conv 추출기가 없고, WavLM-Large 는
`feat_extract_norm="layer"` 라 시간축 통계를 쓰지 않으며, whisper 는 항상
30초 고정 패딩입니다.

## 2. 평가 집합이 갈려 있었습니다

`sample_util.make_dataset(..., max_sample_length=480000)` 는 30초를 넘는
발화를 **통째로 버립니다**(`select`, 자르는 것이 아님). 오디오 길이 필터이고
라벨 길이가 아닙니다 -- `--max_label_len 450` 은 훈련에만 걸립니다.

    CTC   wav2vec2_inference.py:502   인자 없음      2620 / 2939
    CTC   훈련 루프 :1389-1392         인자 전달      2611 / 2932
    RNN-T eval_ckpt.py:130             인자 전달 -> 제거   2620 / 2939
    SlimDucer read_shards(max_sec)     인자 전달      2611 / 2932

**CTC / RNN-T 는 필터를 제거합니다.** 30초 컷은 훈련 메모리 제약이지 평가
정의가 아니고, 긴 발화 16개는 단어 수가 많아 WER 기여가 불균형합니다.

**SlimDucer 는 필터를 유지합니다.** 인코더가 구조적으로 30초까지만 받습니다
(`slimducer.py:230/241/274` 의 `max_mel = 3000`, 버퍼가 3000 프레임 고정).
SlimDucer 표에는 반드시 `n=2611/2932, 30초 초과 제외`를 명시합니다.

dev 는 되돌릴 수 없습니다(전 런의 훈련 로그값). dev 표에는
`n=2694/2857, 30초 초과 제외`를 명시합니다.

## 3. 방법은 이름이 아니라 증거로 판정합니다

`watch_decode_cpu.sh:72` 가 `beta==1.0 ? LS : FAS` 로만 이름을 붙여서
**AWS 런이 전부 `FAS_` 태그로 기록**돼 있습니다. 태그를 읽으면 안 됩니다.

    CTC    체크포인트 디렉터리 이름이 유일한 증거
           training_args.bin 에는 alpha / alpha_mode / fas_eps 가 없습니다
           2026-09-24 이후 시작한 런은 output_dir/run_args.json 에 전체 인자
    RNN-T  rnnt.pt 안의 vars(args) -- 352개 전수 복원 성공

이름 대응은 `CHECKPOINTS.md` 를 보십시오. **전통 LS 가 `fas` 라는 이름의
디렉터리에 들어 있습니다** (FAS + beta=1.0 이 textbook LS 와 bit-exact).

## 4. 사후 검증 규칙

**`test-other` 가 `test-clean` 의 1.3배 미만이면 그 줄은 버립니다.**

"other 는 항상 clean 의 2배 이상"이라고 적어 뒀던 것은 **틀렸습니다**. 실측
o/c 는 모델마다 크게 벌어집니다:

    WavLM-Large    1.75        SmolLM2      1.89
    AuT            2.15~2.26   CTC          2.57
    RNN-T          2.59

1.75 와 2.59 사이입니다. 그래서 임계값을 **1.3** 으로 둡니다 -- 가장 낮은
1.75 에도 여유가 있고, 1.5 로 올리면 정상인 WavLM 런이 오탐으로 걸립니다.
1.3 미만은 측정이 깨졌다는 신호입니다.

이 규칙이 실제로 잡은 것: 평가 워커가 두 split 을 한 로그에 이어 쓰고
`findall(...)[-1]` 로 읽어서, `test-other` 실행이 중간에 죽으면
**`test-clean` 값이 `test-other` 자리에 조용히 들어갔습니다.** 결측이
아니라 그럴듯한 값이라 로그만 봐서는 멀쩡합니다. 89셀을 폐기했습니다.

근본 대책은 **split 마다 독립 로그 + `returncode == 0` 확인**이고, 1.3배
규칙은 이중 방어입니다.

**이 규칙을 dev 에 쓰지 마십시오.** 200발화 캡이 dev-other 를 낙관적으로
만들어 비율이 눌립니다 -- 정상인 AuT 런도 dev@200 기준 o/c 가 1.35 라
오탐이 납니다. test 전용입니다.

그리고 **dev@200 이 test 를 얼마나 낙관하는지는 상수가 아닙니다.** 실측
편향비가 AuT 는 1.23~1.41, WavLM+Llama 는 1.50 입니다. 한 모델에서 구한
환산 비율을 다른 모델에 옮겨 쓰면 안 됩니다 -- dev 값은 test 값을 추정하는
데 쓸 수 없고, 다시 디코드하는 수밖에 없습니다.

## 5. 기하평균이 반대 방향 효과를 지웁니다

인코더·방법 비교를 단일 숫자로 하면 안 됩니다. SlimDucer 100h 실측:

              test-clean   test-other   기하평균   o/c
    AuT         0.06867      0.13409    0.09596   1.95
    WavLM       0.07610      0.11840    0.09492   1.56
    변화        +10.83%      -11.70%     -1.08%

**기하평균 -1.08% 는 +10.8% 와 -11.7% 가 우연히 상쇄된 값입니다.**
"WavLM 이 100h 에서 근소하게 이긴다"는 서술은 이 상쇄를 숨깁니다.
**표에는 split 별 값을 같이 적으십시오.**
