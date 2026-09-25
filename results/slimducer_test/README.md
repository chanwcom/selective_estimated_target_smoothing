# SlimDucer test-set results

One JSON object per line in `results.jsonl`, appended by whoever runs the
decode. Every machine writes here; nothing is overwritten, so a re-run just
adds a line and the later `wall_utc` wins.

## How a line is produced

    python slimducer_eval_ckpt.py \
      --checkpoint <ckpt_step*.pt> \
      --llm_name <hf name> [--audio_backbone ... --audio_name ... --audio_stride ...] \
      --splits test-clean,test-other --device cpu --no_bf16 \
      --tag <tag>

`slimducer_eval_ckpt.py` is in the slimducer repo (commit 2ec491b). It calls
the training script's own `evaluate`, so these numbers share a decode path,
a batching rule and a WER definition with the dev numbers the runs reported.

## Why CPU and fp32, and why that is not optional

Measured on 200 test-clean utterances, 100 h Qwen3-0.6B checkpoint:

    GPU bf16   0.06066   1.14 s/utt
    CPU bf16   0.06021   2.12 s/utt
    CPU fp32   0.06021   1.86 s/utt

CPU bf16 and CPU fp32 agree to five decimals -- torch accumulates bf16 in
fp32 on CPU -- so `--no_bf16` costs nothing and saves 12 % of the time.

The GPU-bf16 / CPU-fp32 gap is 0.74 % relative (0.00045 absolute, about two
words out of 3,500). **That is a dtype effect, not a device effect, and the
figure is one measurement on 200 utterances -- do not quote it as a
CPU-vs-GPU constant.** Measured 2026-09-24 on the RNN-T side, where
`greedy_decode` runs in fp32 on both devices, CPU and CUDA agree exactly:

    100hr_FAS_a0.15_s0   CUDA 0.09842 / 0.20823   CPU 0.09842 / 0.20823
    100hr_base_a0.0_s1   CUDA 0.05774 / 0.16302   CPU 0.05774 / 0.16302

(full dev, 2694 + 2857 utterances, identical to the printed five decimals,
which resolve well under one word). So fp32-vs-fp32 across devices is zero;
what moves the number is bf16 on GPU, where the accumulation is real. On CPU
bf16 and fp32 agree bit-for-bit because torch accumulates bf16 in fp32.

**CPU fp32 is still the convention here** -- not because the device matters,
but because fixing device and dtype together removes the one path that does
move the number, at no cost.

## Fields

`tag`, `test-clean_wer`, `test-other_wer`, `geo` (sqrt of the two), `n`
(utterances actually scored per split), `step`, `device`, `wall_min`,
and `machine` / `wall_utc` added on filing.

Splits are the whole thing: test-clean 2611, test-other 2932 after the
30 s cap. A line whose `n` differs was capped and is not comparable.

## 기하평균만 보지 마십시오 -- split 별로 보십시오

인코더 비교에서 기하평균이 반대 방향 효과를 지웁니다. 100 h 실측:

                  test-clean   test-other   기하평균   o/c
    AuT             0.06867      0.13409    0.09596   1.95
    WavLM-Large     0.07610      0.11840    0.09492   1.56
    변화            +10.83%      -11.70%     -1.08%

**-1.08% 는 +10.8% 와 -11.7% 가 우연히 상쇄된 값입니다.** "WavLM 이 100 h
에서 근소하게 이긴다"는 서술은 그 상쇄를 숨깁니다. 10 h 에서도 같은 구조로,
WavLM 의 약점이 clean 에 몰려 있습니다 (clean +37.1%, other +10.2%).

이른바 크로스오버(10 h 는 AuT, 100 h 는 WavLM)의 정체도 인코더 성능의
역전이 아니라, WavLM 의 clean 약점이 데이터와 함께 줄고(+37.1% -> +10.8%)
other 는 약점에서 강점으로 뒤집히면서(+10.2% -> -11.7%) 기하평균이 넘어선
것입니다.

`o/c` 비율이 인코더를 가릅니다 -- AuT 1.46~1.95, whisper 1.46 인데 WavLM 만
1.17~1.56 입니다. WavLM 은 clean/other 격차가 평평합니다.

## 평가 집합

이 표는 `n = test-clean 2611 / test-other 2932` 입니다 (30초 초과 제외).
SlimDucer 인코더가 구조적으로 30초까지만 받기 때문입니다
(`slimducer.py` 의 `max_mel = 3000`). **CTC/RNN-T 표는 필터 없이
2620 / 2939 를 쓰므로 두 표의 절대값을 직접 비교하지 마십시오.**
전체 규약은 [../EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md).
