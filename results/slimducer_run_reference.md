# SlimDucer: measured costs, and what the measurements cost to get wrong

Numbers here are measured, not estimated. Anything still an estimate says so.

## Memory and speed, by (encoder, LLM) and card

| encoder + LLM | card | peak VRAM | trainable | s/step |
|---|---|---|---|---|
| AuT + SmolLM2-360M | 4090 | 2.1 GiB | 2.9M | 0.16-0.18 |
| AuT + Qwen3-0.6B | 5090 | 7.4 GiB | 3.0M | 0.29 |
| AuT + Llama-3.2-1B | 5090 | 5.8 GiB | 5.1M | 0.25 |
| AuT + Qwen3-4B | 5090 | 16.5 GiB | 6.2M | 0.35 |
| whisper-small + Qwen3-0.6B | 5090 | 2.9 GiB | 5.3M | - |
| WavLM-Large + Qwen3-4B | 4090 | 17.0 GiB | 9.5M | 0.37 |
| WavLM-Large + Llama-3.2-1B | 4090 | 6.5 GiB | 8.4M | 0.29 |

Parameter count does not predict either column. Llama-3.2-1B is twice
Qwen3-0.6B and uses less of both, because the cost is dominated by the
alignment pass pushing a (T, L+1) grid through a vocab-wide lm_head -- 85 %
of run time -- and Llama's vocabulary is 128,256 against Qwen's 151,936.
Qwen3-4B is 6.7x the parameters of Qwen3-0.6B for 1.2x the step time and
2.2x the memory, because that vocabulary is identical.

## Two traps in short probes

**Speed measured before step 1000 is about half the real figure.** The first
1000 steps use uniform targets, so the alignment never runs the model.
Qwen3-4B + WavLM reads 0.20 s/step there and 0.36-0.37 after. A 1200-step
probe that includes the uniform stretch will promise a run twice as fast as
it is.

**Memory is trustworthy from the start** -- 17.0 GiB held from step 200
through 1000 -- and it does not depend on the data, only on the model shape
and --max_frames/--max_batch_utts. So probe VRAM on the 10h preset: probing
960h costs 20 minutes of loading for a number the 10h probe gives exactly
(17.0 GiB and 9.5M trainable matched the real run to the decimal).

## RAM, which is the constraint nobody budgets for

960h holds 103 GiB of int16 audio in RAM, per process. Two concurrent 960h
runs need 206 GiB. Check `free -g` before starting the second.

    [data] train 281187 utt (960.7 h, 103 GiB int16)
    [plan] 81831 steps/epoch

## 960 h schedule: 375,000 steps, and why not more

Measured at AuT + Qwen3-0.6B, decay fixed at 20 %, only the budget varied:

| total | epochs | dev-clean/other | geo |
|---|---|---|---|
| 250,000 | 3.06 | 0.0455 / 0.0687 | 0.0559 |
| **375,000** | **4.58** | **0.0446 / 0.0672** | **0.0547** |
| 500,000 | 6.11 | 0.0472 / 0.0687 | 0.0570 |

Not monotone: 500k is worse than 250k. The decay entry points were 0.0678 /
0.0683 / 0.0686 -- 1.2 % apart and in the wrong order -- so the budget
cannot be chosen from them. This is why PRESET_SCHEDULE pins 375,000.

## Do not judge from mid-training numbers

At 60 % through decay the three runs above read 0.0593 / 0.0596 / 0.0596 --
indistinguishable -- and finished 0.0547 to 0.0570. Four separate
comparisons in this project have had their ordering reverse between a
mid-stable reading and the final one (LLM swaps at 10h and 100h, encoder
swaps at 10h and 100h). Rankings before the decay finishes carry almost no
information.

## Machines

| host | GPUs | RAM | CPU | paths |
|---|---|---|---|---|
| u24 (this one) | 4 x RTX 5090 31.4 GiB | 251 GiB | 80 threads | `/mnt/synology_nas_00/chanwcom/models/u24/` |
| slpl_4090_00 | RTX 4090 | - | - | `/mnt/synology_nas_00/chanwcom/models/slpl_4090_00/` |
| oem-WS-C621E-SAGE (Inha) | 2 x RTX 4090 24.5 GiB | 314 GiB | 40 threads | `165.246.42.219:/mnt/data/home/chanwcom/models/` -- NOT on the NAS, scp |

Inha does not mount the NAS. Results from there arrive by message and are
filed here by hand.
