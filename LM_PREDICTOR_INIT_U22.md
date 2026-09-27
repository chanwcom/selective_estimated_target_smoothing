# Pre-training the RNN-T predictor as a character LM (u22)

Everything here was produced on u22 between 2026-09-19 18:00 and 21:00 KST.
Nothing in this document touches the CTC or transducer grids that were
already running; the RNN-T changes live in a copy,
`wav2vec2_rnnt_lminit.py`, and `wav2vec2_rnnt.py` is byte-identical to the
committed version.

## 1. Why

The 1 h transducer runs fail in a specific way, and the failure points at
the predictor rather than the encoder. Measured on the full dev splits,
five seeds each:

| labelled data | dev-clean | dev-other | final train loss |
|---|---|---|---|
| 10 h | 0.09413 ± 0.00075 | 0.18832 ± 0.00179 | 1.570 |
| 1 h | 0.18648 ± 0.00081 | 0.27935 ± 0.00278 | **0.955** |

The 1 h runs end at a **lower** training loss than the 10 h runs while
their WER is twice as high. Training loss is not a proxy for recognition
here -- it is the signature of a model that has memorised its transcripts.
1 h is roughly 1,000 utterances and the schedule is 3,000 steps, so the
predictor sees the same transcripts on the order of 250 times.

The predictor is the only module positioned to memorise them. It is an
autoregressive model over labels alone (it never sees audio), it is
randomly initialised while the encoder is pre-trained, and
`--head_lr_mult 10.0` gives it ten times the encoder's learning rate --
the flag's own help says a single LR "either wrecks the encoder or starves
the heads".

So: give the predictor a language model trained on text the paired audio
does not constrain, and optionally stop the transcripts from overwriting
it.

**On the resource budget.** "Low-resource" in this literature constrains
*labelled audio*, not text or unlabelled audio. wav2vec 2.0 itself
fine-tunes on 1 h/10 h while using a language model built from the
LibriSpeech LM corpus, and its encoder is pre-trained on 960 h of
unlabelled audio. Using `librispeech-lm-norm.txt` is therefore inside the
standard 1 h/10 h protocol. Using the *train-960 transcripts* would not
be: those are labels for audio we are claiming not to have.

## 2. Corpus

| | |
|---|---|
| source | <https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz> |
| downloaded | 2026-09-19 18:10 KST |
| compressed | 1,507,274,412 B (1.51 GB) |
| uncompressed | 4,287,216,164 B (4.29 GB) |
| words | ~800 M |
| normalisation | upper case, no punctuation -- same convention as the LibriSpeech transcripts, so no extra text processing is needed |

Also downloaded, unused so far but needed for shallow fusion:

| | |
|---|---|
| `4-gram.arpa.gz` | 1,355,172,078 B -- feeds `wav2vec2_inference.py --lm` once converted to a KenLM binary |
| `librispeech-vocab.txt` | 1,737,588 B |

All three live in `/mnt/synology_nas_00/chanwcom/lm_resources/` (NAS).
They are deliberately **not** under `/home`: u22's root filesystem hit
100% on 2026-09-16 and killed a 13,000-step CTC cell outright.

## 3. Tokenisation

`lm_resources/tokenize_corpus.py`.

The vocabulary must be the one the transducer already uses, or the
embedding cannot be transferred:

```
cognitive_workflow_kit/resources/spm/librispeech_unigram_32.model
```

32 pieces, character level:

```
['<pad>', '</s>', '<s>', '<unk>', '▁',
 'E','A','T','S','I','O','R','N','H','L','D','U','M','C','Y','P','F',
 'B','G','V','W','K',"'",'X','J','Q','Z']
```

`<pad>` is id 0 and the transducer uses id 0 as blank
(`[model] vocab=32 blank=0` in every run log).

Two decisions worth recording:

* **Each line is prefixed with blank (id 0).** The RNN-T predictor is fed
  `[blank, y_1, ..., y_U]`, so blank occupies the BOS slot. The LM has to
  learn the same convention or the transferred weights see an input
  distribution they were never trained on.
* **Output is a flat `uint8` array.** A 32-symbol vocabulary fits in one
  byte, so 4.33 B tokens occupy 4.33 GB and training reads it through
  `np.memmap` -- data loading cost is effectively zero and no dataloader
  workers are needed.

Result: `lm_resources/lm_tokens.u8`, 4,327,634,423 B = **4.33 B tokens**,
20 worker processes, ~14 min wall.

## 4. LM architecture and training

`lm_resources/train_char_lm.py`. The body is deliberately identical in
shape to `LstmPredictor`:

```python
embed : nn.Embedding(32, hidden)                     # -> predictor
rnn   : nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)   # -> predictor
out   : nn.Linear(hidden, 32)                        # discarded
```

`out` is dropped on transfer: the predictor emits a representation that is
added inside the joint, not a distribution over labels.

`hidden = 512`, not 320. The stock predictor's LSTM emits `joint_dim`
(320) straight into the joint's addition, which has two consequences --
the LM would have to be squeezed into 320 units, and freezing the LSTM
would leave *nothing* trainable in the predictor path. Sizing the body at
512 and projecting down to 320 fixes both (see §5).

| hyper-parameter | value |
|---|---|
| hidden | 512 |
| layers | 1 (LSTM) |
| params | 2.13 M (body) + output layer |
| sequence length | 256 |
| batch | 128 sequences (= 32,768 tokens/step) |
| steps | 120,000 |
| tokens seen | 3.93 B (~0.91 epochs of the corpus) |
| optimiser | AdamW, weight decay 0.01, grad-norm clip 1.0 |
| LR | 2e-3 peak, 2,000-step linear warmup, cosine to 0 |
| validation | last 2 M tokens held out |
| device | one RTX 4090 (GPU 3) |
| wall clock | **48.9 min** (24.5 ms/step) |

Result: `lm_resources/char_lm_h512.pt`, 8.5 MB, saved at step 120,000 with
`{"model": state_dict, "hidden": 512, "vocab": 32, "step": 120000}`.

Final numbers:

```
[120000/120000] train=1.0798 val=1.3454 bpc=1.9409 lr=0.00e+00 seen=3.93B
```

* Validation loss oscillates between 1.29 and 1.36 across the last 10 k
  steps; bpc 1.86-1.94.
* **Caveat to resolve before drawing conclusions from these numbers:**
  validation is measured on a *single* random crop per logging step, so
  most of that oscillation is measurement noise, and the 0.25 gap between
  train (1.08) and val (1.29-1.35) has not been separated from genuine
  over-fitting. Re-measure over many batches before quoting bpc anywhere.
* bpc here is not comparable to the usual English character-LM figures
  (1.2-1.5): this vocabulary is a 32-piece SentencePiece including an
  explicit word-boundary symbol, not raw characters.

## 5. Loading the LM into the predictor

`wav2vec2_rnnt_lminit.py`, a copy of `wav2vec2_rnnt.py` with four changes.

**(a) `LstmPredictor` gains a projection.** New argument `lm_hidden`:

```python
body = lm_hidden or hidden
self.embed = nn.Embedding(vocab_size, body)
self.rnn   = nn.LSTM(body, body, num_layers=1, batch_first=True)
self.proj  = nn.Linear(body, hidden) if lm_hidden else nn.Identity()
```

`proj` is applied in **both** `forward` and `step` -- `step` is the
incremental path used for decoding, and missing it there would make
decoding disagree with training.

With `lm_hidden=None` the module is `nn.Identity()` and the file behaves
exactly like the original.

**(b) Three new flags.**

| flag | effect |
|---|---|
| `--predictor_hidden 512` | width of embed+LSTM; enables `proj` |
| `--predictor_init <ckpt>` | loads `embed.*` and `rnn.*` only; raises if any key is unexpected, which is what catches a width mismatch |
| `--freeze_predictor` | `requires_grad = False` on embed and rnn |

**(c) Frozen parameters are dropped from the optimiser**, not merely
zero-grad: the parameter groups now filter on `p_.requires_grad`.

**(d) Nothing else changes.** Same encoder, same joint, same loss, same
schedule.

Smoke test (60 steps, 1 h profile, condition C):

```
[predictor] init from .../char_lm_h512.pt (lm hidden=512 step=120000) loaded=6 missing=[...] unexpected=[]
[predictor] frozen 2.1M params; trainable left: proj.weight, proj.bias
[model] vocab=32 blank=0 trainable=90.8M stride=2 joint_dim=320
[20/60] loss=840.016 ... [40/60] loss=456.347 ... [60/60] loss=335.549
```

`unexpected=[]` confirms the width matched; `missing` lists `proj.*`,
which is correct -- the LM has no such layer.

## 6. The experiment

Three conditions, 1 h, `alpha = 0` (plain transducer, no smoothing), seeds
0-4, evaluated on the full dev splits.

| | predictor body | initialised from | trained |
|---|---|---|---|
| **A** | 320, no `proj` | random | yes |
| **B** | 512 + `proj` | `char_lm_h512.pt` | yes |
| **C** | 512 + `proj` | `char_lm_h512.pt` | **frozen** (only `proj`) |

### 6a. Width 512 was a mistake; the 320 re-run

The only structural requirement was an adapter, not a wider body. The
stock predictor's LSTM emits `joint_dim` straight into the joint's
addition, so freezing embed+LSTM leaves the predictor path constant --
`proj` exists to fix exactly that, and `Linear(320, 320)` would have done
it. Widening to 512 was chosen for LM quality and bought a confound: B and
C differ from A in initialisation, in freezing, **and** in predictor
capacity (LSTM 0.82 M -> 2.10 M).

| | embed | LSTM | proj | predictor total |
|---|---|---|---|---|
| stock (no proj) | 10,240 | 821,760 | - | 832,000 |
| 320 + proj | 10,240 | 821,760 | 102,720 | 934,720 |
| 512 + proj | 16,384 | 2,101,248 | 164,160 | 2,281,792 |

`proj` is small either way -- 0.11-0.18 % of the 91.5 M model. The real
delta is the LSTM.

The re-run fixes the width at 320 for every condition, including a
random-init control that also carries `proj`, so the conditions differ
only in initialisation and freezing:

| | predictor | init | trained |
|---|---|---|---|
| **A320** | 320 + `proj` | random | yes |
| **B320** | 320 + `proj` | `char_lm_h320.pt` | yes |
| **C320** | 320 + `proj` | `char_lm_h320.pt` | frozen |
| **E320** | 320 + `proj` | **random** | **frozen** |

**E320 is the sharp control.** If E ≈ C, the language model contributed
nothing and the whole effect is "the predictor was prevented from
memorising the 1,000 training transcripts". If E ≈ A, the LM knowledge is
real. Nothing in the 512 run could separate these.

Seeds 0-2, 12 cells. A and E do not need the LM, so they are queued first
and cover the ~35 min the 320-wide LM takes to train.

The 512 LM (`char_lm_h512.pt`) cannot be loaded into a 320-wide body, so
a second LM is trained with `--hidden 320`; everything else about the LM
recipe is unchanged.

### 6b. First results from the 512 run (seeds 0-1, full dev)

Kept because they motivated the re-run, not as a result to quote.

| condition | dev-clean | dev-other |
|---|---|---|
| A (stock, 5 seeds) | 0.18648 ± 0.00081 | 0.27935 ± 0.00278 |
| B (LM init, trained) | 0.19265 / 0.20354 | 0.27730 / 0.28584 |
| **C (LM init, frozen)** | **0.16970 / 0.17203** | **0.26155 / 0.25899** |

C beats A by ~0.017 dev-clean, roughly twenty times A's seed sd. B is
*worse* than A, which is consistent with the initialisation being
overwritten (head LR is 10x the encoder's) while the extra capacity
remains available for memorisation. That pattern points at freezing rather
than at the LM as the active ingredient -- which is what E320 tests.

A is the existing 1 h baseline and is already complete:
`rnnt_logs/1hr/1hr_base_a0.0_s{0..4}.log`. B and C are 10 cells, ~30 min
each, two GPUs.

**What each condition tests.** B asks whether a pre-trained body helps at
all. C asks whether the memorisation path is what costs the 1 h runs their
WER: with embed+LSTM frozen, the predictor cannot fit the 1,000 training
transcripts, so if the 1 h penalty is memorisation, C should beat both A
and B. B > A with C ≈ A would instead mean the initialisation helps but is
being overwritten (the head LR is 10x), which is testable by lowering
`--head_lr_mult`.

**Why this is decidable at n=5.** The A baseline's seed spread is
0.00081 on dev-clean and 0.00278 on dev-other, so a shift of ~0.005 on
dev-clean is several standard deviations. That is unusually tight for this
project -- the CTC grids had a seed sd of 0.0028 and could not separate
alpha values at all.

**Confounds to keep in mind.**

* B and C add `proj` (512->320, 164 k params) and widen the body from 320
  to 512, so they differ from A in capacity as well as initialisation. A
  clean ablation needs a fourth condition: `--predictor_hidden 512`
  *without* `--predictor_init`. It is not queued yet.
* Frozen-vs-trained comparisons usually favour freezing in low resource
  and reverse at higher resource. If C wins at 1 h, it should be checked
  at 10 h before being described as a general result.

## 7. Where everything is

| what | path |
|---|---|
| corpus, tokens, LM checkpoint, scripts | `/mnt/synology_nas_00/chanwcom/lm_resources/` |
| LM training log | `lm_resources/pipeline.log` |
| RNN-T variant | `selective_estimated_target_smoothing/wav2vec2_rnnt_lminit.py` |
| B/C worker + queue + claims | `~/rnnt1hr_lminit/` (u22-local) |
| B/C logs | `models/<host>/rnnt_logs/1hr_lminit/` (NAS) |
| B/C checkpoints | `models/<host>/rnnt1hr_lminit/` (NAS) |
| A logs (existing baseline) | `models/<host>/rnnt_logs/1hr/` and `rnnt_logs_u22/1hr/` |

`CHECKPOINT_TOP_DIR` resolves to `models/$(hostname)` = `models/u22`,
which is a symlink to `models/slpl_4090_00` where the CTC runs already
live.

## 8. Reproducing

```bash
cd /mnt/synology_nas_00/chanwcom/lm_resources
# corpus -> tokens (20 workers, ~14 min)
python tokenize_corpus.py --text librispeech-lm-norm.txt --out lm_tokens.u8 --workers 20
# tokens -> LM (one 4090, ~49 min)
CUDA_VISIBLE_DEVICES=3 python train_char_lm.py \
    --tokens lm_tokens.u8 --out char_lm_h512.pt \
    --hidden 512 --seq 256 --batch 128 --steps 120000
# LM -> transducer (condition C; drop --freeze_predictor for B)
cd ../local_repository/selective_estimated_target_smoothing
python wav2vec2_rnnt_lminit.py --finetune_profile libri_light_1hr \
    --max_steps 3000 --warmup_steps 1000 --num_stable_steps 1500 --num_decay_steps 500 \
    --dynamic_batching --max_batch_audio_len 6400000 --grad_accum 1 \
    --max_sample_audio_len 480000 --max_label_len 450 \
    --dataloader_num_workers 2 --encoder_stride 2 --predictor lstm \
    --predictor_hidden 512 --predictor_init ../../lm_resources/char_lm_h512.pt \
    --freeze_predictor --alpha 0.0 --beta 0.0 --alpha_mode fixed --seed 0 \
    --eval_steps 1000 --eval_examples 200 --log_every 500 \
    --output_dir <dir>
```

## 9. Open items

1. Re-measure LM validation over many batches (§4 caveat).
2. The 320 re-run (§6a) supersedes the capacity control that condition D
   would have provided; D itself is not queued.
3. If C320 wins at 1 h, repeat at 10 h.
4. Build the KenLM binary from `4-gram.arpa.gz` and run shallow fusion --
   independent of all of the above, and it is the cheaper way to inject
   language knowledge.
5. The periodic in-training eval is 200 utterances and is for watching the
   curve only; the numbers quoted here are full-split decodes.
