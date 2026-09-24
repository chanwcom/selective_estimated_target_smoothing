"""Fine-tunes a wav2vec2 encoder as an RNN-Transducer, with SHC smoothing.

Deliberately parallel to `wav2vec2_finetuning_sets.py`: same webdataset
pipeline, same SentencePiece vocabulary, same WSD schedule, same run-name
convention. The ONLY intended difference is the loss -- CTC there, the
transducer lattice here -- so that an alpha tuned under CTC can be checked
for transfer without anything else moving.

Model
-----
  encoder   : facebook/wav2vec2-base, pre-trained (standard practice for
              transducers too -- e.g. BigSSL, SpeechStew, NeMo's
              SSL-initialised Conformer-Transducer recipes). Its output is
              optionally strided in time; transducers are usually run at a
              lower frame rate than CTC, and the joint tensor is
              (B, T, U+1, C) so halving T halves the dominant allocation.
  predictor : stateless by default -- an embedding of the last
              `predictor_context` labels (Ghodsi et al. 2020). Chosen over
              an LSTM because it has almost no internal language model,
              which keeps a later shallow-fusion analysis interpretable:
              any LM gain is the external LM's, not the predictor's.
  joint     : tanh(W_e h_enc + W_p h_pred) -> linear to C.

Memory
------
The joint output is the whole cost: B x T x (U+1) x C floats, plus the
same again for the smoothed target and the gradient. At C = 32 (character
SentencePiece) that is affordable, which it would not be with a
1000-piece vocabulary. `--max_sample_audio_len` and `--encoder_stride`
are the two knobs that matter.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Wav2Vec2Model
from transformers.optimization import get_wsd_schedule

import repo_config
from common import sample_util
from cwk.loss.pytorch import rnnt_shc_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wav2vec2_inference import (DataCollatorCTCWithPadding,  # noqa: E402
                                Wav2Vec2SPMTokenizer, clean_special_tokens)


class StatelessPredictor(nn.Module):
    """Embedding of the last `context` labels, concatenated then projected.

    No recurrence, so a step costs one gather and one matmul and the whole
    (U+1) axis is computed in parallel. `context` = 1 makes the predictor
    depend only on the immediately preceding label, which is the variant
    with the weakest internal LM.
    """

    def __init__(self, vocab_size, hidden, context=2, blank=0):
        super().__init__()
        self.context = context
        self.blank = blank
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden * context, hidden)

    def forward(self, labels):
        """labels: (B, U) -> (B, U+1, hidden), one row per lattice column u.

        Column u must see labels[:, :u], i.e. the u labels already emitted,
        so the window for column u ends at u-1. Positions before the start
        of the sequence are filled with blank.
        """
        b, u = labels.shape
        pad = labels.new_full((b, self.context), self.blank)
        padded = torch.cat([pad, labels.clamp(min=0)], dim=1)  # (B, C+U)
        # For column u (0..U) take padded[:, u : u + context].
        idx = (torch.arange(u + 1, device=labels.device).view(-1, 1)
               + torch.arange(self.context, device=labels.device).view(1, -1))
        win = padded[:, idx]                                   # (B, U+1, C)
        return self.proj(self.embed(win).flatten(2))

    def step(self, last_label, state):
        """Same interface as LstmPredictor.step; `state` is the label window."""
        if state is None:
            state = last_label.new_full((last_label.shape[0], self.context),
                                        self.blank)
        win = torch.cat([state[:, 1:], last_label.unsqueeze(1)], dim=1)
        return self.proj(self.embed(win).flatten(1)), win


class LstmPredictor(nn.Module):
    """Single-layer LSTM over the emitted labels, blank-prepended.

    Column u must see labels[:, :u], so the input sequence is
    [blank, y_1, ..., y_U] and its u-th output is the state after u
    labels. Unlike the stateless variant this can represent position, and
    the overfit sanity check needs that: with a 2-label context the same
    context recurs many times within one transcript with different
    continuations, which caps how well any model can fit it.
    """

    def __init__(self, vocab_size, hidden, blank=0):
        super().__init__()
        self.blank = blank
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)

    def forward(self, labels):
        b, _ = labels.shape
        start = labels.new_full((b, 1), self.blank)
        seq = torch.cat([start, labels.clamp(min=0)], dim=1)   # (B, U+1)
        out, _ = self.rnn(self.embed(seq))
        return out                                             # (B, U+1, H)

    def step(self, last_label, state):
        """One incremental step, for decoding.

        Recomputing `forward` over the whole prefix at every frame costs
        O(T*U) LSTM work per utterance, which is what made full-split
        evaluation unaffordable (0.67 s per utterance). Carrying the state
        makes it O(T + U).
        """
        out, new_state = self.rnn(self.embed(last_label).unsqueeze(1), state)
        return out[:, 0], new_state


class RnntModel(nn.Module):
    """wav2vec2 encoder + stateless predictor + additive joint."""

    def __init__(self, vocab_size, encoder_name="facebook/wav2vec2-base",
                 joint_dim=320, predictor_context=2, encoder_stride=2,
                 blank=0, freeze_feature_encoder=True,
                 predictor="lstm"):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(encoder_name)
        if freeze_feature_encoder:
            # Same choice the CTC script makes: the convolutional feature
            # extractor is kept frozen, so the two setups differ only in
            # the loss.
            self.encoder.feature_extractor._freeze_parameters()
        enc_dim = self.encoder.config.hidden_size
        self.encoder_stride = encoder_stride
        self.enc_proj = nn.Linear(enc_dim * encoder_stride, joint_dim)
        self.predictor = (
            LstmPredictor(vocab_size, joint_dim, blank=blank)
            if predictor == "lstm" else
            StatelessPredictor(vocab_size, joint_dim,
                               context=predictor_context, blank=blank))
        self.out = nn.Linear(joint_dim, vocab_size)
        self.blank = blank

    def encode(self, input_values, attention_mask):
        h = self.encoder(input_values,
                         attention_mask=attention_mask).last_hidden_state
        lens = self.encoder._get_feat_extract_output_lengths(
            attention_mask.sum(-1)).to(torch.long)
        s = self.encoder_stride
        if s > 1:
            # Reshape-stack rather than slice: keeping all frames and
            # folding them into the channel axis loses no information,
            # which plain subsampling would.
            b, t, d = h.shape
            pad = (-t) % s
            if pad:
                h = torch.cat([h, h.new_zeros(b, pad, d)], dim=1)
            h = h.reshape(b, (t + pad) // s, d * s)
            lens = torch.div(lens + s - 1, s, rounding_mode="floor")
        return self.enc_proj(h), lens

    def forward(self, input_values, attention_mask, labels):
        enc, enc_lens = self.encode(input_values, attention_mask)   # (B,T,J)
        pred = self.predictor(labels)                               # (B,U+1,J)
        joint = torch.tanh(enc.unsqueeze(2) + pred.unsqueeze(1))
        return self.out(joint), enc_lens                            # (B,T,U+1,C)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finetune_profile", default="libri_light_1hr")
    p.add_argument("--vocab_size", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--alpha_mode", default="fixed",
                   choices=["fixed", "active_support",
                            "floored_active_support",
                            "frame_label_support",
                            "diagonal_active_support",
                            "aws", "diagonal_occupancy", "mos",
                            "diagonal_projected", "aws_alpha_beta",
                            "alignment_biased", "asap"])
    p.add_argument("--fas_eps", type=float, default=1e-10)
    p.add_argument("--alpha_off_step", type=int, default=0,
                   help="Turn the smoothing OFF after this many optimiser "
                        "steps (0 keeps it on for the whole run). Measured "
                        "on 10 h, FAS alpha=0.10 eps=1e-10 beats the "
                        "baseline by 1.6 %% at step 2000 and then loses to "
                        "it by 33.8 %% at step 6000, so the smoothing helps "
                        "the early search and blocks the late sharpening. "
                        "This exposes that split as a knob.")
    p.add_argument("--gate", default="none",
                   choices=["none", "low", "high", "blank_only", "label_only"],
                   help="Restrict smoothing by the unsmoothed target's max: "
                        "'low' exempts nodes already above --gate_thresh "
                        "(breaks the self-referential flattening loop), "
                        "'high' smooths only those (penalize overconfidence).")
    p.add_argument("--gate_thresh", type=float, default=0.9)
    p.add_argument("--sharpen", type=float, default=0.0,
                   help="Target sharpening, the opposite of smoothing: at "
                        "every node already putting more than "
                        "--sharpen_thresh on one class, move the target "
                        "that fraction of the way to a one-hot (1.0 snaps "
                        "it). Takes the loss from marginalizing over "
                        "alignments toward a hard Viterbi alignment. "
                        "Applies to blank- and label-dominant nodes alike, "
                        "and is independent of --alpha.")
    p.add_argument("--sharpen_thresh", type=float, default=0.9)
    p.add_argument("--diag_align", default="departure",
                   choices=["departure", "arrival"],
                   help="diagonal_active_support only: whether a node reads the transitions leaving its own anti-diagonal (departure, the default and the one consistent with how this loss defines its target) or those arriving at it.")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--num_stable_steps", type=int, default=None)
    p.add_argument("--num_decay_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Fixed samples per batch. Ignored when "
                        "--dynamic_batching is set.")
    p.add_argument("--dynamic_batching", action="store_true", default=False,
                   help="Use the same length-budget batching the CTC script "
                        "uses, so the data pipeline is identical and the "
                        "loss is the only difference between the two setups.")
    p.add_argument("--max_batch_audio_len", type=int, default=6400000,
                   help="Length budget per batch, in audio samples. Defaults "
                        "to the CTC runs' 6400000 (400 s) so the optimizer "
                        "sees the same batch and the loss stays the only "
                        "difference between the two setups.\n"
                        "Measured peak memory on 100hr at --encoder_stride 2 "
                        "(80 steps each, full-epoch batch-shape scan for the "
                        "worst case):\n"
                        "  1600000 ->  5.80 GiB   (B ~ 7)\n"
                        "  3200000 -> 10.86 GiB   (B ~ 14)\n"
                        "  6400000 -> 20.68 GiB   (B ~ 27)\n"
                        "So 6400000 is one job per 32 GB card. On a 24 GB "
                        "card it leaves only ~2.3 GiB of headroom, which is "
                        "not enough to sit on for days; use 3200000 with "
                        "--grad_accum 2 there instead -- same effective "
                        "batch, 10.86 GiB. The transducer needs more than "
                        "CTC at the same budget (CTC fits two jobs per card "
                        "at gpu_memory_fraction 0.46 = 14.4 GiB) because the "
                        "joint tensor is B x T x (U+1) x C and the budget "
                        "bounds B*T_max but nothing bounds U_max; that is "
                        "also why --max_label_len exists.")
    p.add_argument("--max_dynamic_batch_size", type=int, default=None)
    p.add_argument("--length_bucket_window_mult", type=int, default=50)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--max_sample_audio_len", type=int, default=480000,
                   help="Samples longer than this are dropped. Matches the "
                        "CTC script's default (30 s). libri-light's "
                        "segments are long: a 7.5 s cap left only 28 "
                        "utterances of the 1 h subset, i.e. 7 batches per "
                        "epoch. The joint tensor grows linearly in T, so "
                        "this and --encoder_stride are the memory knobs.")
    p.add_argument("--max_label_len", type=int, default=None,
                   help="Drop samples whose transcript exceeds this many "
                        "tokens. The audio filter alone does not bound the "
                        "joint tensor: it is B x T x (U+1) x C, and the "
                        "length budget bounds B*T_max but nothing bounds "
                        "U_max. On LibriSpeech the two are tightly "
                        "correlated (~15 chars/s, so a 30 s cap implies "
                        "U <~ 450 and the measured max is 392), but that "
                        "is a property of this corpus -- on a set whose "
                        "transcripts run longer per second the same audio "
                        "budget costs proportionally more memory. Setting "
                        "this makes peak memory a guarantee rather than an "
                        "observation.")
    p.add_argument("--encoder_stride", type=int, default=2)
    p.add_argument("--joint_dim", type=int, default=320)
    p.add_argument("--predictor_context", type=int, default=2)
    p.add_argument("--predictor", default="lstm",
                   choices=["lstm", "stateless"])
    p.add_argument("--head_lr_mult", type=float, default=10.0,
                   help="Predictor/joint/output LR relative to the "
                        "encoder's. The encoder is pre-trained and the "
                        "heads are random, so a single LR either wrecks "
                        "the encoder or starves the heads.")
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--eval_examples", type=int, default=200,
                   help="Utterances per split for the PERIODIC eval. 200 is "
                        "about 4000 reference words, whose sampling error "
                        "(~0.004 WER) is larger than the differences between "
                        "methods (0.001-0.003) -- it is a curve to watch, "
                        "not a number to compare.")
    p.add_argument("--final_eval_examples", type=int, default=0,
                   help="Utterances per split for the eval at --max_steps. "
                        "0 means the whole split, which is what makes that "
                        "one comparable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_memory_fraction", type=float, default=None,
                   help="Cap the fraction of the card this process may use. "
                        "Its purpose here is to rehearse a smaller card on a "
                        "bigger one: 23/31.4 = 0.73 makes a 32 GB 5090 behave "
                        "like a 24 GB 4090, so whether the run fits there is "
                        "measured rather than extrapolated. Note the log's "
                        "`peak` is max_memory_ALLOCATED, which is well below "
                        "what the process holds on the card -- 20.90 vs "
                        "27.70 GiB was measured at --max_batch_audio_len "
                        "6400000 -- so the allocated figure must not be "
                        "compared against card capacity directly.")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Micro-batches per optimizer step. Pair with a "
                        "halved --max_batch_audio_len to keep the effective "
                        "batch while fitting a 24 GB card. Note the loss is "
                        "then averaged per micro-batch rather than per "
                        "utterance, which differs slightly from a single "
                        "large dynamic batch.")
    p.add_argument("--overfit_batches", type=int, default=0,
                   help="Sanity check: train on this many fixed batches "
                        "only and decode them back. A correct transducer "
                        "pipeline drives the loss to near zero and "
                        "reproduces the transcripts; if it cannot, the bug "
                        "is in the code rather than in the budget.")
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    return p.parse_args()


def build_processor(vocab_size):
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    spm = os.path.join(repo_config.RESOURCE_TOP_DIR,
                       f"librispeech_unigram_{vocab_size}.model")
    processor.tokenizer = Wav2Vec2SPMTokenizer(spm)
    return processor, spm


@torch.no_grad()
def greedy_decode(model, processor, batch, device, max_symbols=5):
    """Frame-synchronous greedy transducer decode, batched.

    Every utterance in the batch advances one frame per outer iteration,
    so the Python loop runs T_max times rather than sum_b T_b, and the
    predictor is stepped incrementally instead of re-run over the whole
    prefix. Together those took a 400-utterance evaluation from 4.5 min to
    seconds, which is what makes evaluating the full dev splits -- 5567
    utterances -- affordable at all; at the old cost a single full eval was
    62 min against 2.1 h of training.

    The emitted sequence is identical to the per-utterance version: the
    same argmax is taken in the same order, only the batching changes.
    """
    model.eval()
    # cuDNN's fused LSTM gives a different answer for a length-41 call than
    # for 41 length-1 calls -- measured 2.6e-3 relative, which is enough to
    # flip an argmax at a borderline frame and then diverge for the rest of
    # the utterance (19 of 32 hypotheses differed). The step-by-step path is
    # the one decoding must use, so cuDNN is disabled here; with it off the
    # two agree exactly (0.0e0, and to 3e-15 in float64 on CPU). The
    # predictor is a single 320-unit layer, so the 4.8x slowdown on that one
    # module is small against the encoder and the Python loop.
    _cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        enc, enc_lens = model.encode(batch["input_values"].to(device),
                                     batch["attention_mask"].to(device))
        return _greedy_decode_inner(model, processor, enc, enc_lens,
                                     max_symbols)
    finally:
        torch.backends.cudnn.enabled = _cudnn
        model.train()


@torch.no_grad()
def _greedy_decode_inner(model, processor, enc, enc_lens, max_symbols):
    device = enc.device
    b, t_max, _ = enc.shape
    blank = model.blank
    last = torch.full((b,), blank, dtype=torch.long, device=device)
    state = None
    pred, state = model.predictor.step(last, state)     # state after u = 0
    hyps = [[] for _ in range(b)]
    n_emit = torch.zeros(b, dtype=torch.long, device=device)
    for t in range(t_max):
        alive = (t < enc_lens.to(device))
        if not bool(alive.any()):
            break
        n_emit.zero_()
        for _ in range(max_symbols):
            logit = model.out(torch.tanh(enc[:, t] + pred))      # (B, C)
            k = logit.argmax(-1)
            # An utterance keeps emitting only while it is alive, has not
            # hit the per-frame cap, and did not just produce blank.
            emit = alive & (k != blank) & (n_emit < max_symbols)
            if not bool(emit.any()):
                break
            idx = torch.nonzero(emit, as_tuple=True)[0]
            for i in idx.tolist():
                hyps[i].append(int(k[i]))
            n_emit = n_emit + emit.long()
            # Step the predictor only for the utterances that emitted; the
            # others must keep their state, so the new state is merged in.
            new_pred, new_state = model.predictor.step(
                torch.where(emit, k, last), state)
            m = emit.view(-1, 1)
            pred = torch.where(m, new_pred, pred)
            if isinstance(state, tuple):
                state = tuple(
                    torch.where(emit.view(1, -1, 1), n, o)
                    for n, o in zip(new_state, state))
            else:
                state = torch.where(m, new_state, state)
            last = torch.where(emit, k, last)
    return [clean_special_tokens(processor.tokenizer.decode(
        h, group_tokens=False)) for h in hyps]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    prof = __import__("wav2vec2_finetuning_sets")._FINETUNE_PROFILES[
        args.finetune_profile]
    max_steps = args.max_steps or prof["max_steps"]
    warmup = args.warmup_steps if args.warmup_steps is not None else prof[
        "warmup_steps"]

    if args.gpu_memory_fraction and args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        print(f"[mem] capped at {args.gpu_memory_fraction:.3f} of the card "
              f"({args.gpu_memory_fraction * torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)",
              flush=True)

    processor, spm = build_processor(args.vocab_size)
    vocab = len(processor.tokenizer)
    blank = processor.tokenizer.pad_token_id
    collator = DataCollatorCTCWithPadding(processor=processor,
                                          padding="longest")
    db = repo_config.DB_TOP_DIR
    train_top_dir = os.path.join(db, prof["train_subdir"])
    if args.dynamic_batching:
        # Byte-for-byte the CTC script's call, so the sample stream, the
        # length filter, the bucketing window and the collation are the
        # same objects; only `max_batch_length` is set lower (see the
        # flag's help).
        train_ds = sample_util.make_dataset(
            train_top_dir, True, spm,
            dynamic_batch=sample_util.DynamicBatchConfig(
                collate_fn=collator,
                max_batch_length=args.max_batch_audio_len,
                max_batch_size=args.max_dynamic_batch_size,
                window_mult=args.length_bucket_window_mult,
                seed=args.seed),
            max_sample_length=args.max_sample_audio_len)
    else:
        train_ds = sample_util.make_dataset(
            train_top_dir, True, spm,
            batch_size=args.batch_size,
            length_bucket_window_mult=args.length_bucket_window_mult,
            max_sample_length=args.max_sample_audio_len, seed=args.seed)
    # Periodic eval runs on BOTH dev splits. One split is not enough to
    # pick anything: dev-clean's spread across methods is smaller than the
    # seed-to-seed spread, so dev-other is what carries the signal.
    dev_sets = {
        name: sample_util.make_dataset(
            os.path.join(db, f"librispeech/webdataset/{name}"), True, spm,
            max_sample_length=args.max_sample_audio_len)
        for name in ("dev-clean", "dev-other")}

    model = RnntModel(vocab, joint_dim=args.joint_dim,
                      predictor_context=args.predictor_context,
                      encoder_stride=args.encoder_stride,
                      blank=blank, predictor=args.predictor).to(args.device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] vocab={vocab} blank={blank} trainable={n_par/1e6:.1f}M "
          f"stride={args.encoder_stride} joint_dim={args.joint_dim}",
          flush=True)

    head = [p_ for n, p_ in model.named_parameters()
            if not n.startswith("encoder.")]
    enc = [p_ for n, p_ in model.named_parameters()
           if n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": args.learning_rate},
         {"params": head, "lr": args.learning_rate * args.head_lr_mult}],
        weight_decay=0.005)
    stable = (args.num_stable_steps if args.num_stable_steps is not None
              else max(0, max_steps - warmup) // 2)
    decay = (args.num_decay_steps if args.num_decay_steps is not None
             else max_steps - warmup - stable)
    sched = get_wsd_schedule(opt, num_warmup_steps=warmup,
                             num_stable_steps=stable,
                             num_decay_steps=decay, decay_type="linear")
    print(f"[lr] WSD warmup {warmup} + stable {stable} + decay {decay} "
          f"= {warmup + stable + decay} (max_steps={max_steps})", flush=True)

    # Dynamic batching yields already-collated batches (the collate_fn
    # lives in DynamicBatchConfig), so the loader must not batch again --
    # the CTC script relies on the same distinction. The fixed path only
    # length-buckets the sample stream and leaves collation to the loader,
    # which is what HF Trainer's data_collator does there.
    train_loader = (
        DataLoader(train_ds, batch_size=None,
                   num_workers=args.dataloader_num_workers)
        if args.dynamic_batching else
        DataLoader(train_ds, batch_size=args.batch_size,
                   collate_fn=collator,
                   num_workers=args.dataloader_num_workers))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))
    step = 0
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    epoch = 0
    dropped = 0

    def _batches():
        """Re-iterates the loader until max_steps.

        The CTC script gets this for free: HF Trainer rebuilds the
        iterator every epoch, so `max_steps` can exceed one pass. A plain
        loop has to do it explicitly -- without this, a 1 h subset with
        batch_size 4 ends after a few dozen steps and the run silently
        reports nothing.
        """
        nonlocal epoch
        while True:
            n = 0
            for bb in train_loader:
                n += 1
                yield bb
            epoch += 1
            print(f"[epoch {epoch}] {n} batches", flush=True)
            if n == 0:
                raise RuntimeError("training loader yielded no batches")

    fixed = None
    if args.overfit_batches:
        fixed = []
        for bb in train_loader:
            fixed.append(bb)
            if len(fixed) >= args.overfit_batches:
                break
        print(f"[overfit] {len(fixed)} batches, "
              f"labels={[int((b['labels'] != -100).sum()) for b in fixed]}",
              flush=True)

    def _fixed_batches():
        while True:
            for bb in fixed:
                yield bb

    for batch in (_fixed_batches() if fixed else _batches()):
        if step >= max_steps:
            break
        lab_all = batch["labels"]
        if args.max_label_len:
            keep = ((lab_all != -100).sum(-1) <= args.max_label_len)
            n_drop = int((~keep).sum())
            if n_drop:
                dropped += n_drop
                if not bool(keep.any()):
                    continue
                batch = {k: v[keep] for k, v in batch.items()}
        iv = batch["input_values"].to(args.device)
        am = batch["attention_mask"].to(args.device)
        lab = batch["labels"].clone()
        lab_lens = (lab != -100).sum(-1)
        lab = lab.masked_fill(lab == -100, blank).to(args.device)
        lab_lens = lab_lens.to(args.device)
        if int(lab_lens.max()) == 0:
            continue
        # Trim to the batch's real label width; the joint tensor is
        # (B, T, U+1, C) so a stray padded column costs real memory.
        lab = lab[:, :int(lab_lens.max())]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(args.device == "cuda")):
            logits, enc_lens = model(iv, am, lab)
        # The lattice recursions are run outside autocast: they accumulate
        # T + U logaddexp's and bf16 drifts too far (see
        # calculate_rnnt_alpha_beta).
        # Positional through to `diag_align`, so asap_eps has to be given
        # explicitly even though only alpha_mode="asap" reads it.
        # alpha is read per step, not once: --alpha_off_step drops it to 0
        # partway through. `step` counts optimiser steps, so every
        # micro-batch inside one accumulation window sees the same value.
        cur_alpha = (0.0 if (args.alpha_off_step
                             and step >= args.alpha_off_step)
                     else args.alpha)
        loss = rnnt_shc_loss.RnntShcLoss.apply(
            lab, lab_lens, logits.float(), enc_lens, blank,
            cur_alpha, args.beta, args.alpha_mode, args.fas_eps,
            1e-3, args.diag_align, args.gate, args.gate_thresh,
            args.sharpen, args.sharpen_thresh)
        loss = loss.mean() / args.grad_accum
        scaler.scale(loss).backward()
        run_loss += float(loss.detach()) * args.grad_accum
        run_n += 1
        if run_n % args.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            args.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % args.log_every == 0:
                peak = (torch.cuda.max_memory_allocated() / 2**30
                        if args.device == "cuda" else 0.0)
                resv = (torch.cuda.max_memory_reserved() / 2**30
                        if args.device == "cuda" else 0.0)
                print(f"[{step}/{max_steps}] loss={run_loss/run_n:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"B={logits.shape[0]} T={logits.shape[1]} "
                      f"U={logits.shape[2]-1} alloc={peak:.2f} "
                      f"resv={resv:.2f}GiB "
                      f"dropped={dropped} "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
                run_loss, run_n = 0.0, 0
            if fixed and (step % args.eval_steps == 0 or step == max_steps):
                import evaluate
                wer = evaluate.load("wer")
                hyps, refs = [], []
                for bb in fixed:
                    hyps += greedy_decode(model, processor, bb, args.device)
                    rl = bb["labels"].clone()
                    rl = rl.masked_fill(rl == -100, blank)
                    refs += [clean_special_tokens(t) for t in
                             processor.tokenizer.batch_decode(
                                 rl, group_tokens=False)]
                print(f"[{step}] OVERFIT WER="
                      f"{wer.compute(predictions=hyps, references=refs):.5f}",
                      flush=True)
                print(f"   REF: {refs[0][:70]!r}", flush=True)
                print(f"   HYP: {hyps[0][:70]!r}", flush=True)
            elif step % args.eval_steps == 0 or step == max_steps:
                import evaluate
                wer = evaluate.load("wer")
                out = []
                for name, ds in dev_sets.items():
                    hyps, refs = [], []
                    cap = (args.final_eval_examples if step == max_steps
                           else args.eval_examples)
                    cap = cap or 10 ** 9
                    for db_batch in DataLoader(ds, batch_size=32,
                                               collate_fn=collator):
                        if len(refs) >= cap:
                            break
                        hyps += greedy_decode(model, processor, db_batch,
                                              args.device)
                        rl = db_batch["labels"].clone()
                        rl = rl.masked_fill(rl == -100, blank)
                        refs += [clean_special_tokens(t) for t in
                                 processor.tokenizer.batch_decode(
                                     rl, group_tokens=False)]
                    n = min(len(hyps), len(refs), cap)
                    w = wer.compute(predictions=hyps[:n], references=refs[:n])
                    out.append(f"{name}={w:.5f}(n={n})")
                    if name == "dev-clean":
                        example = hyps[0][:55]
                print(f"[{step}] " + "  ".join(out) +
                      f"  e.g. {example!r}", flush=True)
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if args.device == "cuda" else 0.0)
    resv = (torch.cuda.max_memory_reserved() / 2**30
            if args.device == "cuda" else 0.0)
    print(f"[done] steps={step} alloc={peak:.2f}GiB reserved={resv:.2f}GiB "
          f"dropped={dropped} wall={(time.time()-t0)/60:.1f}min", flush=True)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args),
                    "peak_gib": peak, "steps": step},
                   os.path.join(args.output_dir, "rnnt.pt"))
        print(f"saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
