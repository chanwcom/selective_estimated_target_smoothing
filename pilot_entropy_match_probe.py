#!/usr/bin/env python3
"""Measures what alpha entropy matching WOULD pick, without using it.

This is an "observe-only" probe: it runs the model forward, builds the
CTC alignment posterior exactly as ShcLoss does, asks
`apply_entropy_matched_smoothing` for the alpha it would solve for, and
reports the distribution. Nothing is trained and no loss is consumed, so
running this cannot affect any experiment in flight.

WHY. Entropy matching removes the smoothing hyperparameter by solving
for it. That is only an improvement if the value it solves for is
sensible. The tuned optimum for fixed-alpha SETS is 0.02 at 1hr and 0.01
at 10hr; the concern is that the entropy-matching equation lands nearer
0.15-0.25, an order of magnitude of over-smoothing. Whether a clamp
(--entropy_match_alpha_max) or partial repayment (--entropy_match_kappa)
is needed depends entirely on that number, and a queued sweep that
guesses wrong wastes GPU-hours. This answers it in minutes.

WHAT TO LOOK AT.
  * alpha percentiles. Near 0.02 means the rule works unclamped and the
    "no hyperparameter" claim survives intact. Near 0.2 means a ceiling
    is required. Pinned at 1.0 means case B dominates (see below).
  * case_b rate. Fraction of examples where the acoustic posterior's
    entropy exceeds anything the mixture can reach (its ceiling is
    mean_t log N), so alpha clamps at the maximum. Expected to be high
    early in training, when the model is near-uniform over the whole
    vocabulary while the active set stays small. If it is high LATE in
    training too, `eps` is the lever: it sets N, hence the ceiling.
  * h_prime_1. Should be ~0. It is the derivative of entropy with
    respect to alpha at alpha=1, which the monotonicity proof says
    vanishes for a uniform mixing distribution. Non-zero here would mean
    the class-space assumption is violated somewhere and bisection is
    not safe.
  * intervention. mean L1 distance between the smoothed and original
    target. This, not alpha, is what training actually feels: a large
    alpha over a target that is already near-uniform on its active set
    barely changes anything.

Usage:
    # Late training: a finished checkpoint from the fixed-alpha sweep.
    python pilot_entropy_match_probe.py \\
        --checkpoint /mnt/data/home/chanwcom/models/<run>/checkpoint-2000

    # Early training: the pretrained encoder with a fresh CTC head, i.e.
    # roughly what step 0 looks like.
    python pilot_entropy_match_probe.py --pretrained
"""

# pylint: disable=import-error, no-member

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import argparse
import os

# Third-party imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC, AutoProcessor

# Custom imports
from common import sample_util
from cwk.loss.pytorch import seq_loss_util, shc_loss, shc_loss_util
from wav2vec2_finetuning_sets import (DataCollatorCTCWithPadding,
                                      Wav2Vec2SPMTokenizer)

_DEFAULT_TRAIN_DIR = "/mnt/data/database/libri_light/1h"
_DEFAULT_RESOURCE_DIR = ("/mnt/data/home/chanwcom/local_repository/"
                         "cognitive_workflow_kit_emnlp_2026/resources/spm")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a trained checkpoint directory to probe.")
    source.add_argument(
        "--pretrained", action="store_true", default=False,
        help="Probe facebook/wav2vec2-base with a fresh CTC head instead, "
             "approximating the start of training.")
    parser.add_argument("--train_top_dir", type=str,
                        default=_DEFAULT_TRAIN_DIR)
    parser.add_argument("--resource_top_dir", type=str,
                        default=_DEFAULT_RESOURCE_DIR)
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument(
        "--num_batches", type=int, default=20,
        help="How many batches to probe.")
    parser.add_argument(
        "--max_batch_audio_len", type=int, default=800000,
        help="Deliberately far below the 6400000 used for training: this "
             "runs forward-only and should stay small enough to coexist "
             "with whatever is already occupying the GPU.")
    parser.add_argument("--max_sample_audio_len", type=int, default=480000)
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Defaults to cpu so that probing cannot disturb a training "
             "run holding the GPU. Forward-only over a few batches is "
             "cheap enough that this is usually fine.")
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


@torch.no_grad()
def alignment_posterior_in_class_space(model, batch, device):
    """Reproduces ShcLoss's gamma, scattered into class space.

    Returns (ground_truth_prob, acoustic_probs, logits_len), all built
    the same way ShcLoss.forward builds them, so the probe measures the
    quantity training would actually see.
    """
    input_values = batch["input_values"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_values=input_values,
                    attention_mask=attention_mask)
    logits = outputs["logits"]
    logits_len = model._get_feat_extract_output_lengths(
        attention_mask.sum(-1))
    target_lens = (labels >= 0).to(torch.int32).sum(dim=1)

    augmented = seq_loss_util.to_blank_augmented_labels(
        {"SEQ_DATA": labels, "SEQ_LEN": target_lens}, 0, False, False)
    aug_labels, aug_lens = augmented["SEQ_DATA"], augmented["SEQ_LEN"]
    clamped = torch.clamp(aug_labels, min=0)

    log_probs = torch.log_softmax(logits, dim=-1)
    trans = seq_loss_util.label_trans_allowance_table_ctc(
        aug_labels, aug_lens)
    log_label_probs = seq_loss_util.calculate_log_label_prob(
        clamped, log_probs)
    log_alpha, log_beta, _ = shc_loss.calculate_alpha_beta(
        trans, log_label_probs, aug_lens, logits_len)

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma, axis=2, keepdim=True)
    ground_truth_prob = shc_loss._scatter_to_class_space(
        torch.exp(log_gamma), log_probs, clamped)

    return ground_truth_prob, log_probs.exp(), logits_len


def _percentiles(values):
    q = torch.tensor([0.10, 0.50, 0.90])
    return torch.quantile(values, q)


def main():
    args = parse_args()
    device = torch.device(args.device)

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    spm_model_path = os.path.join(
        args.resource_top_dir,
        f"librispeech_unigram_{args.vocab_size}.model")
    processor.tokenizer = Wav2Vec2SPMTokenizer(spm_model_path)
    collator = DataCollatorCTCWithPadding(processor=processor,
                                          padding="longest")

    dataset = sample_util.make_dataset(
        args.train_top_dir, True, spm_model_path,
        dynamic_batch=sample_util.DynamicBatchConfig(
            collate_fn=collator,
            max_batch_length=args.max_batch_audio_len,
            max_batch_size=None,
            window_mult=1.0,
            seed=0),
        max_sample_length=args.max_sample_audio_len)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    source = ("facebook/wav2vec2-base" if args.pretrained
              else args.checkpoint)
    model = AutoModelForCTC.from_pretrained(
        source, vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True)
    model.to(device).eval()
    print(f"probing: {source}\ndevice : {device}\n")

    collected = {k: [] for k in
                 ("alpha", "h_lo", "h_hi", "h_target", "case_b", "case_c",
                  "n_active", "h_prime_1")}
    interventions = []

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        target, acoustic, logits_len = alignment_posterior_in_class_space(
            model, batch, device)
        smoothed, stats = (
            shc_loss_util.apply_entropy_matched_smoothing(
                target, acoustic, logits_len, eps=args.eps,
                return_stats=True))
        for key in collected:
            collected[key].append(stats[key].detach().cpu())

        # What training actually feels: L1 distance per valid frame.
        max_time = target.shape[1]
        time_idx = torch.arange(max_time, device=target.device)
        valid = time_idx.unsqueeze(0) < logits_len.unsqueeze(1)
        delta = (smoothed - target).abs().sum(dim=-1)[valid]
        interventions.append(delta.detach().cpu())
        print(f"  batch {i + 1}/{args.num_batches} "
              f"({target.shape[0]} examples)", end="\r")

    merged = {k: torch.cat(v).float() for k, v in collected.items()}
    n = merged["alpha"].numel()
    print(f"\n\n{'=' * 62}\n{n} examples probed\n{'=' * 62}")

    a10, a50, a90 = _percentiles(merged["alpha"])
    print(f"alpha            p10={a10:.4f}  p50={a50:.4f}  p90={a90:.4f}"
          f"   (max={merged['alpha'].max():.4f})")
    print(f"  vs tuned optimum: 0.02 (1hr) / 0.01 (10hr)")
    print(f"case_b (ceiling) {merged['case_b'].mean() * 100:5.1f}%   "
          f"case_c (no-op) {merged['case_c'].mean() * 100:5.1f}%")
    print(f"entropy (nats)   H(q~)={merged['h_lo'].mean():.3f}  "
          f"target=H(p)={merged['h_target'].mean():.3f}  "
          f"ceiling=log N={merged['h_hi'].mean():.3f}")
    print(f"active classes N mean={merged['n_active'].mean():.2f}")

    delta = torch.cat(interventions).float()
    d10, d50, d90 = _percentiles(delta)
    print(f"intervention L1  p10={d10:.4f}  p50={d50:.4f}  p90={d90:.4f}")

    hp1 = merged["h_prime_1"].abs().max()
    verdict = "OK (monotone, bisection valid)" if hp1 < 1e-4 else "PROBLEM"
    print(f"max |h'(1)|      {hp1:.2e}   {verdict}")


if __name__ == "__main__":
    main()
