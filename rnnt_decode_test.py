"""Decode finished RNN-T checkpoints on test-clean/test-other, on CPU.

Why this exists. Everything the RNN-T sweep has produced so far is dev, and
it was measured inside the training loop on the GPU that was training. Two
problems with using that for a paper table:

  * it is dev, not test;
  * CPU and GPU decoding of the same checkpoint were measured to differ by
    0.74% relative, which is the same order as the effect being reported
    (AWS is -2 to -4% against its paired baseline). A table that mixes the
    two is measuring the device as much as the method.

`wav2vec2_inference.py` covers the CTC side. There is no RNN-T equivalent:
`wav2vec2_rnnt.py` only evaluates inline, and only on dev. This adds the
missing piece rather than touching that file, which is shared with another
machine's running work.

Nothing here reimplements decoding. `greedy_decode` is imported from the
training script, so the hypotheses are produced by the same code path that
produced the dev numbers; the checkpoint's own stored `args` rebuild the
model, so no hyperparameter is guessed.

    python rnnt_decode_test.py --ckpt_glob '<NAS>/rnnt100hr_aws/*/rnnt.pt' \
        --out rnnt_test_decode.jsonl

Resumable: a (tag, split) pair already in `--out` is skipped, so the script
can be stopped and restarted, and a sweep that is still producing
checkpoints can be re-run to pick up the new ones.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

# CPU only, and said three ways: the environment variable is what the
# encoder's CUDA init reads, `--device cpu` is what the model placement
# reads, and the training runs own every card on this box -- decoding must
# never take one.
if not any(_a.startswith("--device") for _a in sys.argv[1:]):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import repo_config
from common import sample_util
from wav2vec2_rnnt import (RnntModel, build_processor, clean_special_tokens,
                           greedy_decode)
from wav2vec2_finetuning_sets import DataCollatorCTCWithPadding


def load_done(path: str) -> set[tuple[str, str]]:
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["tag"], r["split"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_glob", required=True,
                    help="glob matching rnnt.pt files; the tag is the name "
                         "of the directory holding each one")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="test-clean,test-other")
    ap.add_argument("--device", default="cpu",
                    help="Where to decode. CPU and GPU decoding of the same "
                         "checkpoint were measured to differ by 0.74% "
                         "relative, the same order as the effect being "
                         "reported, so every cell in one table must use the "
                         "same value. CPU is the default because training "
                         "usually owns every card; a large encoder is slow "
                         "enough there that a full sweep needs cuda.")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="small on purpose. Greedy RNN-T decoding is a "
                         "Python loop over frames, so a wide batch buys "
                         "little and costs memory the training runs need.")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch intra-op threads. The box has 24 cores and "
                         "the training dataloaders hold most of them, so "
                         "this stays small by default.")
    ap.add_argument("--claim_dir", default="",
                    help="lets several copies of this script share one "
                         "--out. Each (tag, split) is claimed with mkdir, "
                         "which is atomic, before decoding starts -- the "
                         "--out file alone is not enough to divide the work "
                         "because a record is only written when a decode "
                         "finishes, so two lanes would spend an hour each "
                         "on the same checkpoint.")
    ap.add_argument("--limit", type=int, default=0,
                    help="decode only this many utterances per split. For "
                         "timing runs ONLY -- a capped number is not "
                         "comparable with a full one and is written with "
                         "n= so it cannot be mistaken for one.")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    splits = [s for s in args.splits.split(",") if s]
    done = load_done(args.out)
    ckpts = sorted(glob.glob(args.ckpt_glob))
    print(f"[plan] {len(ckpts)} checkpoints x {len(splits)} splits, "
          f"{len(done)} already done", flush=True)

    import evaluate
    wer_metric = evaluate.load("wer")
    db = repo_config.DB_TOP_DIR

    # The processor and the datasets depend only on the vocabulary size,
    # which is identical across this sweep, so they are built once and
    # reused. The model is rebuilt per checkpoint.
    processor = spm = collator = None
    datasets: dict[str, object] = {}

    for ck in ckpts:
        tag = os.path.basename(os.path.dirname(ck))
        if all((tag, s) in done for s in splits):
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        a = blob["args"]

        if processor is None:
            processor, spm = build_processor(a["vocab_size"])
            collator = DataCollatorCTCWithPadding(processor=processor,
                                                  padding="longest")
            for s in splits:
                # No max_sample_length. Training caps utterance length for
                # memory, but that is a training constraint, not part of
                # what "the test set" means -- and the CTC path
                # (wav2vec2_inference.py) does not cap, so passing the cap
                # here would evaluate CTC on 2620/2939 utterances and RNN-T
                # on 2611/2932 and put both in one table.
                datasets[s] = sample_util.make_dataset(
                    os.path.join(db, f"librispeech/webdataset/{s}"), True,
                    spm)

        vocab = len(processor.tokenizer)
        blank = processor.tokenizer.pad_token_id
        # The encoder name has to come from the checkpoint, not from
        # RnntModel's default. That default is facebook/wav2vec2-base,
        # which was the only encoder this script ever saw -- a large
        # checkpoint loaded into it fails on the first shape mismatch.
        # Older checkpoints predate --model_name and were all base.
        model = RnntModel(vocab, joint_dim=a["joint_dim"],
                          encoder_name=a.get("model_name",
                                             "facebook/wav2vec2-base"),
                          predictor_context=a["predictor_context"],
                          encoder_stride=a["encoder_stride"],
                          blank=blank, predictor=a["predictor"])
        model.load_state_dict(blob["model"])
        model.eval()
        model.to(args.device)
        del blob

        for s in splits:
            if (tag, s) in done:
                continue
            if args.claim_dir:
                os.makedirs(args.claim_dir, exist_ok=True)
                try:
                    os.mkdir(os.path.join(args.claim_dir, f"{tag}__{s}"))
                except FileExistsError:
                    continue
            t0 = time.time()
            hyps, refs = [], []
            for batch in DataLoader(datasets[s], batch_size=args.batch_size,
                                    collate_fn=collator):
                if args.limit and len(refs) >= args.limit:
                    break
                hyps += greedy_decode(model, processor, batch, args.device)
                rl = batch["labels"].clone()
                rl = rl.masked_fill(rl == -100, blank)
                refs += [clean_special_tokens(t) for t in
                         processor.tokenizer.batch_decode(rl,
                                                          group_tokens=False)]
            n = min(len(hyps), len(refs))
            if args.limit:
                n = min(n, args.limit)
            w = wer_metric.compute(predictions=hyps[:n], references=refs[:n])
            rec = {"tag": tag, "split": s, "decoder": "greedy",
                   "device": args.device, "wer": w, "n": n,
                   "minutes": (time.time() - t0) / 60}
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{tag}] {s} wer={w:.5f} n={n} "
                  f"({rec['minutes']:.1f} min)", flush=True)
            done.add((tag, s))
        del model

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
