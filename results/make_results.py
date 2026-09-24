#!/usr/bin/env python3
"""Rebuild RESULTS.md -- the single trusted table.

Admission rule, applied to every row:
  * produced by wav2vec2_inference.py (CTC) or rnnt/eval_ckpt.py (RNN-T),
    never scraped from a training log;
  * greedy decoding (--decoder pipeline for CTC);
  * split is test-clean or test-other, whole split;
  * the method is resolved from the CHECKPOINT DIRECTORY, not from a tag.

Anything failing one of those is left out and counted in "제외".

Why the directory and not the tag: watch_decode_cpu.sh labels by beta alone
(`beta==1.0 -> LS else FAS`), so every AWS run landed in decode.jsonl tagged
`FAS_`. The suffix is the only reliable discriminator.
"""
import json, os, re, glob, math, statistics, collections, datetime

NAS   = "/mnt/synology_nas_00/chanwcom"
MODELS= f"{NAS}/models"
REPO  = f"{NAS}/local_repository/selective_estimated_target_smoothing"
OUT   = f"{NAS}/results"

# ---------------------------------------------------------------- method
def method_of(dirname, alpha, beta):
    """(method, note) from the run-name suffix. alpha==0 is baseline."""
    if alpha == 0.0:
        return "baseline", None
    if dirname.endswith("_aws_eps1eneg10_hmatch") or "_aws_eps1eneg10" in dirname:
        return "AWS", None
    if "_classspace_fas_eps1eneg10" in dirname:
        # FAS at beta=1.0 is bit-exact textbook LS (see
        # apply_floored_active_support_smoothing docstring).
        return ("LS" if beta == 1.0 else "FAS"), None
    if "_classspace" in dirname:
        return "LS", None
    return None, f"미분류 접미사: {dirname}"

def parse_run(nm):
    a = re.search(r'alpha_([0-9p]+)', nm)
    b = re.search(r'beta_([0-9p]+)',  nm)
    s = re.search(r'seed(\d)',        nm)
    if not (a and b and s): return None
    return (float(a.group(1).replace('p','.')),
            float(b.group(1).replace('p','.')), int(s.group(1)))

# ------------------------------------------------------- source A: 100 h
def load_ctc_100h():
    """decode.jsonl keys on a tag; rebuild tag -> dir exactly as the
    watcher did, then re-derive the method from the dir."""
    tag2dir = {}
    for host in os.listdir(MODELS):
        hp = os.path.join(MODELS, host)
        if not os.path.isdir(hp) or os.path.islink(hp): continue
        for ck in glob.glob(f"{hp}/*100hr_wsd_shc_15000steps*/checkpoint-15000"):
            nm = os.path.basename(os.path.dirname(ck))
            p = parse_run(nm)
            if not p: continue
            alpha, beta, seed = p
            watcher_m = "LS" if beta == 1.0 else "FAS"
            astr = re.search(r'alpha_([0-9p]+)', nm).group(1).replace('p','.')
            tag = f"{watcher_m}_a{astr}_seed{seed}_{host}"
            # Two different runs can produce the same watcher tag: the tag
            # carries only (beta-derived name, alpha, seed, host), so a FAS
            # run and an AWS run at the same alpha/seed/host collide. The
            # decode row cannot then be attributed, so drop both.
            tag2dir.setdefault(tag, []).append((nm, alpha, beta, seed))

    rows, skipped = [], []
    src = f"{REPO}/inference_logs_wsd/decode.jsonl"
    for line in open(src):
        r = json.loads(line)
        if r["decoder"] != "pipeline":
            skipped.append((r["tag"], r["split"], "beam_search")); continue
        if r["split"] not in ("test-clean", "test-other"):
            skipped.append((r["tag"], r["split"], "dev split")); continue
        if r["tag"] not in tag2dir:
            skipped.append((r["tag"], r["split"], "체크포인트 경로 못 찾음")); continue
        cands = tag2dir[r["tag"]]
        if len(cands) > 1:
            skipped.append((r["tag"], r["split"],
                            "태그 충돌 — 어느 체크포인트인지 특정 불가 ("
                            + " / ".join(sorted(c[0] for c in cands)) + ")"))
            continue
        nm, alpha, beta, seed = cands[0]
        m, note = method_of(nm, alpha, beta)
        if m is None:
            skipped.append((r["tag"], r["split"], note)); continue
        rows.append(dict(task="CTC", hours="100h", method=m, alpha=alpha,
                         seed=seed, split=r["split"], wer=r["wer"],
                         run=nm, src="inference_logs_wsd/decode.jsonl"))
    return rows, skipped

# --------------------------------------------------------- source B: 1 h
def load_ctc_1h():
    rows, skipped = [], []
    src = f"{OUT}/ctc_test/results.jsonl"
    if not os.path.exists(src): return rows, skipped
    for line in open(src):
        r = json.loads(line)
        if r["split"] not in ("test-clean", "test-other"):
            skipped.append((r["tag"], r["split"], "dev split")); continue
        nm = os.path.basename(os.path.dirname(r["ckpt"]))
        p = parse_run(nm)
        if not p:
            skipped.append((r["tag"], r["split"], "런 이름 파싱 실패")); continue
        alpha, beta, seed = p
        hours = "1h" if "1hr" in nm else ("10h" if "10hr" in nm else "100h")
        m, note = method_of(nm, alpha, beta)
        if m is None:
            skipped.append((r["tag"], r["split"], note)); continue
        rows.append(dict(task="CTC", hours=hours, method=m, alpha=alpha,
                         seed=seed, split=r["split"], wer=r["wer"],
                         run=nm, src="results/ctc_test/results.jsonl"))
    return rows, skipped

# ----------------------------------------------- source C: RNN-T test
def load_rnnt():
    """rnnt_test/results.jsonl -- rnnt/eval_ckpt.py 가 쓴 줄.

    RNN-T 는 방법을 이름이 아니라 rnnt.pt 안의 args 에서 읽는다. 그래서
    여기서도 alpha_mode/beta/fas_eps 를 그대로 신뢰한다. AWS 는 fas_eps 가
    1e-10 인 것만 받는다 (1e-2, 0.1 로 돌린 런이 실제로 섞여 있다).
    """
    rows, skipped = [], []
    src = f"{OUT}/rnnt_test/results.jsonl"
    if not os.path.exists(src): return rows, skipped
    for line in open(src):
        r = json.loads(line)
        if r.get("error") or r.get("rc") not in (0, None):
            skipped.append((r.get("tag",""), "", "eval 실패")); continue
        if "test-clean" not in r or "test-other" not in r:
            skipped.append((r.get("tag",""), "", "split 누락")); continue
        mode, beta, eps = r.get("method"), r.get("beta"), r.get("fas_eps")
        alpha = r.get("alpha") or 0.0
        if alpha == 0.0:
            m = "baseline"
        elif mode == "aws":
            if eps != 1e-10:
                skipped.append((r["tag"], "", f"AWS fas_eps={eps} (1e-10 아님)")); continue
            m = "AWS"
        elif mode == "floored_active_support":
            m = "LS" if beta == 1.0 else "FAS"
        else:
            skipped.append((r["tag"], "", f"논문 외 모드 {mode}")); continue
        for sp in ("test-clean", "test-other"):
            rows.append(dict(task="RNN-T", hours=r["hours"], method=m, alpha=alpha,
                             seed=r.get("seed"), split=sp, wer=r[sp],
                             run=r["tag"], ckpt=r.get("ckpt", ""),
                             src="results/rnnt_test/results.jsonl"))
    return rows, skipped


# ------------------------------------------------------------------ main
def cellify(rows):
    """(task,hours,method,alpha,seed) -> {split: wer}; keep only complete."""
    by = collections.defaultdict(dict)
    meta = {}
    for r in rows:
        k = (r["task"], r["hours"], r["method"], r["alpha"], r["seed"])
        by[k][r["split"]] = r["wer"]
        # run name plus the path the number was actually decoded from, so a
        # reader can go straight to the weights instead of guessing which
        # machine's copy produced the row.
        meta[k] = {"run": r["run"], "ckpt": r.get("ckpt", ""),
                   "src": r.get("src", "")}
    return ({k: v for k, v in by.items() if len(v) == 2},
            {k: v for k, v in by.items() if len(v) != 2}, meta)

def table(cells, task, hours):
    sub = {k: v for k, v in cells.items() if k[0] == task and k[1] == hours}
    if not sub: return None
    grp = collections.defaultdict(dict)
    for k, v in sub.items():
        grp[(k[2], k[3])][k[4]] = (v["test-clean"], v["test-other"],
                                   math.sqrt(v["test-clean"] * v["test-other"]))
    base = grp.get(("baseline", 0.0), {})
    bg = [x[2] for x in base.values()]
    bm = statistics.mean(bg) if bg else None
    bs = statistics.stdev(bg) if len(bg) > 1 else None
    L = []
    L.append(f"| 방법 | α | n | test-clean | test-other | 기하평균 | Δ vs baseline | t |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    order = {"baseline": 0, "LS": 1, "FAS": 2, "AWS": 3}
    for (m, a) in sorted(grp, key=lambda x: (order.get(x[0], 9), x[1])):
        d = grp[(m, a)]
        n = len(d)
        c = statistics.mean([x[0] for x in d.values()])
        o = statistics.mean([x[1] for x in d.values()])
        g = [x[2] for x in d.values()]
        gm = statistics.mean(g)
        if bm and m != "baseline" and n > 1 and bs:
            sd = statistics.stdev(g)
            se = math.sqrt(bs**2/len(bg) + sd**2/n)
            dl, t = f"{(gm-bm)/bm*100:+.2f}%", f"{(gm-bm)/se:.2f}"
        elif bm and m != "baseline":
            dl, t = f"{(gm-bm)/bm*100:+.2f}%", "—"
        else:
            dl, t = "—", "—"
        star = " ⚠" if n < 5 else ""
        L.append(f"| {m} | {a:g} | {n}{star} | {c:.5f} | {o:.5f} | **{gm:.5f}** | {dl} | {t} |")
    return "\n".join(L)

rows, skipped = [], []
for f in (load_ctc_100h, load_ctc_1h, load_rnnt):
    r, s = f(); rows += r; skipped += s
cells, partial, meta = cellify(rows)

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
md = [f"""# 검증된 test 결과 — 단일 출처

**이 파일만 믿으십시오.** 훈련 로그에서 긁은 값, dev 값, beam search 값은
들어오지 않습니다. 사람이 손으로 고치지 마십시오 — `make_results.py` 가
다시 씁니다.

    python /mnt/synology_nas_00/chanwcom/results/make_results.py

마지막 생성: {now}

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
"""]

for task in ("CTC", "RNN-T"):
    for hours in ("1h", "100h"):
        md.append(f"\n## {task} — {hours}\n")
        t = table(cells, task, hours)
        md.append(t if t else "_아직 없음._\n")

if partial:
    md.append("\n## 한쪽 split 만 있는 칸 (미완)\n")
    for k in sorted(partial, key=str):
        md.append(f"- `{k[0]} {k[1]} {k[2]} a={k[3]:g} s{k[4]}` — "
                  f"{', '.join(sorted(partial[k]))} 만 있음")

if skipped:
    c = collections.Counter(x[2] for x in skipped)
    md.append("\n## 제외된 줄\n")
    for reason, n in c.most_common():
        md.append(f"- {reason}: {n} 줄")

open(f"{OUT}/RESULTS.md", "w").write("\n".join(md) + "\n")
with open(f"{OUT}/verified.jsonl", "w") as fh:
    for k in sorted(cells, key=str):
        fh.write(json.dumps(dict(task=k[0], hours=k[1], method=k[2], alpha=k[3],
                                 seed=k[4], **meta[k], **cells[k]),
                            ensure_ascii=False) + "\n")
print(f"cells={len(cells)}  partial={len(partial)}  skipped={len(skipped)}")
