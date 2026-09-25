#!/usr/bin/env python3
"""Rebuild CHECKPOINTS.md -- where every checkpoint for the paper lives.

Answers one question: for a given (task, scale, method, alpha), which files
on disk hold the weights? The method is NOT taken from a directory name or
a tag; it is resolved from the evidence each task actually carries.

  CTC    : only the run-name suffix survives (training_args.bin has no
           alpha/alpha_mode/fas_eps), so the suffix is the evidence.
  RNN-T  : rnnt.pt stores vars(args), so alpha_mode/fas_eps/beta are read
           straight out of the checkpoint and the name is ignored.
"""
import os, re, glob, json, collections, datetime

NAS   = "/mnt/synology_nas_00/chanwcom"
M     = f"{NAS}/models"
SCRAT = ("/tmp/claude-2000/-mnt-synology-nas-00-chanwcom-local-repository-"
         "selective-estimated-target-smoothing/"
         "3d975a45-dbb0-4ea6-b6e2-2f6f96a23c5a/scratchpad")

def scale_of(s):
    if "100hr" in s: return "100h"
    if "10hr"  in s: return "10h"
    if "1hr"   in s: return "1h"
    return "?"

# ---------------------------------------------------------------- CTC
def ctc_method(name, alpha, beta):
    """방법은 접미사 + beta 로만 판정한다. 애매하면 애매하다고 적는다."""
    if alpha == 0.0: return "baseline"
    if "_aws_eps1eneg10" in name: return "AWS"
    if "_mos_eps"  in name: return "MOS"
    if "_asap_eps" in name: return "ASAP"
    if "_actsup"   in name: return "AS"
    if "_shmatch"  in name: return "C-SETS-SH"
    if "_abs"      in name: return "ABS"
    if "_label_only" in name or "_blank_only" in name: return "blank-gate"
    # peak-thresholded FAS -- eps 가 아니라 thr 로 활성집합을 정한다. LS 아님.
    if "_classspace_fas_thr" in name: return "FAS-thr"
    if "_classspace_fas_eps" in name:
        if beta == 1.0: return "LS"          # textbook LS 와 bit-exact
        if beta == 0.0: return "FAS"
        return f"FAS(beta={beta:g})"
    # 맨 _hmatch: aws/mos/abs/entropy_matched 중 무엇인지 사후 판별 불가
    if re.search(r"_hmatch$", name) and "_eps" not in name: return "판별불가"
    if "_classspace" in name:
        # class space 의 고정 alpha 스무딩. beta 가 0 이 아니면 SETS 변형이다.
        return "LS" if beta == 0.0 else f"SETS(beta={beta:g})"
    return "기타"

def scan_ctc():
    rows = []
    for host in sorted(os.listdir(M)):
        hp = os.path.join(M, host)
        if not os.path.isdir(hp) or os.path.islink(hp) or "PREBUGFIX" in host:
            continue
        for d in sorted(glob.glob(hp + "/libri_*_shc_*")):
            n = os.path.basename(d)
            a = re.search(r"alpha_([0-9p]+)", n)
            b = re.search(r"beta_([0-9p]+)",  n)
            s = re.search(r"seed(\d)",        n)
            st= re.search(r"_(\d+)steps",     n)
            if not (a and b and s): continue
            alpha = float(a.group(1).replace("p", "."))
            beta  = float(b.group(1).replace("p", "."))
            cks   = sorted(glob.glob(d + "/checkpoint-*"))
            final = None
            for ck in cks:
                p = os.path.join(ck, "trainer_state.json")
                if not os.path.exists(p): continue
                try: js = json.load(open(p))
                except Exception: continue
                if js.get("global_step") and js["global_step"] >= (js.get("max_steps") or 0):
                    final = ck
            if final is None and cks:
                # save_pretrained 최종본: trainer_state 가 없고 model.safetensors 만
                for ck in cks:
                    if (os.path.exists(os.path.join(ck, "model.safetensors"))
                            and not os.path.exists(os.path.join(ck, "optimizer.pt"))):
                        final = ck
            rows.append(dict(task="CTC", host=host, scale=scale_of(n),
                             steps=int(st.group(1)) if st else None,
                             method=ctc_method(n, alpha, beta), alpha=alpha,
                             beta=beta, seed=int(s.group(1)), dir=d,
                             final=final, complete=final is not None))
    return rows

# -------------------------------------------------------------- RNN-T
def rnnt_method(a):
    mode, beta = a.get("alpha_mode"), a.get("beta")
    if not a.get("alpha"): return "baseline"
    if mode == "aws": return "AWS" if a.get("fas_eps") == 1e-10 else f"AWS(eps={a.get('fas_eps')})"
    if mode == "floored_active_support": return "LS" if beta == 1.0 else "FAS"
    if mode == "fixed": return "baseline"
    return mode

def scan_rnnt():
    import torch
    rows = []
    for p in sorted(set(glob.glob(f"{M}/**/rnnt.pt", recursive=True))
                    | set(glob.glob(f"{SCRAT}/**/rnnt.pt", recursive=True))):
        try: a = torch.load(p, map_location="cpu", weights_only=False, mmap=True).get("args")
        except Exception: a = None
        if not a: continue
        prof = str(a.get("finetune_profile") or "")
        rows.append(dict(task="RNN-T",
                         host=("NAS/" + p.split("/models/")[-1].split("/")[0]
                               if "/models/" in p else "scratchpad"),
                         scale=scale_of(prof + " " + p), steps=a.get("max_steps"),
                         method=rnnt_method(a), alpha=a.get("alpha"),
                         beta=a.get("beta"), seed=a.get("seed"),
                         dir=os.path.dirname(p), final=p, complete=True))
    return rows

rows = scan_ctc() + scan_rnnt()
now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")

md = [f"""# 체크포인트가 어디 있나

**이 파일은 자동 생성됩니다. 손으로 고치지 마십시오.**

    python /mnt/synology_nas_00/chanwcom/results/make_checkpoints_md.py

마지막 생성: {now}   ·   결과 수치는 [RESULTS.md](RESULTS.md)   ·   측정 규약은 [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)

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
"""]

order = {"baseline":0, "LS":1, "FAS":2, "AWS":3}
for task in ("CTC", "RNN-T"):
    for scale in ("1h", "10h", "100h"):
        sub = [r for r in rows if r["task"] == task and r["scale"] == scale]
        if not sub: continue
        md.append(f"\n## {task} — {scale}\n")
        grp = collections.defaultdict(list)
        for r in sub: grp[(r["method"], r["steps"], r["alpha"], r["beta"])].append(r)
        md.append("| 방법 | step | α | β | seeds | 완주 | 위치 |")
        md.append("|---|---:|---:|---:|---|---:|---|")
        for k in sorted(grp, key=lambda x: (order.get(x[0], 9), str(x[0]),
                                            x[1] or 0, x[2] or 0)):
            v = grp[k]
            seeds = sorted({r["seed"] for r in v if r["seed"] is not None})
            done  = sum(1 for r in v if r["complete"])
            hosts = sorted({r["host"] for r in v})
            ex    = v[0]["dir"].replace(M + "/", "").replace(SCRAT + "/", "scratchpad/")
            ex    = re.sub(r"seed\d", "seed*", ex)
            ex    = re.sub(r"_s\d($|/)", r"_s*\1", ex)
            md.append(f"| {k[0]} | {k[1] or '?'} | {k[2] if k[2] is not None else '?'} | "
                      f"{k[3] if k[3] is not None else '?'} | {seeds} | {done}/{len(v)} | `{ex}` |")
        md.append(f"\n호스트: {', '.join(sorted({r['host'] for r in sub}))}")

open(f"{NAS}/results/CHECKPOINTS.md", "w").write("\n".join(md) + "\n")
with open(f"{NAS}/results/checkpoints.jsonl", "w") as f:
    for r in sorted(rows, key=lambda r: (r["task"], r["scale"], str(r["method"]),
                                         r["alpha"] or 0, r["seed"] or 0)):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"CTC {sum(1 for r in rows if r['task']=='CTC')}  "
      f"RNN-T {sum(1 for r in rows if r['task']=='RNN-T')}  -> CHECKPOINTS.md")
