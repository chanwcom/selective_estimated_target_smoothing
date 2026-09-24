#!/bin/bash
# results.jsonl -> table.md. 아무 때나 돌려도 되고, 도중이어도 있는 만큼만 그린다.
R=/mnt/synology_nas_00/chanwcom/results/slimducer_test
python3 - "$R" > $R/table.md <<'PY'
import json,sys,math,collections,datetime,os,re
R=sys.argv[1]; f=f"{R}/results.jsonl"
rows={}
for l in (open(f) if os.path.exists(f) else []):
    try: d=json.loads(l)
    except: continue
    rows[d["tag"]]=d                      # 나중 줄이 이김
# dev 기준값 (--dev_limit 200). 비교용으로만 옆에 둔다.
DEV={"llm10h_qwen3_0.6b":0.14420,"llm10h_llama32_1b":0.14860,
     "llm10h_smollm2_360m":0.16580,"enc10h_whisper_small":0.18320,
     "enc10h_w2v2_base":0.97790,"enc10h_wavlm_large":0.16950,
     "llm100h_qwen3_0.6b":0.08513,"llm100h_llama32_1b":0.08078,
     "llm100h_qwen3_4b":0.08695,"llm100h_smollm2_360m":0.09265,
     "enc100h_wavlm_large":0.08166}
GROUP=[("10h -- LLM 교체 (인코더 AuT 고정)","llm10h_"),
       ("10h -- 인코더 교체 (LLM Qwen3-0.6B 고정)","enc10h_"),
       ("100h -- LLM 교체 (인코더 AuT 고정)","llm100h_"),
       ("100h -- 인코더 교체 (LLM Qwen3-0.6B 고정)","enc100h_")]
print(f"# SlimDucer test 결과  ({datetime.datetime.now():%Y-%m-%d %H:%M} 기준)\n")
print("test-clean n=2611, test-other n=2932 (전체). CPU fp32 디코딩.")
print("geo = sqrt(clean*other), 낮을수록 좋음.\n")
print("dev 열은 `--dev_limit 200` 으로 잰 옛 값이다. 200 발화짜리라 절대값도")
print("격차도 전체를 대표하지 않는다 -- 비교는 test 열로 한다.\n")
for title, pre in GROUP:
    ks=sorted([t for t in rows if t.startswith(pre)], key=lambda t: rows[t]["geo"])
    if not ks: continue
    # 그룹의 기준선: 10h/100h 모두 AuT + Qwen3-0.6B
    ref = "llm10h_qwen3_0.6b" if "10h" in pre and "100" not in pre else "llm100h_qwen3_0.6b"
    rg = rows.get(ref,{}).get("geo")
    print(f"\n## {title}\n")
    print("| 구성 | test-clean | test-other | geo | Δ vs Qwen3-0.6B | (dev@200) | test/dev |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for t in ks:
        d=rows[t]; dv=DEV.get(t)
        delta=f"{(d['geo']/rg-1)*100:+.2f}%" if rg else "—"
        dvs=f"{dv:.5f}" if dv else "—"
        td=f"{(d['geo']/dv-1)*100:+.1f}%" if dv else "—"
        nm=t[len(pre):]
        print(f"| {nm} | {d['test-clean_wer']:.5f} | {d['test-other_wer']:.5f} | "
              f"{d['geo']:.5f} | {delta} | {dvs} | {td} |")
print(f"\n---\n{len(rows)} 개 기록됨. 출처 machine: "
      + ", ".join(sorted({r.get('machine','?') for r in rows.values()})))
PY
echo "wrote $R/table.md"
