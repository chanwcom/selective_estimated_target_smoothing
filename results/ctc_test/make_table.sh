#!/bin/bash
# results.jsonl -> table.md. 아무 때나 돌려도 되고, 도중이어도 있는 만큼만 그린다.
R=/mnt/synology_nas_00/chanwcom/results/ctc_test
python3 - "$R" > $R/table.md <<'PY'
import json,sys,math,collections,datetime,os
R=sys.argv[1]; f=f"{R}/results.jsonl"
rows={}
for l in open(f) if os.path.exists(f) else []:
    try: d=json.loads(l)
    except: continue
    rows[(d["tag"],d["split"])]=d          # 나중 줄이 이김
byt=collections.defaultdict(dict)
for (t,s),d in rows.items(): byt[t][s]=d["wer"]
import re
G=collections.defaultdict(list)
for t,sp in byt.items():
    m=re.match(r"ctc_(\w+?)_(baseline|LS|FAS|AWS)_a([0-9p.]+)_s(\d)$", t)
    if not m: continue
    if not {"test-clean","test-other"} <= set(sp): continue
    G[(m[1],m[2],float(m[3].replace("p",".")))].append(
        (sp["test-clean"], sp["test-other"], int(m[4])))
print(f"# CTC test 결과  ({datetime.datetime.now():%Y-%m-%d %H:%M} 기준)\n")
print("test-clean n=2620, test-other n=2939. CPU 디코딩. geo = sqrt(clean*other), 낮을수록 좋음.\n")
g=lambda c,o: math.sqrt(c*o)
for prof in sorted({k[0] for k in G}):
    ks=[k for k in G if k[0]==prof]
    base=G.get((prof,"baseline",0.0))
    bm=sum(g(c,o) for c,o,_ in base)/len(base) if base else None
    print(f"\n## {prof}\n")
    print("| 방법 | alpha | 시드 | test-clean | test-other | geo | Δ |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for k in sorted(ks, key=lambda x:(x[1]!="baseline", x[1], x[2])):
        L=G[k]; n=len(L)
        c=sum(x[0] for x in L)/n; o=sum(x[1] for x in L)/n
        gm=sum(g(x[0],x[1]) for x in L)/n
        d=f"{(gm/bm-1)*100:+.2f}%" if bm else "—"
        a="—" if k[1]=="baseline" else f"{k[2]:g}"
        print(f"| {k[1]} | {a} | {n} | {c:.5f} | {o:.5f} | {gm:.5f} | {d} |")
tot=len(byt); done=sum(1 for v in byt.values() if len(v)==2)
print(f"\n---\n체크포인트 {done} 완료 / {tot} 착수")
PY
echo "wrote $R/table.md"
