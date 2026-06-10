#!/usr/bin/env python3
"""Freq decomposition: global count vs diversity vs concentration (3 seeds)."""
import json, glob, time, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED=20260603; OUT=Path("technique_transfer/results"); OUT.mkdir(parents=True,exist_ok=True)
RDLogger.DisableLog("rdApp.*")
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}",flush=True)

log("Loading...")
manifest={}
mp=Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
with mp.open(encoding="utf-8") as f:
    for l in f:
        if l.strip(): q=json.loads(l); manifest[q["query_id"]]=q
def mfp(s):
    m=Chem.MolFromSmiles(s)
    if m is None:return None
    f=np.zeros(2048,dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,2048),f);return f
def mprops(s):
    m=Chem.MolFromSmiles(s)
    if m is None:return None
    return np.array([m.GetNumHeavyAtoms(),Descriptors.RingCount(m),Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.TPSA(m)],dtype=np.float32)
all_ofs=sorted({q["old_fragment_smiles"] for q in manifest.values()})
of_fps,of_props={},{}
for s in all_ofs:
    fp_=mfp(s);pr_=mprops(s)
    if fp_ is not None and pr_ is not None:of_fps[s]=fp_;of_props[s]=pr_
qlabels=defaultdict(dict)
for split in ["train","test"]:
    pattern=f"plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/matrices/{split}/{split}_features_shard_0*.jsonl"
    for shard in sorted(glob.glob(pattern)):
        with open(shard,encoding="utf-8") as f:
            for l in f:
                if not l.strip():continue
                r=json.loads(l)
                if r["query_id"] in manifest:qlabels[r["query_id"]][r["candidate"]]=int(r["label"])
cand_fps,cand_props={},{}
for labels in qlabels.values():
    for c in labels:
        if c in cand_fps:continue
        fp_=mfp(c);pr_=mprops(c)
        if fp_ is not None and pr_ is not None:cand_fps[c]=fp_;cand_props[c]=pr_
cand_list=sorted(cand_fps.keys())
of_queries=defaultdict(list)
for q in manifest.values():
    if q["old_fragment_smiles"] in of_fps:of_queries[q["old_fragment_smiles"]].append(q["query_id"])
labeled_ofs=sorted({manifest[qid]["old_fragment_smiles"] for qid,labels in qlabels.items()
    if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})
log(f"  {len(labeled_ofs)} OFs")

# Precompute 7 static features
of_data={}
for of_s in labeled_ofs:
    ofp=of_fps[of_s];op=of_props[of_s];ofp_n=ofp/(ofp.sum()+1e-10)
    Xl,yl,qinfo=[],[],[]
    for qid in of_queries.get(of_s,[]):
        labels=qlabels.get(qid,{})
        if not labels:continue
        cands=list(labels.keys());nc=len(cands)
        feats=np.zeros((nc,7),dtype=np.float32);cil=[]
        for idx,c in enumerate(cands):
            cfp=cand_fps.get(c);cp=cand_props.get(c)
            if cfp is None:continue
            inter=float(cfp.dot(ofp));denom=float(cfp.sum()+ofp.sum()-inter)
            m=inter/max(denom,0.001)
            cfp_n=cfp/(cfp.sum()+1e-10);bc=0.0
            if cfp.sum()>0 and ofp.sum()>0:
                corr=float(np.corrcoef(cfp_n,ofp_n)[0,1])
                bc=corr if np.isfinite(corr) else 0.0
            feats[idx]=[m,bc,abs(cp[0]-op[0]),abs(cp[1]-op[1]),abs(cp[2]-op[2])/max(op[2],1),abs(cp[3]-op[3])/max(abs(op[3])+1,1),abs(cp[4]-op[4])/max(op[4]+1,1)]
            cil.append(cand_list.index(c) if c in cand_list else -1)
        Xl.append(feats);yl.append(np.array([labels[c] for c in cands],dtype=np.int8));qinfo.append((qid,nc,cil))
    if Xl:of_data[of_s]={"X7":np.vstack(Xl),"y":np.concatenate(yl),"queries":qinfo}
log(f"  {sum(len(v['X7']) for v in of_data.values()):,} pairs")

def hit10(ld,s,c):
    o=np.argsort(-s);r=np.array([ld[c[i]] for i in o]);p=np.where(r==1)[0]
    return int(len(p)>0 and p[0]<10)

# 3-seed comparison
log("=== FREQ DECOMPOSITION (3 seeds) ===")
results=[]
for si in range(3):
    rng=np.random.RandomState(SEED+si);sh=list(labeled_ofs);rng.shuffle(sh)
    n_tr=int(len(sh)*0.7);tr_ofs,te_ofs=sh[:n_tr],sh[n_tr:]
    n_of=len(tr_ofs)

    # Freq maps
    freq_raw={};cand_of_set=defaultdict(set)
    for o in tr_ofs:
        y=of_data[o]["y"];queries=of_data[o]["queries"];off=0
        for _,nc,cil in queries:
            for j in range(nc):
                if y[off+j]==1:
                    ci=cil[j]
                    if ci>=0:
                        c=cand_list[ci];freq_raw[c]=freq_raw.get(c,0)+1;cand_of_set[c].add(o)
            off+=nc
    max_raw=max(freq_raw.values()) if freq_raw else 1
    freq_div={c:len(s) for c,s in cand_of_set.items()}

    def freq_base(c):return freq_raw.get(c,0)/max(max_raw,1)
    def freq_log(c):return np.log(1+freq_raw.get(c,0))/np.log(1+max_raw)
    def freq_div_norm(c):return freq_div.get(c,1)/(n_of+1)
    def freq_conc(c):return freq_raw.get(c,0)/max(freq_div.get(c,1),1)/max_raw

    # Build training matrices
    X7_tr=np.vstack([of_data[o]["X7"] for o in tr_ofs]);y_tr=np.concatenate([of_data[o]["y"] for o in tr_ofs])
    n_rows=len(y_tr)
    tr_cands=[]
    off=0
    for o in tr_ofs:
        for _,nc,cil in of_data[o]["queries"]:
            for j in range(nc):
                ci=cil[j];tr_cands.append(cand_list[ci] if ci>=0 else "")
            off+=nc

    def build_X(n_base,n_extra):
        """n_base=7 (original 7 static), n_extra=number of freq cols."""
        cols=7+n_extra;X=np.zeros((n_rows,cols),dtype=np.float32);X[:,:7]=X7_tr
        return X

    # Baseline: 1 freq col
    Xb=build_X(7,1)
    for j,c in enumerate(tr_cands):Xb[j,7]=freq_base(c)
    # Decomposed: 3 freq cols (raw, log, div, conc — pick best combo)
    Xd=build_X(7,4)
    for j,c in enumerate(tr_cands):
        Xd[j,7]=freq_base(c);Xd[j,8]=freq_log(c);Xd[j,9]=freq_div_norm(c);Xd[j,10]=freq_conc(c)

    sc_b=StandardScaler();sc_d=StandardScaler()
    lr_b=LogisticRegression(C=1.0,max_iter=2000,random_state=SEED);lr_b.fit(sc_b.fit_transform(Xb),y_tr)
    lr_d=LogisticRegression(C=1.0,max_iter=2000,random_state=SEED);lr_d.fit(sc_d.fit_transform(Xd),y_tr)

    sh10={"base":[],"decomp":[]}
    for o in te_ofs:
        queries=of_data[o]["queries"];yt=of_data[o]["y"];X7=of_data[o]["X7"];off=0
        for _,nc,cil in queries:
            cands=[cand_list[ci] if ci>=0 else "" for ci in cil]
            ld=dict(zip(cands,yt[off:off+nc].astype(int)))
            if sum(ld.values())==0:off+=nc;continue
            Xqb=np.zeros((nc,8),dtype=np.float32);Xqb[:,:7]=X7[off:off+nc]
            Xqd=np.zeros((nc,11),dtype=np.float32);Xqd[:,:7]=X7[off:off+nc]
            Xqb[:,7]=[freq_base(c) for c in cands]
            Xqd[:,7]=[freq_base(c) for c in cands];Xqd[:,8]=[freq_log(c) for c in cands]
            Xqd[:,9]=[freq_div_norm(c) for c in cands];Xqd[:,10]=[freq_conc(c) for c in cands]
            sh10["base"].append(hit10(ld,lr_b.predict_proba(sc_b.transform(Xqb))[:,1],cands))
            sh10["decomp"].append(hit10(ld,lr_d.predict_proba(sc_d.transform(Xqd))[:,1],cands))
            off+=nc

    bm=float(np.mean(sh10["base"]));dm=float(np.mean(sh10["decomp"]))
    results.append({"seed":si,"base":bm,"decomp":dm,"delta":dm-bm})
    log(f"  S{si}: base={bm:.4f} decomp={dm:.4f} delta={dm-bm:+.4f}")

df=pd.DataFrame(results)
print(f"\n=== FREQ DECOMPOSITION ===")
print(df.to_string(index=False))
print(f"base={df['base'].mean():.4f} decomp={df['decomp'].mean():.4f} delta={df['delta'].mean():+.4f}")
df.to_csv(OUT/"freq_decomposition.csv",index=False)
log("Done")
