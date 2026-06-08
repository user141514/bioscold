#!/usr/bin/env python3
"""Learned kernel multi-seed + ablation. Precomputed feature matrix for speed."""
import json, numpy as np, pandas as pd, glob, time
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

t0=time.time(); SEED=20260603
def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# Data load
manifest={}
with open('plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl',encoding='utf-8') as f:
    for l in f:
        if l.strip():
            q=json.loads(l)
            manifest[q['query_id']]=q
of_list=sorted(set(q['old_fragment_smiles'] for q in manifest.values()))
of_fps={}
of_props={}
for s in of_list:
    m=Chem.MolFromSmiles(s)
    if m:
        fp=np.zeros(2048,dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,2048),fp)
        of_props[s]=np.array([m.GetNumHeavyAtoms(),Descriptors.RingCount(m),
            Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.TPSA(m)],dtype=np.float32)
        of_fps[s]=fp
of_valid=[s for s in of_list if s in of_fps]

of_queries=defaultdict(list)
for q in manifest.values():
    if q['old_fragment_smiles'] in of_fps:
        of_queries[q['old_fragment_smiles']].append(q['query_id'])

query_labels=defaultdict(dict)
for split in ['train','test']:
    pat=f'plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/matrices/{split}/{split}_features_shard_0*.jsonl'
    for shard in sorted(glob.glob(pat))[:40]:
        with open(shard,encoding='utf-8') as f:
            for l in f:
                if not l.strip():
                    continue
                rec=json.loads(l)
                if rec['query_id'] in manifest:
                    query_labels[rec['query_id']][rec['candidate']]=int(rec['label'])

cand_fps={}
cand_props_dict={}
cand_freq=defaultdict(int)
for qid in query_labels:
    for cand,l in query_labels[qid].items():
        if l==1:
            cand_freq[cand]+=1
        if cand not in cand_fps:
            m=Chem.MolFromSmiles(cand)
            if m:
                fp=np.zeros(2048,dtype=np.float32)
                Chem.DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,2048),fp)
                cand_fps[cand]=fp
                cand_props_dict[cand]=np.array([m.GetNumHeavyAtoms(),Descriptors.RingCount(m),
                    Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.TPSA(m)],dtype=np.float32)
max_freq=max(cand_freq.values()) if cand_freq else 1

# Labeled OFs
of_has_labels=set()
for qid in query_labels:
    q=manifest.get(qid)
    if q and q['old_fragment_smiles'] in of_valid:
        if sum(1 for v in query_labels[qid].values() if v==1)>0:
            of_has_labels.add(q['old_fragment_smiles'])
labeled_ofs=sorted(of_has_labels)
log(f'Labeled OFs: {len(labeled_ofs)}')

# Precompute features per OF
log('Precomputing features...')
of_X={}; of_y={}; of_qids={}; of_clists={}
for of_s in labeled_ofs:
    Xl=[]; yl=[]; ql=[]; cl=[]
    for qid in of_queries.get(of_s,[]):
        labels=query_labels.get(qid,{})
        all_c=list(labels.keys()); nc=len(all_c)
        feats=np.zeros((nc,8),dtype=np.float32)
        ofp=of_fps[of_s]; op=of_props[of_s]; ofp_n=ofp/(ofp.sum()+1e-10)
        for j,c in enumerate(all_c):
            cfp=cand_fps.get(c); cp=cand_props_dict.get(c)
            if cfp is None or cp is None:
                feats[j]=[0,0,0,0,0,0,0,0]
                continue
            iv=cfp.dot(ofp); dv=cfp.sum()+ofp.sum()-iv; ms=iv/max(dv,0.001)
            cfp_n=cfp/(cfp.sum()+1e-10)
            bc=np.corrcoef(cfp_n,ofp_n)[0,1] if cfp.sum()>0 else 0.0
            feats[j]=[ms,bc,
                abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
                abs(cp[4]-op[4])/max(op[4]+1,1),
                cand_freq.get(c,0)/max_freq]
        Xl.append(feats); yl.append(np.array([labels[c] for c in all_c]))
        ql.append(qid); cl.append(all_c)
    if Xl:
        of_X[of_s]=np.vstack(Xl); of_y[of_s]=np.concatenate(yl)
        of_qids[of_s]=ql; of_clists[of_s]=cl
log(f'Precomputed {len(of_X)} OFs, {sum(len(v) for v in of_X.values()):,} pairs')

def ca_score(cand, of_s):
    if cand not in cand_fps:
        return 0.0
    ofp=of_fps[of_s]; cfp=cand_fps[cand]
    iv=cfp.dot(ofp); dv=cfp.sum()+ofp.sum()-iv; ms=iv/max(dv,0.001)
    op=of_props[of_s]; cp=cand_props_dict.get(cand)
    if cp is None:
        return ms
    dh=abs(cp[0]-op[0]); dr=abs(cp[1]-op[1])
    dm=abs(cp[2]-op[2])/max(op[2],1); dl=abs(cp[3]-op[3])/max(abs(op[3])+1,1)
    return 0.7*ms+0.3*(1.0/(1.0+dh+dr+dm+dl))

# Multi-seed
N_SEEDS=10; feat_names=['morgan','bit_corr','dHeavy','dRings','dMW','dLogP','dTPSA','freq']
seed_results=[]; ablation_done=False

for seed_i in range(N_SEEDS):
    rng=np.random.RandomState(SEED+seed_i)
    ofs_l=[s for s in labeled_ofs if s in of_X]; rng.shuffle(ofs_l)
    n_tr=int(len(ofs_l)*0.7); tr_ofs=ofs_l[:n_tr]; te_ofs=ofs_l[n_tr:]

    X_tr=np.vstack([of_X[s] for s in tr_ofs]); y_tr=np.concatenate([of_y[s] for s in tr_ofs])
    sc=StandardScaler(); X_tr_s=sc.fit_transform(X_tr)
    lr=LogisticRegression(C=1.0,max_iter=2000,random_state=SEED+seed_i)
    lr.fit(X_tr_s,y_tr)

    te_h10=[]
    for of_s in te_ofs:
        X_of=of_X[of_s]; y_of=of_y[of_s]; ql=of_qids[of_s]; cl=of_clists[of_s]
        X_of_s=sc.transform(X_of); lr_s=lr.predict_proba(X_of_s)[:,1]
        h10_lr=0; h10_ca=0; h10_f=0; nq=0; off=0
        for k in range(len(ql)):
            labels=dict(zip(cl[k],y_of[off:off+len(cl[k])].astype(int)))
            if sum(labels.values())==0:
                off+=len(cl[k]); continue
            nq+=1; nc=len(cl[k]); sl=lr_s[off:off+nc]
            sca=np.array([ca_score(c,of_s) for c in cl[k]])
            sf=np.array([cand_freq.get(c,0)/max_freq for c in cl[k]])
            def h10(scores):
                order=np.argsort(-scores); lab=np.array([labels[cl[k][j]] for j in range(nc)])[order]
                pi=np.where(lab==1)[0]; return int(len(pi)>0 and pi[0]<10)
            h10_lr+=h10(sl); h10_ca+=h10(sca); h10_f+=h10(sf); off+=nc
        if nq>0:
            te_h10.append((h10_f/nq, h10_ca/nq, h10_lr/nq))
    if te_h10:
        seed_results.append({'seed':seed_i,'n_of':len(te_h10),
            'freq':np.mean([v[0] for v in te_h10]),'ca':np.mean([v[1] for v in te_h10]),'lr':np.mean([v[2] for v in te_h10])})

    # Ablation on first seed
    if seed_i==0 and not ablation_done:
        ablation_done=True
        base_h10_lr=np.mean([v[2] for v in te_h10])
        log('Ablation:')
        for drop_i in range(8):
            keep=[j for j in range(8) if j!=drop_i]
            sc_a=StandardScaler(); X_a_s=sc_a.fit_transform(X_tr[:,keep])
            lr_a=LogisticRegression(C=1.0,max_iter=2000,random_state=SEED)
            lr_a.fit(X_a_s,y_tr)
            ah10=0; at=0
            for of_s in te_ofs:
                X_of=of_X[of_s]; y_of=of_y[of_s]; ql=of_qids[of_s]; cl=of_clists[of_s]
                X_of_s=sc_a.transform(X_of[:,keep]); lr_s=lr_a.predict_proba(X_of_s)[:,1]; off=0
                for k in range(len(ql)):
                    labels=dict(zip(cl[k],y_of[off:off+len(cl[k])].astype(int)))
                    if sum(labels.values())==0:
                        off+=len(cl[k]); continue
                    at+=1; nc=len(cl[k]); sl=lr_s[off:off+nc]
                    order=np.argsort(-sl); lab=np.array([labels[cl[k][j]] for j in range(nc)])[order]
                    pi=np.where(lab==1)[0]; ah10+=int(len(pi)>0 and pi[0]<10); off+=nc
            adrop=ah10/max(at,1)
            log(f'  Drop {feat_names[drop_i]:10s}: H10={adrop:.4f}  delta={adrop-base_h10_lr:+.4f}')
    if (seed_i+1)%3==0:
        log(f'  Seed {seed_i+1}/{N_SEEDS}')

sr=pd.DataFrame(seed_results)
log(f'\n=== MULTI-SEED ({N_SEEDS} seeds) ===')
for name,col in [('Frequency','freq'),('Content-Aware','ca'),('Learned Kernel','lr')]:
    log(f'{name:20s} H10={sr[col].mean():.4f}+/-{sr[col].std():.4f}')
dlr=sr.lr.mean()-sr.ca.mean()
log(f'Learned vs CA: +{dlr:.4f} ({100*dlr/sr.ca.mean():.0f}%) p<0.001: {dlr>3*sr.lr.std()}')
log(f'Time: {(time.time()-t0)/60:.1f} min')
