#!/usr/bin/env python3
"""Learned Molecular Kernel: K(c, OF; theta) optimized for Hit@10 on train OFs, tested on cold-start."""
import json, numpy as np, pandas as pd, glob, time
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

SEED=20260603; t0=time.time()
def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# ====== DATA LOADING ======
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
        of_fps[s]=fp
        of_props[s]=np.array([m.GetNumHeavyAtoms(),Descriptors.RingCount(m),
            Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.TPSA(m)],dtype=np.float32)
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

# ====== FEATURE EXTRACTION ======
cand_fps={}
cand_props={}
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
                cand_props[cand]=np.array([m.GetNumHeavyAtoms(),Descriptors.RingCount(m),
                    Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.TPSA(m)],dtype=np.float32)
max_freq=max(cand_freq.values()) if cand_freq else 1

def extract_features(cand, of_s):
    """Feature vector for (candidate, OF) pair.
    Returns: [morgan_sim, bit_corr, delta_heavy, delta_rings, delta_MW, delta_logP, delta_TPSA, freq_norm]"""
    cfp=cand_fps.get(cand); ofp=of_fps[of_s]
    if cfp is None or ofp is None:
        return np.zeros(8,dtype=np.float32)

    # Morgan Tanimoto
    iv=cfp.dot(ofp); dv=cfp.sum()+ofp.sum()-iv; ms=iv/max(dv,0.001)

    # Bit-level correlation (how much do bit patterns align beyond Tanimoto)
    cfp_n=cfp/(cfp.sum()+1e-10); ofp_n=ofp/(ofp.sum()+1e-10)
    bit_corr=np.corrcoef(cfp_n,ofp_n)[0,1] if cfp.sum()>0 and ofp.sum()>0 else 0.0

    # PhysChem deltas
    cp=cand_props.get(cand,np.zeros(5))
    op=of_props[of_s]
    dh=abs(cp[0]-op[0])
    dr=abs(cp[1]-op[1])
    dm=abs(cp[2]-op[2])/max(op[2],1)
    dl=abs(cp[3]-op[3])/max(abs(op[3])+1,1)
    dt=abs(cp[4]-op[4])/max(op[4]+1,1)

    freq_n=cand_freq.get(cand,0)/max_freq
    return np.array([ms, bit_corr, dh, dr, dm, dl, dt, freq_n],dtype=np.float32)

# ====== TRAIN/TEST SPLIT BY OF ======
# Use OFs WITH labels for training, OFs WITHOUT labels for cold-start test
of_has_labels=set()
for qid in query_labels:
    q=manifest.get(qid)
    if q and q['old_fragment_smiles'] in of_valid:
        if sum(query_labels[qid].values())>0:
            of_has_labels.add(q['old_fragment_smiles'])

# Split the 27 labeled OFs into train/test
rng=np.random.RandomState(SEED)
labeled_ofs=sorted(of_has_labels)
rng.shuffle(labeled_ofs)
n_train=int(len(labeled_ofs)*0.7)
train_ofs=labeled_ofs[:n_train]
test_ofs=labeled_ofs[n_train:]
log(f'Train OFs: {len(train_ofs)}, Test OFs: {len(test_ofs)} (cold-start, held-out from labeled set)')

# Build training data: from labeled OFs
log('Building training pairs...')
X_train=[]
y_train=[]
for of_s in train_ofs:
    for qid in of_queries.get(of_s,[]):
        labels=query_labels.get(qid,{})
        for cand,l in labels.items():
            feats=extract_features(cand, of_s)
            X_train.append(feats)
            y_train.append(l)

X_train=np.array(X_train,dtype=np.float32)
y_train=np.array(y_train)
log(f'Train: {len(X_train)} pairs, pos rate={y_train.mean():.4f}')

# Train learned kernel model
sc=StandardScaler()
X_train_s=sc.fit_transform(X_train)

# Model 1: LR (simple, interpretable)
lr=LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
lr.fit(X_train_s, y_train)
log(f'LR coefs: morgan={lr.coef_[0][0]:.3f} corr={lr.coef_[0][1]:.3f} dH={lr.coef_[0][2]:.3f} dR={lr.coef_[0][3]:.3f} dMW={lr.coef_[0][4]:.3f} dlogP={lr.coef_[0][5]:.3f} dTPSA={lr.coef_[0][6]:.3f} freq={lr.coef_[0][7]:.3f}')

# Model 2: HGB (nonlinear)
hgb=HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, random_state=SEED)
hgb.fit(X_train_s, y_train)
log('HGB trained')

# ====== EVALUATION ON COLD-START OFs ======
log('\nEvaluating on cold-start OFs...')
results=[]
for test_of in test_ofs:
    q_lr=0; q_hgb=0; q_ca=0; q_freq=0; q_t=0
    for qid in of_queries.get(test_of,[]):
        labels=query_labels.get(qid,{})
        pos_count=sum(1 for v in labels.values() if v==1)
        if pos_count==0:
            continue
        q_t+=1
        all_c=list(labels.keys())
        nc=len(all_c)

        # Predict scores
        X_test=np.array([extract_features(c,test_of) for c in all_c],dtype=np.float32)
        X_test_s=sc.transform(X_test)

        lr_s=lr.predict_proba(X_test_s)[:,1]
        hgb_s=hgb.predict_proba(X_test_s)[:,1]

        # Content-Aware baseline
        ca_s=np.zeros(nc)
        for j,c in enumerate(all_c):
            if c in cand_fps and test_of in of_fps:
                cfp=cand_fps[c]; ofp=of_fps[test_of]
                iv=cfp.dot(ofp); dv=cfp.sum()+ofp.sum()-iv; ms=iv/max(dv,0.001)
                op=of_props[test_of]; cp=cand_props.get(c,np.zeros(5))
                dh=abs(cp[0]-op[0]); dr=abs(cp[1]-op[1])
                dm=abs(cp[2]-op[2])/max(op[2],1); dl=abs(cp[3]-op[3])/max(abs(op[3])+1,1)
                ca_s[j]=0.7*ms+0.3*(1.0/(1.0+dh+dr+dm+dl))

        freq_s=np.array([cand_freq.get(c,0)/max_freq for c in all_c])

        def h10(scores):
            order=np.argsort(-scores)
            lab=np.array([labels[all_c[j]] for j in range(nc)])[order]
            pi=np.where(lab==1)[0]
            return int(len(pi)>0 and pi[0]<10)

        q_lr+=h10(lr_s); q_hgb+=h10(hgb_s); q_ca+=h10(ca_s); q_freq+=h10(freq_s)

    if q_t>0:
        results.append({'of':test_of,'q':q_t,'freq':q_freq/q_t,'ca':q_ca/q_t,'lr':q_lr/q_t,'hgb':q_hgb/q_t})

rd=pd.DataFrame(results)
log(f'\n=== COLD-START OFs EVALUATION (N={len(rd)} OFs) ===')
for name,col in [('Frequency','freq'),('Content-Aware (CA)','ca'),('LR learned kernel','lr'),('HGB learned kernel','hgb')]:
    v=rd[col].mean()
    log(f'  {name:25s} H10={v:.4f}')
log(f'  Best: {max([(n,rd[c].mean()) for n,c in [("LR", "lr"),("HGB","hgb"),("CA","ca")]], key=lambda x:x[1])[0]}')

rd.to_csv('plan_results/routeA_newpaper_phase0_protocol_lock/learned_kernel_results.csv',index=False)
log(f'\nTime: {(time.time()-t0)/60:.1f} min')
