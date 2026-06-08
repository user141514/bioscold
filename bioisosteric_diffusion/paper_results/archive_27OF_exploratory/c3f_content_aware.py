#!/usr/bin/env python3
"""C3F + Content-Aware Fallback for cold-start OFs."""
import json, numpy as np, pandas as pd, time, glob
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors

SEED=20260603; t0=time.time()
def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# Setup (same as LOO-CV)
manifest={}
with open('plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl',encoding='utf-8') as f:
    for l in f:
        if l.strip():
            q=json.loads(l)
            manifest[q['query_id']]=q
all_ofs=sorted(set(q['old_fragment_smiles'] for q in manifest.values()))

of_fps={}; of_props={}
for s in all_ofs:
    m=Chem.MolFromSmiles(s)
    if m:
        fp=np.zeros(2048,dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,2048),fp)
        of_fps[s]=fp
        of_props[s]={
            'heavy':m.GetNumHeavyAtoms(),'rings':Descriptors.RingCount(m),
            'mw':Descriptors.MolWt(m),'logp':Descriptors.MolLogP(m),
            'tpsa':Descriptors.TPSA(m)}

of_list=[s for s in all_ofs if s in of_fps]
n_of=len(of_list)
fp_mat=np.array([of_fps[s] for s in of_list],dtype=np.float32)
inter=np.dot(fp_mat,fp_mat.T)
ts=fp_mat.sum(axis=1)[:,None]; denom=ts+ts.T-inter; denom[denom==0]=1.0
sim_full=pd.DataFrame(inter/denom,index=of_list,columns=of_list)

of_queries=defaultdict(list)
for q in manifest.values():
    if q['old_fragment_smiles'] in of_list:
        of_queries[q['old_fragment_smiles']].append(q['query_id'])

# Candidate success index
of_pos=defaultdict(lambda: defaultdict(float))
for shard in sorted(glob.glob('plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/matrices/train/train_features_shard_0*.jsonl'))[:80]:
    with open(shard,encoding='utf-8') as f:
        for l in f:
            if not l.strip():
                continue
            rec=json.loads(l)
            q=manifest.get(rec['query_id'])
            if q and q['old_fragment_smiles'] in of_list:
                of_pos[q['old_fragment_smiles']][rec['candidate']]+=1
for of_s in of_pos:
    total=sum(of_pos[of_s].values())
    if total>0:
        for c in of_pos[of_s]:
            of_pos[of_s][c]/=total

# Query labels
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

# Cache candidate FPs for content score
log('Caching candidate fingerprints...')
cand_fps={}; cand_props={}
for qid in query_labels:
    for cand in query_labels[qid]:
        if cand not in cand_fps:
            m=Chem.MolFromSmiles(cand)
            if m:
                fp=np.zeros(2048,dtype=np.float32)
                Chem.DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,2048),fp)
                cand_fps[cand]=fp
                cand_props[cand]={
                    'heavy':m.GetNumHeavyAtoms(),'rings':Descriptors.RingCount(m),
                    'mw':Descriptors.MolWt(m),'logp':Descriptors.MolLogP(m),
                    'tpsa':Descriptors.TPSA(m)}
log(f'Cached {len(cand_fps)} candidate FPs')

# Content-based score
def content_score(cand, test_of):
    if cand not in cand_fps or test_of not in of_fps:
        return 0.0
    cfp=cand_fps[cand]; ofp=of_fps[test_of]
    inter_v=np.dot(cfp,ofp)
    denom_v=cfp.sum()+ofp.sum()-inter_v
    morgan_sim=inter_v/max(denom_v,0.001)
    op=of_props[test_of]; cp=cand_props[cand]
    d_heavy=abs(cp['heavy']-op['heavy'])
    d_rings=abs(cp['rings']-op['rings'])
    d_mw=abs(cp['mw']-op['mw'])/(max(op['mw'],1))
    d_logp=abs(cp['logp']-op['logp'])/max(abs(op['logp'])+1,1)
    physchem=1.0/(1.0+d_heavy+d_rings+d_mw+d_logp)
    return 0.7*morgan_sim+0.3*physchem

# === LOO-CV + Content-Aware ===
K=5; alpha=0.3
log(f'\n=== LOO-CV + Content-Aware Fallback (K={K}, alpha={alpha}) ===')
results=[]

for i,test_of in enumerate(of_list):
    train_ofs=[s for j,s in enumerate(of_list) if j!=i]
    test_qids=of_queries.get(test_of,[])
    if not test_qids:
        continue

    sims=sim_full.loc[test_of,train_ofs].sort_values(ascending=False)
    top_k=sims.head(K); weights=top_k.values; ws=weights.sum()
    has_signal=(ws>0.001)

    q_h_r=0; q_h_c=0; q_h_ca=0; q_t=0; q_r_ca=0; q_l_ca=0
    for qid in test_qids:
        labels=query_labels.get(qid,{})
        pos_labels={c:l for c,l in labels.items() if l==1}
        if not pos_labels:
            continue
        q_t+=1
        cands=list(labels.keys()); n_c=len(cands)

        # C3F
        c3f_s=np.zeros(n_c)
        if has_signal:
            for tr_of,w in zip(top_k.index,weights):
                pdict=of_pos.get(tr_of,{})
                if pdict:
                    c3f_s+=np.array([pdict.get(c,0.0) for c in cands])*w
            c3f_s/=ws

        # Content
        content_s=np.array([content_score(c,test_of) for c in cands])

        # Hybrid
        if has_signal:
            c3f_w=ws/(ws+alpha)
            hybrid_s=c3f_w*c3f_s+(1-c3f_w)*content_s
        else:
            hybrid_s=content_s

        # Random
        rng=np.random.RandomState(hash(qid)%2**32)
        order_r=rng.permutation(n_c)

        def h10(order):
            lab=np.array([labels.get(cands[j],0) for j in range(n_c)])[order]
            pi=np.where(lab==1)[0]
            return int(len(pi)>0 and pi[0]<10)

        rh=h10(order_r); ch=h10(np.argsort(-c3f_s)); cah=h10(np.argsort(-hybrid_s))
        q_h_r+=rh; q_h_c+=ch; q_h_ca+=cah
        if rh==0 and cah==1:
            q_r_ca+=1
        elif rh==1 and cah==0:
            q_l_ca+=1

    if q_t>0:
        results.append({
            'test_of':test_of,'n_q':q_t,'has_c3f':has_signal,
            'rand_h10':q_h_r/q_t,'c3f_h10':q_h_c/q_t,'ca_h10':q_h_ca/q_t,
            'rescue_ca':q_r_ca,'loss_ca':q_l_ca,'top_sim':sims.iloc[0]})

    if (i+1)%20==0:
        log(f'  OF {i+1}/{n_of}: {test_of[:25]} sig={has_signal} CA_H10={q_h_ca/max(q_t,1):.3f}')

rd=pd.DataFrame(results); n_v=len(rd)
h_r=rd.rand_h10.mean(); h_c=rd.c3f_h10.mean(); h_ca=rd.ca_h10.mean()

log(f'\n=== RESULTS (N={n_v} OFs) ===')
log(f'Random baseline:     H10={h_r:.4f}')
log(f'C3F only:            H10={h_c:.4f}  vs Rand: {h_c-h_r:+.4f}')
log(f'C3F + Content-Aware: H10={h_ca:.4f}  vs Rand: {h_ca-h_r:+.4f}')
log(f'Content-Aware lift over C3F: {h_ca-h_c:+.4f}')

# By C3F signal
log('\nBy C3F signal availability:')
for sig in ['YES','NO']:
    g=rd[rd.has_c3f==sig]
    if len(g)==0: continue
    hr_g=g.rand_h10.mean(); hc_g=g.c3f_h10.mean(); hca_g=g.ca_h10.mean()
    log(f'  Signal={sig:4s} N={len(g):3d} | Rand={hr_g:.4f} C3F={hc_g:.4f} CA={hca_g:.4f} | lift={hca_g-hc_g:+.4f}')

rd.to_csv('plan_results/routeA_newpaper_phase0_protocol_lock/c3f_content_aware.csv',index=False)
log(f'\n{(time.time()-t0)/60:.1f} min')
