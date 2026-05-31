#!/usr/bin/env python3
"""
D4A2D1-FULL-GATE — Full dual-encoder training gate.
ArmA full-vocab multi-positive softmax only. 100k train, 15k val, temp sweep, freq fusion, bootstrap.
Output: plan_results/routeA_chembl37k_d4a2d1_full_gate/
"""
import json, csv, os, sys, time, math, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import psutil
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.logger().setLevel(RDLogger.ERROR); warnings.filterwarnings('ignore')

BASE = Path("E:/zuhui/bioisosteric_diffusion")
D4A0 = BASE / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze"
DESIGN = BASE / "plan_results/routeA_chembl37k_d4a2d0_dual_encoder_design"
D4A1 = BASE / "plan_results/routeA_chembl37k_d4a1_learned_ranker"
OUT = BASE / "plan_results/routeA_chembl37k_d4a2d1_full_gate"
MANIFEST = D4A0 / "d4a0_query_split_manifest.jsonl"
VOCAB_CSV = D4A0 / "d4a0_train_replacement_vocabulary.csv"
SEED = 20260523
torch.manual_seed(SEED); np.random.seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)
NOW = time.strftime("%Y-%m-%dT%H:%M:%S")
SIG_ORDER = ['C|C', 'C|N', 'C|O', 'C|S', 'N|S']
SIG_TO_IDX = {s: i for i, s in enumerate(SIG_ORDER)}
FP_BITS, N_SIG, HIDDEN, OUT_DIM = 2048, 5, 256, 128
TEMPS = [0.03, 0.05, 0.07, 0.1]
ALPHAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
N_BOOTSTRAP = 2000
HGB_T10 = 0.7008

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def write_csv(name, rows, fields):
    with open(OUT/name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def write_json(name, obj):
    with open(OUT/name, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)
def write_md(name, txt):
    with open(OUT/name, "w", encoding="utf-8") as f: f.write(txt)

def morgan_fp(smiles, nBits=FP_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    arr = np.zeros(nBits, dtype=np.float32); DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

class DualEncoder(nn.Module):
    def __init__(self, qdim=FP_BITS+N_SIG, cdim=FP_BITS, h=HIDDEN, od=OUT_DIM, temp=0.07):
        super().__init__()
        self.q_enc = nn.Sequential(nn.Linear(qdim, h), nn.ReLU(), nn.Linear(h, od))
        self.c_enc = nn.Sequential(nn.Linear(cdim, h), nn.ReLU(), nn.Linear(h, od))
        self.log_t = nn.Parameter(torch.tensor(np.log(temp)))
    @property
    def temp(self): return torch.exp(self.log_t)
    def forward(self, q, c):
        return (self.q_enc(q) @ self.c_enc(c).T) / self.temp

def mp_softmax_loss(sc, pm):
    plse = torch.logsumexp(sc.masked_fill(~pm, -1e9), dim=1)
    return (torch.logsumexp(sc, dim=1) - plse).mean()

def compute_topk(scores, pm):
    """scores[N,V], mask[N,V] → top1,5,10,20,50,mrr,per_query_hits"""
    N = scores.shape[0]; si = scores.argsort(descending=True, dim=1)
    t1=t5=t10=t20=t50=0; mrr=0.0; pqh = []
    for i in range(N):
        ps = set(torch.where(pm[i])[0].tolist())
        if not ps: pqh.append(0); continue
        rk = si[i].tolist()
        hits = sum(1 for j in ps if j in rk[:10])
        if rk[0] in ps: t1+=1
        if set(rk[:5])&ps: t5+=1
        if set(rk[:10])&ps: t10+=1
        if set(rk[:20])&ps: t20+=1
        if set(rk[:50])&ps: t50+=1
        for ri,idx in enumerate(rk):
            if idx in ps: mrr+=1.0/(ri+1); break
        pqh.append(hits)
    n=float(N); return t1/n, t5/n, t10/n, t20/n, t50/n, mrr/n, pqh

def compute_rank_metrics(rank, pm_np):
    N=pm_np.shape[0]; t1=t5=t10=t20=t50=0; mrr=0.0
    for i in range(N):
        ps=set(np.where(pm_np[i])[0])
        if not ps: continue
        if rank[0] in ps: t1+=1
        if set(rank[:5])&ps: t5+=1
        if set(rank[:10])&ps: t10+=1
        if set(rank[:20])&ps: t20+=1
        if set(rank[:50])&ps: t50+=1
        for ri,idx in enumerate(rank):
            if idx in ps: mrr+=1/(ri+1); break
    n=float(N); return t1/n,t5/n,t10/n,t20/n,t50/n,mrr/n

def bootstrap_metric(scores, pm, metric_fn, n_boot=N_BOOTSTRAP):
    """Bootstrap CI for a metric function that returns scalar."""
    N = scores.shape[0]; rng = np.random.RandomState(SEED)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(N, N, replace=True)
        _,_,_,_,_,_,val = metric_fn(scores[idx], pm[idx])
        vals.append(val)
    vals = np.array(vals)
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def bootstrap_compare(scores_a, pm_a, scores_b, pm_b, n_boot=N_BOOTSTRAP):
    """Compare two methods by Top10 difference via bootstrap."""
    assert scores_a.shape[0] == scores_b.shape[0] == pm_a.shape[0] == pm_b.shape[0]
    N = scores_a.shape[0]; rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(N, N, replace=True)
        _,_,t10a,_,_,_,_ = compute_topk(scores_a[idx], pm_a[idx])
        _,_,t10b,_,_,_,_ = compute_topk(scores_b[idx], pm_b[idx])
        diffs.append(t10a - t10b)
    diffs = np.array(diffs)
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

# ═══ DATA LOADING ═══
log("="*60); log("LOADING DATA"); log("="*60)
vocab = []
with open(VOCAB_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        r['global_train_frequency'] = int(r['global_train_frequency'])
        vocab.append(r)
v_smi = []; v_freq = []; v_fp_list = []
for r in vocab:
    fp = morgan_fp(r['replacement_smiles'])
    if fp is not None: v_smi.append(r['replacement_smiles']); v_freq.append(r['global_train_frequency']); v_fp_list.append(fp)
V = len(v_smi); cand_t = torch.tensor(np.array(v_fp_list, dtype=np.float32))
log(f"Vocab: {V} fragments, cand_tensor: {list(cand_t.shape)}")
vfreq_arr = np.array(v_freq, dtype=np.int32)
log_af = np.log(np.array(v_freq, dtype=np.float32) + 1)  # log(attach_freq+1) for fusion
log_af_t = torch.tensor(log_af, dtype=torch.float32)

entries = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
tq_raw = [e for e in entries if e['split']=='train']
vq_raw = [e for e in entries if e['split']=='val']
test_raw = [e for e in entries if e['split']=='test']
log(f"Queries: train={len(tq_raw)}, val={len(vq_raw)}, test={len(test_raw)}")

def build_query_set(queries, name):
    fps, ohs, poses = [], [], []
    for e in queries:
        fp = morgan_fp(e['old_fragment_smiles'])
        sig = e.get('attachment_signature')
        if fp is None or sig not in SIG_TO_IDX: continue
        ps = [i for i, vs in enumerate(v_smi) if vs in e['positive_replacement_set']]
        if not ps: continue
        fps.append(fp); ohs.append(SIG_TO_IDX[sig]); poses.append(ps)
    ft = torch.tensor(np.array(fps, dtype=np.float32))
    oh = torch.zeros(len(ohs), N_SIG, dtype=torch.float32)
    for i, idx in enumerate(ohs): oh[i, idx] = 1.0
    qt = torch.cat([ft, oh], dim=1)
    pm = torch.zeros(qt.shape[0], V, dtype=torch.bool)
    for qi, idxs in enumerate(poses): pm[qi, idxs] = True
    log(f"  {name}: {qt.shape[0]} queries, {pm.sum().item()} positives")
    return qt, pm, poses

train_q, train_pm, _ = build_query_set(tq_raw, "train")
val_q, val_pm, val_pos = build_query_set(vq_raw, "val")
BEST_TEMP = None; BEST_ALPHA = None; BEST_MODEL = None; BEST_VAL_T10 = -1

# ═══ TEMPERATURE SWEEP ═══
log("="*60); log("TEMPERATURE SWEEP"); log("="*60)
sweep_rows = []
for temp in TEMPS:
    model = DualEncoder(temp=temp)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    BS = 256
    for ep in range(1, 4):  # 3 epochs per temp for sweep
        model.train()
        perm = torch.randperm(train_q.shape[0])
        for st in range(0, train_q.shape[0], BS):
            en = min(st+BS, train_q.shape[0]); batch_q = train_q[perm[st:en]]; batch_pm = train_pm[perm[st:en]]
            sc = model(batch_q, cand_t); loss = mp_softmax_loss(sc, batch_pm)
            if torch.isnan(loss).item(): break
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vs = model(val_q, cand_t); t1,t5,t10,t20,t50,mrr,_ = compute_topk(vs, val_pm)
    sweep_rows.append({"temp":temp,"val_top1":round(t1,4),"val_top5":round(t5,4),"val_top10":round(t10,4),"val_mrr":round(mrr,4)})
    log(f"  T={temp:.4f}: T10={t10:.4f} MRR={mrr:.4f}")
    if t10 > BEST_VAL_T10: BEST_VAL_T10 = t10; BEST_MODEL = model; BEST_TEMP = temp
write_csv("d4a2d1_full_temp_sweep.csv", sweep_rows, ["temp","val_top1","val_top5","val_top10","val_mrr"])

log(f"Best temp: {BEST_TEMP} (T10={BEST_VAL_T10:.4f})")
model = BEST_MODEL
model.log_t = nn.Parameter(torch.tensor(np.log(BEST_TEMP)))  # reset best temp

# ═══ FULL TRAINING ═══
log("="*60); log("FULL TRAINING (ArmA full-vocab)"); log("="*60)
BS, MAX_EP = 256, 20
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
train_log = []; best_t10 = BEST_VAL_T10; best_ep = 0; patience = 5; no_improve = 0

for ep in range(1, MAX_EP+1):
    t0 = time.time()
    perm = torch.randperm(train_q.shape[0])
    model.train(); losses = []
    for st in range(0, train_q.shape[0], BS):
        en = min(st+BS, train_q.shape[0])
        sc = model(train_q[perm[st:en]], cand_t)
        loss = mp_softmax_loss(sc, train_pm[perm[st:en]])
        if torch.isnan(loss).item(): log(f"  NAN ep{ep}"); break
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    tl = float(np.mean(losses)) if losses else float('nan')
    model.eval()
    with torch.no_grad():
        vs = model(val_q, cand_t); vl = mp_softmax_loss(vs, val_pm).item()
        t1,t5,t10,t20,t50,mrr,_ = compute_topk(vs, val_pm)
    rt = round(time.time()-t0, 1); mem = round(psutil.Process().memory_info().rss/1e6, 1)
    log(f"  Ep{ep:2d}/{MAX_EP} tl={tl:.4f} vl={vl:.4f} T1={t1:.4f} T5={t5:.4f} T10={t10:.4f} MRR={mrr:.4f} {rt}s {mem}MB")
    train_log.append({"epoch":ep,"train_loss":round(tl,6),"val_loss":round(vl,6),
        "val_top1":round(t1,4),"val_top5":round(t5,4),"val_top10":round(t10,4),
        "val_top20":round(t20,4),"val_top50":round(t50,4),"val_mrr":round(mrr,4),
        "mem_mb":mem,"runtime_sec":rt})
    if t10 > best_t10 + 0.002:
        best_t10 = t10; best_ep = ep; no_improve = 0
    else:
        no_improve += 1
    torch.save(model.state_dict(), OUT/f"d4a2d1_epoch{ep}.pt")
    if no_improve >= patience: log(f"  Early stop at ep{ep} (patience={patience})"); break

write_csv("d4a2d1_full_training_log.csv", train_log,
    ["epoch","train_loss","val_loss","val_top1","val_top5","val_top10","val_top20","val_top50","val_mrr","mem_mb","runtime_sec"])
log(f"Training done. Best epoch: {best_ep}, T10={best_t10:.4f}")

# Load best model
if (OUT/f"d4a2d1_epoch{best_ep}.pt").exists():
    model.load_state_dict(torch.load(OUT/f"d4a2d1_epoch{best_ep}.pt", map_location='cpu'))
model.eval()

# ═══ FREQUENCY FUSION SWEEP ═══
log("="*60); log("FREQUENCY FUSION SWEEP"); log("="*60)
with torch.no_grad(): de_scores = model(val_q, cand_t)
fusion_rows = []
for alpha in ALPHAS:
    fused = de_scores + alpha * log_af_t
    t1,t5,t10,t20,t50,mrr,_ = compute_topk(fused, val_pm)
    fusion_rows.append({"alpha":alpha,"val_top1":round(t1,4),"val_top5":round(t5,4),"val_top10":round(t10,4),"val_mrr":round(mrr,4)})
    log(f"  alpha={alpha:.3f}: T10={t10:.4f} MRR={mrr:.4f}")
best_fusion = max(fusion_rows, key=lambda r: r['val_top10'])
BEST_ALPHA = best_fusion['alpha']
log(f"Best alpha: {BEST_ALPHA} (T10={best_fusion['val_top10']:.4f})")
write_csv("d4a2d1_full_fusion_sweep.csv", fusion_rows, ["alpha","val_top1","val_top5","val_top10","val_mrr"])

# ═══ VAL EVALUATION ═══
log("="*60); log("VAL EVALUATION"); log("="*60)
model.eval()
with torch.no_grad():
    de_scores = model(val_q, cand_t)
    de_t1,de_t5,de_t10,de_t20,de_t50,de_mrr,_ = compute_topk(de_scores, val_pm)

if BEST_ALPHA > 0:
    fused_scores = de_scores + BEST_ALPHA * log_af_t
    _,_,f_t10,_,_,f_mrr,_ = compute_topk(fused_scores, val_pm)
    log(f"  Fused(T10={f_t10:.4f} MRR={f_mrr:.4f}) vs DE(T10={de_t10:.4f} MRR={de_mrr:.4f})")

freq_r = np.argsort(-vfreq_arr)
b1_t1,b1_t5,b1_t10,b1_t20,b1_t50,b1_mrr = compute_rank_metrics(freq_r, val_pm.numpy())

# Bootstrap DE vs attach_freq on val
b_mean, b_lo, b_hi = bootstrap_compare(de_scores, val_pm,
    torch.tensor(np.tile(freq_r, (val_q.shape[0],1)), dtype=torch.long), val_pm)

val_metrics = [
    {"method":"DualEncoder","top1":round(de_t1,4),"top5":round(de_t5,4),"top10":round(de_t10,4),"top20":round(de_t20,4),"top50":round(de_t50,4),"mrr":round(de_mrr,4)},
    {"method":"B1_attach_freq","top1":round(b1_t1,4),"top5":round(b1_t5,4),"top10":round(b1_t10,4),"top20":round(b1_t20,4),"top50":round(b1_t50,4),"mrr":round(b1_mrr,4)},
]
if BEST_ALPHA > 0:
    val_metrics.append({"method":f"DE+logAF(a={BEST_ALPHA})","top1":0,"top5":0,"top10":round(f_t10,4),"top20":0,"top50":0,"mrr":round(f_mrr,4)})
write_csv("d4a2d1_full_val_metrics.csv", val_metrics,
    ["method","top1","top5","top10","top20","top50","mrr"])

# Subset analysis on val
fhigh = np.percentile(vfreq_arr, 75); flow = np.percentile(vfreq_arr, 25)
# hard baseline miss: queries where B1 Top10 = 0 (attach_freq finds nothing in top 10)
hard_miss_idx = []
for qi in range(val_q.shape[0]):
    rank = np.argsort(-vfreq_arr)
    ps = set(torch.where(val_pm[qi])[0].tolist())
    if not ps: continue
    if not set(rank[:10]) & ps: hard_miss_idx.append(qi)

sub_rows = []
for name, query_idxs in [
    ("multi_pos", [qi for qi in range(val_q.shape[0]) if val_pm[qi].sum()>1]),
    ("single_pos", [qi for qi in range(val_q.shape[0]) if val_pm[qi].sum()==1]),
    ("freq_repl", [qi for qi in range(val_q.shape[0]) if any(vfreq_arr[p]>=fhigh for p in torch.where(val_pm[qi])[0].tolist())]),
    ("rare_repl", [qi for qi in range(val_q.shape[0]) if any(vfreq_arr[p]<=flow for p in torch.where(val_pm[qi])[0].tolist())]),
    ("hard_baseline_miss", hard_miss_idx),
]:
    if len(query_idxs) < 5: continue
    idxs = torch.tensor(query_idxs, dtype=torch.long)
    if BEST_ALPHA > 0:
        ss = (de_scores + BEST_ALPHA * log_af_t)[idxs]
    else:
        ss = de_scores[idxs]
    sp = val_pm[idxs]
    st1,st5,st10,st20,st50,smrr,_ = compute_topk(ss, sp)
    _,_,b1s10,_,_,b1smrr,_ = compute_topk(torch.tensor(np.tile(freq_r, (len(idxs),1))), sp)
    sub_rows.append({"subset":name,"n":len(idxs),
        "de_t1":round(st1,4),"de_t5":round(st5,4),"de_t10":round(st10,4),"de_t20":round(st20,4),"de_t50":round(st50,4),"de_mrr":round(smrr,4),
        "b1_t10":round(b1s10,4),"b1_mrr":round(b1smrr,4)})
    log(f"  {name}(n={len(idxs)}): DE_T10={st10:.4f} B1_T10={b1s10:.4f}")
write_csv("d4a2d1_full_val_by_subset.csv", sub_rows,
    ["subset","n","de_t1","de_t5","de_t10","de_t20","de_t50","de_mrr","b1_t10","b1_mrr"])

# Top1 warning
if de_t1 < b1_t1:
    log("  TOP1 WARNING: DE Top1 < B1 Top1 — model is a top-K proposal model, not a top-1 selector")

# ═══ TEST EVALUATION ═══
log("="*60); log("TEST EVALUATION"); log("="*60)
test_q, test_pm, _ = build_query_set(test_raw, "test")
model.eval()
with torch.no_grad():
    de_ts = model(test_q, cand_t)
    if BEST_ALPHA > 0: de_ts = de_ts + BEST_ALPHA * log_af_t
    de_t1t,de_t5t,de_t10t,de_t20t,de_t50t,de_mrrt,_ = compute_topk(de_ts, test_pm)
freq_r = np.argsort(-vfreq_arr)
b1_t1t,b1_t5t,b1_t10t,b1_t20t,b1_t50t,b1_mrrt = compute_rank_metrics(freq_r, test_pm.numpy())
log(f"  DE: T1={de_t1t:.4f} T5={de_t5t:.4f} T10={de_t10t:.4f} T50={de_t50t:.4f} MRR={de_mrrt:.4f}")
log(f"  B1: T1={b1_t1t:.4f} T5={b1_t5t:.4f} T10={b1_t10t:.4f} MRR={b1_mrrt:.4f}")

# Bootstrap on test
b_mean_t, b_lo_t, b_hi_t = bootstrap_compare(de_ts, test_pm,
    torch.tensor(np.tile(freq_r, (test_q.shape[0],1)), dtype=torch.long), test_pm)
log(f"  Bootstrap DE-B1 delta: {b_mean_t:.4f} [{b_lo_t:.4f}, {b_hi_t:.4f}]")

test_metrics = [
    {"method":"DualEncoder","top1":round(de_t1t,4),"top5":round(de_t5t,4),"top10":round(de_t10t,4),"top20":round(de_t20t,4),"top50":round(de_t50t,4),"mrr":round(de_mrrt,4)},
    {"method":"B1_attach_freq","top1":round(b1_t1t,4),"top5":round(b1_t5t,4),"top10":round(b1_t10t,4),"top20":round(b1_t20t,4),"top50":round(b1_t50t,4),"mrr":round(b1_mrrt,4)},
]
write_csv("d4a2d1_full_test_metrics.csv", test_metrics,
    ["method","top1","top5","top10","top20","top50","mrr"])

bootstrap_rows = [
    {"comparison":"DE_vs_B1_val","delta_mean":round(b_mean,4),"ci_lo":round(b_lo,4),"ci_hi":round(b_hi,4),"significant_05":"YES" if b_lo>0 else "NO"},
    {"comparison":"DE_vs_B1_test","delta_mean":round(b_mean_t,4),"ci_lo":round(b_lo_t,4),"ci_hi":round(b_hi_t,4),"significant_05":"YES" if b_lo_t>0 else "NO"},
]
write_csv("d4a2d1_full_bootstrap.csv", bootstrap_rows,
    ["comparison","delta_mean","ci_lo","ci_hi","significant_05"])

# ═══ VERDICT ═══
log("="*60); log("VERDICT"); log("="*60)
beats_attach = de_t10t > b1_t10t and b_lo_t > 0
beats_hgb = de_t10t > HGB_T10 and b_lo_t > 0
top1_weak = de_t1t < b1_t1t

if beats_hgb:
    verdict = "A. D4A2D1_FULL_EXCEEDS_HGB"
elif beats_attach and de_t10t >= HGB_T10 - 0.03:
    verdict = "B. D4A2D1_FULL_APPROACHES_HGB"
elif beats_attach:
    verdict = "C. D4A2D1_FULL_BEATS_ATTACH_NOT_HGB"
else:
    verdict = "D. D4A2D1_FULL_FAILS_BASELINE"

log(f"Verdict: {verdict}")
log(f"  DE beats attach: {beats_attach} (delta={b_mean_t:.4f} [{b_lo_t:.4f},{b_hi_t:.4f}])")
log(f"  DE meets/exceeds HGB: {de_t10t>=HGB_T10} (DE={de_t10t:.4f} vs HGB={HGB_T10})")
log(f"  Top1 weak: {top1_weak} (DE={de_t1t:.4f} vs B1={b1_t1t:.4f})")

md = f"""# D4A2D1 FULL GATE VERDICT
Date: {NOW}
Verdict: **{verdict}**

## Configuration
- Train: {train_q.shape[0]} queries, Val: {val_q.shape[0]}, Test: {test_q.shape[0]}
- Vocab: {V} fragments, full-vocab multi-positive softmax
- Architecture: DualEncoder(2053→256→128, 2048→256→128)
- Best temperature: {BEST_TEMP}, Best fusion alpha: {BEST_ALPHA}
- Epochs: {len(train_log)} (best={best_ep}), Patience: {patience}
- Bootstrap: {N_BOOTSTRAP} resamples, 95% CI

## Test Metrics
| Method | Top1 | Top5 | Top10 | Top20 | Top50 | MRR |
|--------|:----:|:----:|:-----:|:-----:|:-----:|:---:|
| DualEncoder | {de_t1t:.4f} | {de_t5t:.4f} | **{de_t10t:.4f}** | {de_t20t:.4f} | {de_t50t:.4f} | {de_mrrt:.4f} |
| B1 attach_freq | {b1_t1t:.4f} | {b1_t5t:.4f} | {b1_t10t:.4f} | {b1_t20t:.4f} | {b1_t50t:.4f} | {b1_mrrt:.4f} |

## Bootstrap Analysis
- DE vs B1 delta: {b_mean_t:.4f} [{b_lo_t:.4f}, {b_hi_t:.4f}] — {'significant' if b_lo_t>0 else 'NOT significant'}
- DE vs HGB ({HGB_T10}): {'EXCEEDS' if de_t10t>=HGB_T10 else 'BELOW by '+str(round(HGB_T10-de_t10t,4))+'pp'}

## Top1 Warning
{f'**ACTIVE**: DE Top1 ({de_t1t:.4f}) < B1 Top1 ({b1_t1t:.4f}). Model is a **top-K proposal model**, not a top-1 selector. The model should be used to propose a ranked list for expert review, not to auto-select a single replacement.' if top1_weak else 'DE Top1 >= B1 Top1 — model functions as a top-1 selector.'}

## Claims
- {'✅' if beats_attach else '❌'} DE beats attachment_frequency (bootstrap significant, p<0.05)
- {'✅' if beats_hgb else '❌'} DE exceeds canonical HGB ({HGB_T10}) with bootstrap support
- {'⚠️' if top1_weak else '✅'} Top1 limitation acknowledged

## Verdict Options
A. EXCEEDS_HGB B. APPROACHES_HGB C. BEATS_ATTACH_NOT_HGB D. FAILS_BASELINE
"""
write_md("D4A2D1_FULL_GATE_VERDICT.md", md)

write_json("d4a2d1_full_gate_summary.json", {
    "timestamp":NOW,"verdict":verdict,"best_temp":BEST_TEMP,"best_alpha":BEST_ALPHA,
    "best_epoch":best_ep,"n_epochs":len(train_log),
    "test_de_t10":round(de_t10t,4),"test_b1_t10":round(b1_t10t,4),
    "bootstrap_delta":round(b_mean_t,4),"bootstrap_ci":[round(b_lo_t,4),round(b_hi_t,4)],
    "significant":b_lo_t>0,"beats_hgb":de_t10t>=HGB_T10,"top1_weak":top1_weak
})

# Save best model manifest
torch.save(model.state_dict(), OUT/"d4a2d1_best_model.pt")
checkpoints = sorted(OUT.glob("d4a2d1_epoch*.pt"))
write_json("d4a2d1_model_artifacts_manifest.json", {
    "timestamp":NOW,"verdict":verdict,"best_epoch":best_ep,"best_alpha":BEST_ALPHA,
    "n_checkpoints":len(checkpoints),"best_model":"d4a2d1_best_model.pt",
    "checkpoints":[p.name for p in checkpoints]
})

dlog = f"# D4A2D1 FULL GATE Decision Log\nDate: {NOW}\nVerdict: **{verdict}**\n"
dlog += f"DE_T10={de_t10t:.4f} B1_T10={b1_t10t:.4f} HGB={HGB_T10}\n"
dlog += f"Bootstrap DE-B1: {b_mean_t:.4f} [{b_lo_t:.4f},{b_hi_t:.4f}] significant={'YES' if b_lo_t>0 else 'NO'}\n"
write_md("MAIN_DECISION_LOG.md", dlog)

log(f"\nDONE. Verdict: {verdict}")
log(f"Output: {OUT}")
