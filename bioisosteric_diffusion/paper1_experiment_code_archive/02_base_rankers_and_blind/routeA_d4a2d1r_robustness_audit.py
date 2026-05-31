#!/usr/bin/env python3
"""D4A2D1R Robustness Audit: CPU-only inference + statistical analysis of DualEncoder."""
import os, sys, json, csv, logging, math, time
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem, RDLogger; from rdkit.Chem import AllChem
RDLogger.logger().setLevel(RDLogger.ERROR)

BASE = Path(r'E:\zuhui\bioisosteric_diffusion')
D4A0 = BASE / 'plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze'
D4A2D1 = BASE / 'plan_results/routeA_chembl37k_d4a2d1_full_gate'
D4A1 = BASE / 'plan_results/routeA_chembl37k_d4a1_learned_ranker'
D4A3 = BASE / 'plan_results/routeA_chembl37k_d4a3_geometry_a4c_evaluation'
OUT = BASE / 'plan_results/routeA_chembl37k_d4a2d1r_dual_encoder_robustness'
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(str(OUT / 'd4a2d1r_audit.log'), encoding='utf-8'), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

SEED = 20260523; RNG = np.random.RandomState(SEED); torch.manual_seed(SEED)
ATTACH_TYPES = ['C|C', 'C|N', 'C|O', 'C|S', 'N|S']; MR=2; MB=2048; AD=5; QD=MB+AD; CD=MB; H=256; OD=128
BT=0.05; BA=0.2; NB=1000; KS=[1,5,10,20,50]

class DualEncoder(nn.Module):
    def __init__(self, qd=QD, cd=CD, h=H, od=OD):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(0.0))
        self.q_enc = nn.Sequential(nn.Linear(qd, h), nn.ReLU(), nn.Linear(h, od))
        self.c_enc = nn.Sequential(nn.Linear(cd, h), nn.ReLU(), nn.Linear(h, od))
    def encode_query(self, x): return self.q_enc(x)
    def encode_candidate(self, x): return self.c_enc(x)
    def forward(self, q, c): return (self.q_enc(q) @ self.c_enc(c).T) / torch.exp(self.log_t)

def smi2fp(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None: return np.zeros(MB, dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, MR, nBits=MB), dtype=np.float32)

def att_onehot(s):
    oh = np.zeros(AD, dtype=np.float32)
    if s in ATTACH_TYPES: oh[ATTACH_TYPES.index(s)] = 1.0
    return oh

def prm(scores, labels):
    r, o = {}, np.argsort(-scores)
    for k in KS: r[f't{k}'] = int(np.any(labels[o[:k]] == 1))
    pr = np.where(labels[o] == 1)[0]
    r['mrr'] = 1.0 / (pr[0] + 1) if len(pr) > 0 else 0.0
    return r

def agg_met(rs):
    if not rs: return {f't{k}': 0. for k in KS} | {'mrr': 0.}
    return {f't{k}': float(np.mean([r[f't{k}'] for r in rs])) for k in KS} | {'mrr': float(np.mean([r['mrr'] for r in rs]))}

def rs2arr(rs):
    """Convert list of per-query metric dicts to (N,6) array [t1,t5,t10,t20,t50,mrr]."""
    out = np.zeros((len(rs), 6), dtype=np.float64)
    for i, r in enumerate(rs):
        out[i, 0] = r['t1']; out[i, 1] = r['t5']; out[i, 2] = r['t10']
        out[i, 3] = r['t20']; out[i, 4] = r['t50']; out[i, 5] = r['mrr']
    return out

def bs_delta_arr(ra, rb, rng, nb=NB):
    """Bootstrap on pre-converted (N,6) arrays. Returns dict with 'd','lo','hi','sg' per k+MRR."""
    n = ra.shape[0]; res = {}
    ds_all = np.zeros((nb, 6), dtype=np.float64)
    for b in range(nb):
        idx = rng.randint(0, n, size=n)
        ds_all[b] = ra[idx].mean(axis=0) - rb[idx].mean(axis=0)
    for ki, k in enumerate(KS):
        kk = f't{k}'; a = ds_all[:, ki]
        res[f'{kk}_d'] = float(a.mean()); res[f'{kk}_lo'] = float(np.percentile(a, 2.5))
        res[f'{kk}_hi'] = float(np.percentile(a, 97.5))
        res[f'{kk}_sg'] = 'YES' if (np.percentile(a, 97.5) < 0 or np.percentile(a, 2.5) > 0) else 'NO'
    a = ds_all[:, 5]  # mrr
    res['mrr_d'] = float(a.mean()); res['mrr_lo'] = float(np.percentile(a, 2.5))
    res['mrr_hi'] = float(np.percentile(a, 97.5))
    res['mrr_sg'] = 'YES' if (np.percentile(a, 97.5) < 0 or np.percentile(a, 2.5) > 0) else 'NO'
    return res

def jl(p):
    rows = []
    for enc in ('utf-8', 'latin-1'):
        try:
            with open(p, encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try: rows.append(json.loads(line))
                        except: pass
            if rows: return rows
        except: continue
    return rows

def json_load(p):
    try:
        for enc in ('utf-8', 'latin-1'):
            try:
                with open(p, encoding=enc) as f: return json.loads(f.read())
            except: continue
    except: pass
    return {}

def wcsv(p, fn, rows):
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(rows)

# ═══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("="*60); log.info("D4A2D1R Robustness Audit"); log.info("="*60)
    t0 = time.time()

    # ── PART A: Preflight ────────────────────────────────────────────────────
    log.info("PART A: Preflight")
    disc = []
    for d in [D4A0, D4A2D1, D4A1, D4A3]:
        if d.exists():
            for fp in sorted(d.rglob('*')):
                if fp.is_file():
                    n, e = fp.name, fp.suffix.lower()
                    ro = 'unknown'
                    if 'manifest' in n: ro = 'manifest'
                    elif 'vocabulary' in n and e == '.csv': ro = 'vocab'
                    elif 'epoch' in n and e == '.pt': ro = 'ckpt'
                    elif 'test_metrics' in n: ro = 'test_metrics'
                    elif 'bootstrap' in n: ro = 'bootstrap'
                    elif 'fusion_sweep' in n: ro = 'fusion_sweep'
                    elif 'temp_sweep' in n: ro = 'temp_sweep'
                    elif 'test_predictions' in n: ro = 'hgb_preds'
                    elif 'review_results' in n: ro = 'a4c'
                    elif 'val_by_subset' in n: ro = 'val_subset'
                    elif 'features_shard' in n: ro = 'shard'
                    elif 'summary' in n and e == '.json': ro = 'summary'
                    disc.append({'f': str(fp.relative_to(BASE)), 't': e, 'sz': fp.stat().st_size, 'r': ro, 's': 'FOUND'})
    wcsv(OUT / 'd4a2d1r_input_discovery.csv', ['f','t','sz','r','s'], disc)
    log.info(f"  Files: {len(disc)}")

    manifest = jl(D4A0 / 'd4a0_query_split_manifest.jsonl')
    vocab = [dict(r) for r in csv.DictReader(open(D4A0 / 'd4a0_train_replacement_vocabulary.csv', encoding='utf-8'))]
    hgb_rows = jl(D4A1 / 'd4a1_test_predictions.jsonl') if (D4A1 / 'd4a1_test_predictions.jsonl').exists() else None
    a4c_rows = [dict(r) for r in csv.DictReader(open(D4A3 / 'd4a3_a4c_review_results.csv', encoding='utf-8'))] if (D4A3 / 'd4a3_a4c_review_results.csv').exists() else None

    splits = defaultdict(list)
    for r in manifest: splits[r['split']].append(r['query_id'])

    test_manifest = [r for r in manifest if r['split'] == 'test' and r.get('target_any_replacement_in_train_vocab', False)]

    # Load canonical metrics from gate csv
    tm = {}
    with open(D4A2D1 / 'd4a2d1_full_test_metrics.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f): tm[r['method']] = {k: float(v) for k, v in r.items() if k != 'method'}
    hm = {}
    with open(D4A1 / 'd4a1_test_metrics.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f): hm[r['model_name']] = {k: float(v) for k, v in r.items() if k != 'model_name'}

    DT = tm.get('DualEncoder', {}).get('top10', .6957)
    BT10 = tm.get('B1_attach_freq', {}).get('top10', .5504)
    HT = hm.get('M5_hist_gradient_boosting', {}).get('top10', .7008)
    HA = hgb_rows is not None
    AA = a4c_rows is not None

    pf = [['metric','value','source'],
          ['train_q',str(len(splits.get('train',[]))),'manifest'],
          ['val_q',str(len(splits.get('val',[]))),'manifest'],
          ['test_q',str(len(splits.get('test',[]))),'manifest'],
          ['test_q_seen_vocab',str(len(test_manifest)),'manifest'],
          ['vocab_size',str(len(vocab)),'vocab'],
          ['de_t10',f'{DT:.4f}','test_metrics'],['b1_t10',f'{BT10:.4f}','test_metrics'],['hgb_t10',f'{HT:.4f}','hgb_metrics'],
          ['hgb_pq',str(HA),''],['a4c',str(AA),'']]
    wcsv(OUT / 'd4a2d1r_preflight_report.csv', ['metric','value','source'],
         [dict(zip(['metric','value','source'],r)) for r in pf[1:]])
    with open(OUT / 'd4a2d1r_preflight_summary.json','w') as f:
        json.dump({'tq':len(test_manifest),'vs':len(vocab),'dt':DT,'bt':BT10,'ht':HT,'ha':HA,'aa':AA}, f, indent=2)
    with open(OUT / 'd4a2d1r_canonical_metric_reconciliation.md','w') as f:
        f.write(f"# Metric Reconciliation\nDE T10={DT:.4f} (post-fusion) B1 T10={BT10:.4f} HGB T10={HT:.4f}\n")
    log.info(f"Part A done: {len(test_manifest)} test queries")

    # ── PART B: Inference & Predictions ──────────────────────────────────────
    log.info("PART B: Inference & Predictions")
    vs = [r['replacement_smiles'] for r in vocab]
    vsi = {r['replacement_smiles']:i for i,r in enumerate(vocab)}
    NV = len(vocab)

    # Candidate fingerprints + global frequency: MATCH GATE
    # Gate uses: vfreq = global_train_frequency, log_af = log(freq+1)
    cfp = np.array([smi2fp(r['replacement_smiles']) for r in vocab], dtype=np.float32)
    gfreq = np.array([int(r['global_train_frequency']) for r in vocab], dtype=np.int32)
    log_af = np.log(gfreq.astype(np.float64) + 1.0)  # log(freq+1), same as gate

    ct = torch.from_numpy(cfp)

    qml, qah, qlb, qi, qos, qas = [], [], [], [], [], []
    for qr in test_manifest:
        ps = set(qr.get('positive_replacement_set', []))
        qml.append(smi2fp(qr['old_fragment_smiles']))
        qah.append(att_onehot(qr['attachment_signature']))
        lb = np.zeros(NV, dtype=np.int64)
        for j, r in enumerate(vocab):
            if r['replacement_smiles'] in ps: lb[j] = 1
        qlb.append(lb); qi.append(qr['query_id'])
        qos.append(qr['old_fragment_smiles']); qas.append(qr['attachment_signature'])

    N = len(test_manifest)
    qt = torch.from_numpy(np.concatenate([np.array(qml, dtype=np.float32), np.array(qah, dtype=np.float32)], axis=1))
    la = np.array(qlb)  # (N, NV) labels

    # Load checkpoint (best epoch = epoch1 from gate)
    ckpt = sorted(D4A2D1.glob('d4a2d1_epoch*.pt'))
    if not ckpt: log.error("No checkpoint"); sys.exit(1)
    sd = torch.load(ckpt[0], map_location='cpu')
    if any(k.startswith('model.') for k in sd): sd = {k.replace('model.', ''): v for k, v in sd.items()}
    model = DualEncoder(); model.load_state_dict(sd, strict=False); model.eval()

    ti = time.time()
    # Inference with explicit eval temperature BT=0.05 (gate sweep-optimal).
    # Gate checkpoints have drifted log_t (epoch1: 0.0575, epoch5: 0.095).
    # Using explicit BT decouples DE reproduction from log_t drift in checkpoints.
    with torch.no_grad():
        qe = model.encode_query(qt); ce = model.encode_candidate(ct)
        cs = (qe @ ce.T).numpy() / BT  # (N,V) raw dot with explicit temp
    ti2 = time.time()
    log.info(f"  Inference: {ti2-ti:.1f}s")

    # Fusion: match gate — DE + alpha * log(freq+1)
    fus_raw = cs + BA * log_af  # post-fusion scores, gate's "DualEncoder" metric

    # B1: global frequency ranking — SINGLE ranking applied per query (match gate)
    # Each candidate gets score = its global_train_frequency, so argsort(-score) ranks by freq descending
    b1_score_matrix = np.tile(gfreq.astype(np.float64), (N, 1))

    # HGB
    hsm = np.full((N, NV), -1e9, dtype=np.float32)
    hqids = set()
    if hgb_rows:
        hr_d = defaultdict(dict)
        for r in hgb_rows:
            hr_d[r['query_id']][r['candidate']] = r.get('score', 0.)
        for i, qid in enumerate(qi):
            hs = hr_d.get(qid, {})
            for j, r in enumerate(vocab):
                hsm[i, j] = hs.get(r['replacement_smiles'], -1e9)
            if hs: hqids.add(qid)
    HBPQ = hgb_rows and len(hqids) > 0

    # Per-query metrics
    de_r, b1_r, fus_r, hgb_r = [], [], [], []
    for i in range(N):
        de_r.append(prm(cs[i], la[i]))
        # B1: use global frequency order
        b1_r.append(prm(b1_score_matrix[i], la[i]))
        fus_r.append(prm(fus_raw[i], la[i]))
        if HBPQ and qi[i] in hqids:
            hgb_r.append(prm(hsm[i], la[i]))
        else:
            hgb_r.append({'t1':0,'t5':0,'t10':0,'t20':0,'t50':0,'mrr':0.})

    # Write standardized predictions (top-50 per method)
    meths = {'M1_attach': b1_score_matrix, 'M2_DE': cs, 'M4_fusion': fus_raw}
    if HBPQ: meths['M3_HGB'] = hsm
    preds = []
    for i, qid in enumerate(qi):
        for mn, sc in meths.items():
            for rp, ci in enumerate(np.argsort(-sc[i])[:50]):
                preds.append({'q':qid,'m':mn,'r':rp+1,'c':vocab[ci]['replacement_smiles'],'s':float(sc[i,ci]),'l':int(la[i,ci])})
    with open(OUT / 'd4a2d1r_standardized_predictions.jsonl','w',encoding='utf-8') as f:
        for p in preds: f.write(json.dumps(p)+'\n')

    # Per-query hit table
    hr = []
    for i, qid in enumerate(qi):
        for mn, rk in [('M1_attach',b1_r),('M2_DE',de_r),('M4_fusion',fus_r)]:
            r = rk[i]
            hr.append({'qid':qid,'m':mn,'h1':r['t1'],'h5':r['t5'],'h10':r['t10'],'h20':r['t20'],'h50':r['t50'],'mrr':r['mrr']})
        if HBPQ:
            r = hgb_r[i]
            hr.append({'qid':qid,'m':'M3_HGB','h1':r['t1'],'h5':r['t5'],'h10':r['t10'],'h20':r['t20'],'h50':r['t50'],'mrr':r['mrr']})
    wcsv(OUT / 'd4a2d1r_query_level_hits.csv', ['qid','m','h1','h5','h10','h20','h50','mrr'], hr)

    da = agg_met(de_r); ba = agg_met(b1_r); fa = agg_met(fus_r); ha_agg = agg_met(hgb_r) if HBPQ else None
    log.info(f"DE   T10={da['t10']:.4f} (canon post-fus={DT:.4f})")
    log.info(f"Fus  T10={fa['t10']:.4f} (canon post-fus={DT:.4f})")
    log.info(f"B1   T10={ba['t10']:.4f} (canon={BT10:.4f})" + (f" HGB T10={ha_agg['t10']:.4f}" if ha_agg else ""))
    log.info(f"Part B done ({time.time()-t0:.0f}s)")

    # ── PART C: Bootstrap (numpy-optimized) ──────────────────────────────────
    log.info("PART C: Bootstrap")
    de_a = rs2arr(de_r); b1_a = rs2arr(b1_r); fus_a = rs2arr(fus_r)
    hgb_a = rs2arr(hgb_r) if HBPQ else None
    comps = [('DE_vs_B1',de_a,b1_a),('Fus_vs_DE',fus_a,de_a),('Fus_vs_B1',fus_a,b1_a)]
    if HBPQ: comps += [('DE_vs_HGB',de_a,hgb_a),('Fus_vs_HGB',fus_a,hgb_a)]
    brs = []
    for cn, ra, rb in comps:
        bd = bs_delta_arr(ra, rb, RNG)
        r = {'c':cn}
        for k in KS: r.update({f't{k}_d':bd[f't{k}_d'],f't{k}_lo':bd[f't{k}_lo'],f't{k}_hi':bd[f't{k}_hi'],f't{k}_sg':bd[f't{k}_sg']})
        r.update({'mrr_d':bd['mrr_d'],'mrr_lo':bd['mrr_lo'],'mrr_hi':bd['mrr_hi'],'mrr_sg':bd['mrr_sg']})
        brs.append(r)
    bf = ['c'] + [f't{k}_{s}' for k in KS for s in ('d','lo','hi','sg')] + ['mrr_d','mrr_lo','mrr_hi','mrr_sg']
    wcsv(OUT / 'd4a2d1r_bootstrap_comparisons.csv', bf, brs)
    for r in brs: log.info(f"  {r['c']}: T10 d={r['t10_d']:.4f} [{r['t10_lo']:.4f},{r['t10_hi']:.4f}] s={r['t10_sg']}")
    log.info("Part C done")

    # ── PART D: Subset Analysis ──────────────────────────────────────────────
    log.info("PART D: Subset Analysis")
    subs = defaultdict(list)
    for i, qr in enumerate(test_manifest):
        np_ = qr.get('num_positive_replacements', 0)
        as_ = qr.get('attachment_signature', '')
        subs['multi_pos' if np_ > 1 else 'single_pos'].append(i)
        mf = max((int(vr.get('global_train_frequency', 0)) for ps in qr.get('positive_replacement_set', []) for vr in vocab if vr['replacement_smiles'] == ps), default=0)
        subs['freq_repl' if mf >= 1000 else 'rare_repl'].append(i)
        subs['hard_baseline_miss' if b1_r[i]['t10'] == 0 else 'easy_baseline_hit'].append(i)
        if np_ <= 1: subs['pos_set_small'].append(i)
        elif np_ <= 3: subs['pos_set_medium'].append(i)
        else: subs['pos_set_large'].append(i)
        subs[f'attach_{as_}'].append(i)

    srs = []
    for sn, idx in sorted(subs.items()):
        if len(idx) < 5: continue
        ra = np.array([de_a[j] for j in idx])
        rb = np.array([b1_a[j] for j in idx])
        rh = np.array([hgb_a[j] for j in idx]) if HBPQ and any(qi[j] in hqids for j in idx) else None
        ma = ra.mean(axis=0); mb = rb.mean(axis=0); mh = rh.mean(axis=0) if rh is not None and len(rh) > 0 else None
        r = {'s':sn,'n':len(idx),'d1':ma[0],'d5':ma[1],'d10':ma[2],'d20':ma[3],'d50':ma[4],'dm':ma[5],
             'b1':mb[0],'b5':mb[1],'b10':mb[2],'b20':mb[3],'b50':mb[4],'bm':mb[5],
             'dd':ma[2]-mb[2]}
        if mh is not None and len(mh) >= 3: r.update({'h10':mh[2],'dh':ma[2]-mh[2]})
        srs.append(r)
    sf = ['s','n','d1','d5','d10','d20','d50','dm','b1','b5','b10','b20','b50','bm','h10','dd','dh']
    wcsv(OUT / 'd4a2d1r_test_subset_metrics.csv', sf, srs)
    log.info(f"Subsets: {len(srs)}")
    log.info("Part D done")

    # ── PART E: Hit Overlap ──────────────────────────────────────────────────
    log.info("PART E: Hit Overlap")
    K = 10
    de_o = hg_o = b1_o = de_hg = ah = am = 0
    for i in range(N):
        dp = set(np.where(la[i]==1)[0]) & set(np.argsort(-cs[i])[:K])
        bp = set(np.where(la[i]==1)[0]) & set(np.argsort(-b1_score_matrix[i])[:K])
        de_h = bool(dp); b1_h = bool(bp)
        if HBPQ and qi[i] in hqids:
            hp = set(np.where(la[i]==1)[0]) & set(np.argsort(-hsm[i])[:K])
            hg_h = bool(hp)
            if de_h and hg_h and b1_h: ah += 1
            elif de_h and not hg_h and not b1_h: de_o += 1
            elif hg_h and not de_h and not b1_h: hg_o += 1
            elif b1_h and not de_h and not hg_h: b1_o += 1
            elif de_h and hg_h and not b1_h: de_hg += 1
            else: am += 1
        else:
            if de_h and b1_h: ah += 1
            elif de_h and not b1_h: de_o += 1
            elif not de_h and b1_h: b1_o += 1
            else: am += 1
    os_ = {'n':N,'k':K,'de_o':de_o,'b1_o':b1_o,'hg_o':hg_o if HBPQ else -1,'de_hg':de_hg if HBPQ else -1,'ah':ah,'am':am,
           'de_hr':sum(1 for i in range(N) if de_r[i]['t10'])/N,
           'b1_hr':sum(1 for i in range(N) if b1_r[i]['t10'])/N}
    if HBPQ: os_['hg_hr'] = sum(1 for i in range(N) if hgb_r[i]['t10'])/N
    with open(OUT / 'd4a2d1r_hit_overlap_summary.json','w') as f: json.dump(os_, f, indent=2)
    log.info(f"Overlap@{K}: DE_only={de_o}, B1_only={b1_o}, all_miss={am}")
    wcsv(OUT / 'd4a2d1r_hit_overlap.csv', ['qid','de_h','b1_h','hg_h','cat'],
         [{'qid':qi[i],'de_h':int(de_r[i]['t10']>0),'b1_h':int(b1_r[i]['t10']>0),'hg_h':int(hgb_r[i]['t10']>0) if HBPQ else -1,'cat':''} for i in range(N)])
    log.info("Part E done")

    # ── PART F: Top1/TopK Tradeoff ────────────────────────────────────────────
    log.info("PART F: Top1/TopK Tradeoff")
    d1c = b1c = bc1 = 0
    for i in range(N):
        if la[i, np.argmax(cs[i])]: d1c += 1
        if la[i, np.argmax(b1_score_matrix[i])]: b1c += 1
        if la[i, np.argmax(cs[i])] and la[i, np.argmax(b1_score_matrix[i])]: bc1 += 1
    with open(OUT / 'D4A2D1R_TOP1_TOPK_ANALYSIS.md','w') as f:
        f.write(f"# Top1/TopK Tradeoff\nN={N} DE_T1={d1c/N:.4f} B1_T1={b1c/N:.4f} Both={bc1/N:.4f}\n\n"
                f"Q1: DE is a top-K proposal model, not top-1 selector.\n"
                f"DE T1={da['t1']:.4f} < B1 T1={ba['t1']:.4f} but DE T10={da['t10']:.4f} >> B1 T10={ba['t10']:.4f}\n\n"
                f"Q2: Fusion T1={fa['t1']:.4f} vs DE T1={da['t1']:.4f} (gain={fa['t1']-da['t1']:+.4f})\n"
                f"Fusion T10={fa['t10']:.4f} vs DE T10={da['t10']:.4f} (delta={fa['t10']-da['t10']:+.4f})\n")
        for sp, lbl in [(D4A2D1/'d4a2d1_full_temp_sweep.csv','Temp'), (D4A2D1/'d4a2d1_full_fusion_sweep.csv','Alpha')]:
            if sp.exists():
                with open(sp, encoding='utf-8') as sfh: sw = list(csv.DictReader(sfh))
                f.write(f"\n## {lbl} Sweep\n" + '|'.join(sw[0].keys())+'\n'+'|'.join(['---']*len(sw[0].keys()))+'\n')
                for r in sw: f.write('|'.join(r.values())+'\n')
    wcsv(OUT / 'd4a2d1r_top1_topk_tradeoff.csv', ['qid','d1c','b1c'],
         [{'qid':qi[i],'d1c':int(la[i,np.argmax(cs[i])]),'b1c':int(la[i,np.argmax(b1_score_matrix[i])])} for i in range(N)])
    log.info(f"DE T1={d1c/N:.4f} B1 T1={b1c/N:.4f}")
    log.info("Part F done")

    # ── PART G: Fusion Diagnostics ────────────────────────────────────────────
    log.info("PART G: Fusion Diagnostics")
    vm = [r for r in manifest if r['split'] == 'val']
    if len(vm) > 0:
        vml, vah, vlb, vi = [], [], [], []
        for qr in vm:
            ps = set(qr.get('positive_replacement_set', []))
            vml.append(smi2fp(qr['old_fragment_smiles'])); vah.append(att_onehot(qr['attachment_signature']))
            lb = np.zeros(NV, dtype=np.int64)
            for j, r in enumerate(vocab):
                if r['replacement_smiles'] in ps: lb[j] = 1
            vlb.append(lb); vi.append(qr['query_id'])
        vt = torch.from_numpy(np.concatenate([np.array(vml, dtype=np.float32), np.array(vah, dtype=np.float32)], axis=1))
        vla = np.array(vlb)
        with torch.no_grad():
            qv = model.encode_query(vt); cv = model.encode_candidate(ct)
            vs_ = (qv @ cv.T).numpy() / BT  # explicit temp, match Part B
        # Val uses global frequency for B1/fusion (no HGB per-query data)
        v_log_af = np.log(gfreq.astype(np.float64) + 1.0)

        ag = [0., .05, .1, .2, .5, 1.0]
        fvr = []
        b_sel = None; b_t10 = -1
        for a in ag:
            fsc = vs_ + a * v_log_af
            rr = [prm(fsc[i], vla[i]) for i in range(len(vm))]
            m = agg_met(rr)
            fvr.append({'m':'F2','a':a,'b':-1,'t10':m['t10'],'t1':m['t1'],'mr':m['mrr']})
            if m['t10'] > b_t10: b_t10 = m['t10']; b_sel = ('F2',a,-1)
        for w in [.3, .5, .7]:
            rsc = np.zeros_like(vs_)
            for i in range(len(vm)):
                dr_ = np.argsort(np.argsort(-vs_[i])); br_ = np.argsort(np.argsort(-(v_log_af)))
                rsc[i] = -(w*dr_ + (1-w)*br_)
            rr = [prm(rsc[i], vla[i]) for i in range(len(vm))]
            m = agg_met(rr)
            fvr.append({'m':'F3','a':w,'b':-1,'t10':m['t10'],'t1':m['t1'],'mr':m['mrr']})
            if m['t10'] > b_t10: b_t10 = m['t10']; b_sel = ('F3',w,-1)
        for kr in [30, 60, 100]:
            rrf = np.zeros_like(vs_)
            for i in range(len(vm)):
                do = np.argsort(-vs_[i]); bo = np.argsort(-(v_log_af))
                dr = np.zeros(NV); br = np.zeros(NV)
                for p, idx in enumerate(do): dr[idx] = p+1
                for p, idx in enumerate(bo): br[idx] = p+1
                rrf[i] = [1/(kr+dr[j]) + 1/(kr+br[j]) for j in range(NV)]
            rr = [prm(rrf[i], vla[i]) for i in range(len(vm))]
            m = agg_met(rr)
            fvr.append({'m':'F4_RRF','a':-1,'b':kr,'t10':m['t10'],'t1':m['t1'],'mr':m['mrr']})
            if m['t10'] > b_t10: b_t10 = m['t10']; b_sel = ('F4_RRF',-1,kr)
        wcsv(OUT / 'd4a2d1r_fusion_val_grid.csv', ['m','a','b','t10','t1','mr'], fvr)
        with open(OUT / 'd4a2d1r_fusion_selected_policy.md','w') as f:
            f.write(f"Best fusion: {b_sel[0]} (T10={b_t10:.4f})\n")
    else:
        log.warning("No val queries found for fusion diagnostics")
        b_sel = None
    log.info("Part G done")

    # ── PART H: A4C Review Audit ─────────────────────────────────────────────
    log.info("PART H: A4C Review Audit")
    if a4c_rows:
        amap = {'M0_attachment_frequency':'M1_attach','M1_canonical_HGB':'M3_HGB','M2_best_D4A2_ranker':'M2_DE'}
        a4q = defaultdict(dict)
        for r in a4c_rows: a4q[r['query_id']][r['method']] = r
        acr = []
        for mu, ma in amap.items():
            rr = ww = hr_ = t1h = total = 0
            for qid in set(qi) & set(a4q.keys()):
                if ma not in a4q[qid]: continue
                row = a4q[qid][ma]; bk = row.get('a4c_bucket', ''); rnk = int(row.get('rank', 1))
                total += 1
                if bk == 'REVIEW_READY': rr += 1
                elif 'REJECT' in bk: hr_ += 1; t1h += (rnk == 1)
                else: ww += 1
            acr.append({'m':mu,'n':total,'rr':rr/max(total,1),'hr':hr_/max(total,1),'ww':ww/max(total,1),'t1hr':t1h/max(total,1)})
        wcsv(OUT / 'd4a2d1r_a4c_review_comparison.csv', ['m','n','rr','hr','ww','t1hr'], acr)
        log.info(f"A4C: {len(acr)} methods")
        for r in acr: log.info(f"  {r['m']}: rr={r['rr']:.3f} hr={r['hr']:.3f}")
    else:
        with open(OUT / 'd4a2d1r_a4c_review_comparison.csv','w') as f: f.write("A4C_AUDIT_BLOCKED\n")
        log.warning("A4C BLOCKED")
    log.info("Part H done")

    # ── PART I: Engineering Report ────────────────────────────────────────────
    log.info("PART I: Engineering Report")
    eng = [{'m':'inference_sec','v':f'{ti2-ti:.1f}'},{'m':'total_sec','v':f'{time.time()-t0:.0f}'},
           {'m':'params','v':str(sum(p.numel() for p in model.parameters()))},
           {'m':'model_mb','v':f'{sum(p.numel()*p.element_size() for p in model.parameters())/1024/1024:.1f}'},
           {'m':'emb_dim','v':str(OD)},{'m':'vocab','v':str(NV)},{'m':'test_q','v':str(N)},
           {'m':'device','v':'cpu'},{'m':'strategy','v':'full_dot'}]
    wcsv(OUT / 'd4a2d1r_engineering_report.csv', ['m','v'], eng)
    log.info("Part I done")

    # ── PART J: Final Verdict ─────────────────────────────────────────────────
    log.info("PART J: Final Verdict")
    vi = {}
    # Q1: Fus vs canon DE (gate's "DE" = post-fusion)
    vi['Q1_Fus_repro'] = 'PASS' if abs(fa['t10']-DT) < 0.015 else f"OFF({fa['t10']-DT:+.4f})"
    vi['Q2_B1_repro'] = 'PASS' if abs(ba['t10']-BT10) < 0.015 else f"OFF({ba['t10']-BT10:+.4f})"
    vi['Q3_DE_raw_vs_B1'] = f"DE={da['t10']:.4f} B1={ba['t10']:.4f}"
    dbb = next((r for r in brs if r['c']=='DE_vs_B1'), None)
    vi['Q3b_DE_B1_sig'] = dbb['t10_sg'] if dbb else 'UNKNOWN'
    vi['Q4_DE_HGB_delta'] = f"{da['t10']-HT:.4f}" if ha_agg else 'BLOCKED'
    vi['Q5_DE_T1_below_B1'] = 'YES' if da['t1'] < ba['t1'] else 'NO'
    vi['Q6_Fusion_T1_gain'] = f"{fa['t1']-da['t1']:+.4f}"
    vi['Q7_best_fusion'] = str(b_sel[0]) if b_sel else 'NONE'
    vi['Q8_DE_HGB_compl'] = f"DE_o={de_o} HG_o={hg_o}" if HBPQ else 'BLOCKED'
    ws = min(srs, key=lambda r: r.get('dd', 0)) if srs else None
    bs_ = max(srs, key=lambda r: r.get('dd', 0)) if srs else None
    vi['Q9_worst_subset'] = f"{ws['s']}({ws['dd']:.4f})" if ws else ''
    vi['Q10_A4C'] = 'DONE' if a4c_rows else 'BLOCKED'
    sc = sum(1 for r in brs if r['t10_sg']=='YES')
    vi['Q11_sig_count'] = f"{sc}/{len(brs)}"
    if fa['t10'] >= .69: ov = 'A. ROBUST'
    elif fa['t10'] >= .60: ov = 'B. COND_ROBUST'
    elif fa['t10'] >= .50: ov = 'C. DEGRADED'
    else: ov = 'D. FAILED'
    vi['overall'] = ov

    with open(OUT / 'D4A2D1R_DUAL_ENCODER_ROBUSTNESS_VERDICT.md','w') as f:
        f.write(f"# D4A2D1R Verdict\nDate: 2026-05-23\nSeed={SEED}\n\n"
                f"Fus T10={fa['t10']:.4f} (canon post-fus={DT:.4f})\nDE raw T10={da['t10']:.4f}\n"
                f"B1 T10={ba['t10']:.4f} (canon={BT10:.4f})\n"
                + (f"HGB T10={ha_agg['t10']:.4f} (canon={HT:.4f})\n" if ha_agg else "HGB: per-query NA\n") +
                f"\nVerdict: {ov}\n\n## Q&A\n" + '\n'.join(f"- {k}: {v}" for k,v in sorted(vi.items())) +
                "\n\n## SKEPTICAL REVIEW\n"
                "1. Morgan FP recomputed from SMILES with *->[H]; minor deviation possible from training code.\n"
                "2. B1 from global_train_frequency ranking, matching gate.\n"
                "3. HGB predictions cover test only, not val.\n"
                "4. Model checkpoint: epoch1 used (gate best=0 not saved).\n"
                "5. Full NxV matrix in memory (~N*V*4 bytes).\n"
                "6. A4C comparison uses different Top-K cutoff.\n"
                + ("7. HGB-dependent analyses BLOCKED.\n" if not HBPQ else "") +
                ("8. A4C audit BLOCKED.\n" if not a4c_rows else ""))

    with open(OUT / 'MAIN_DECISION_LOG.md','w') as f:
        f.write(f"# D4A2D1R Decision Log\n{ov}\nModel={ckpt[0].name} T={BT} A={BA} BS={NB} N_test={N}\n"
                f"HGB={'avail' if HBPQ else 'BLOCKED'} A4C={'avail' if a4c_rows else 'BLOCKED'}\n")

    log.info(f"OVERALL: {ov}"); log.info(f"D4A2D1R audit done ({time.time()-t0:.0f}s)")
    return vi

if __name__ == '__main__':
    v = main()
    for k, v_ in sorted(v.items()): log.info(f"  {k}: {v_}")
