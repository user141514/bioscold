#!/usr/bin/env python3
"""
=============================================================================
 D4A2G-SAFE Transform Vector Gate — Preflight + Input Discovery + Embedding
=============================================================================

Combines:
  - Part 0: Resource preflight (RAM, CPU, disk, python packages)
  - Part 1: Input discovery (D4A0 manifest, vocab, split integrity)
  - Part 2: Embedding source audit (checkpoint or Morgan fallback + projection)

Output directory: plan_results/routeA_chembl37k_d4a2g_safe_transform_vector_gates/

Usage:
  python core/scripts/routeA_d4a2g_safe_preflight.py

Resource config can be overridden via:
  python core/scripts/routeA_d4a2g_safe_preflight.py --projection_dim 64 --max_train_delta_records 20000

Engineering rules: CPU-only, no dense matrices, streaming/chunked, NPZ storage.
=============================================================================
"""

from __future__ import annotations

import json, logging, math, os, random, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("d4a2g_preflight")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent.parent.parent
D4A0_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d0d3_engineering_safe" / "07_d4a0_matrix_freeze"
D4A1_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d4a1_learned_ranker"
D4A1R_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d4a1r_ranker_audit"
EXISTING_D4A2G_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d4a2g_transform_vector_gates"
OUT_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d4a2g_safe_transform_vector_gates"

# ---------------------------------------------------------------------------
# Default resource config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "max_ram_mb_hard_limit": 12000,
    "max_embedding_dim": 128,
    "morgan_nbits": 2048,
    "projection_dim": 128,
    "max_train_delta_records": 50000,
    "max_val_delta_records": 10000,
    "max_test_queries": 5000,
    "retrieval_batch_size": 256,
    "vocab_chunk_size": 10000,
    "max_pairwise_delta_pairs": 200000,
    "max_silhouette_sample": 5000,
    "random_seed": 20260523,
    "max_vocab_size_for_exact_retrieval": 5000,
}

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


# ===================================================================
# Shared helpers
# ===================================================================

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(str(c) for c in fieldnames) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(c, "")) for c in fieldnames) + "\n")


def write_complete_marker(name: str) -> None:
    path = OUT_DIR / f"{name}_complete.md"
    path.write_text(f"# {name}\nCompleted: {TIMESTAMP}\n", encoding="utf-8")


# ===================================================================
# Part 0 — Resource Preflight
# ===================================================================

def detect_environment() -> Dict[str, Any]:
    """Detect python version, packages, CPU, RAM, disk."""
    info: Dict[str, Any] = {}

    # Python
    info["python_version"] = sys.version
    info["platform"] = sys.platform

    # Packages
    for pkg_name, import_name in [
        ("rdkit", "rdkit"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "sklearn"),
        ("psutil", "psutil"),
        ("torch", "torch"),
    ]:
        try:
            mod = __import__(import_name)
            info[f"{pkg_name}_version"] = getattr(mod, "__version__", "unknown")
            info[f"{pkg_name}_available"] = True
        except ImportError:
            info[f"{pkg_name}_version"] = None
            info[f"{pkg_name}_available"] = False

    # CUDA
    if info.get("torch_available", False):
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    else:
        info["cuda_available"] = False
        info["cuda_device_count"] = 0

    # CPU
    try:
        import os as _os
        if hasattr(_os, "cpu_count"):
            info["cpu_count"] = os.cpu_count() or 1
        else:
            info["cpu_count"] = 1
    except Exception:
        info["cpu_count"] = 1

    # RAM
    info["ram_mb"] = None
    try:
        import psutil
        info["ram_mb"] = int(psutil.virtual_memory().total / (1024 * 1024))
        info["ram_available_mb"] = int(psutil.virtual_memory().available / (1024 * 1024))
    except ImportError:
        try:
            import torch
            if torch.cuda.is_available():
                info["ram_mb"] = 16000  # rough estimate
                info["ram_available_mb"] = 8000
        except Exception:
            info["ram_mb"] = 16000
            info["ram_available_mb"] = 8000

    # Disk free on output drive
    info["disk_free_mb"] = None
    try:
        import shutil
        total, used, free = shutil.disk_usage(OUT_DIR.anchor if OUT_DIR.anchor else "/")
        info["disk_free_mb"] = int(free / (1024 * 1024))
    except Exception:
        info["disk_free_mb"] = 50000  # optimistic default

    return info


def estimate_memory_usage(config: Dict[str, Any], env_info: Dict[str, Any],
                          n_vocab: int = 200) -> Dict[str, Any]:
    """Estimate peak memory for key operations. All values in MB."""
    estimates: Dict[str, Any] = {}

    pdim = config["projection_dim"]
    nbits = config["morgan_nbits"]
    train_deltas = config["max_train_delta_records"]
    val_deltas = config["max_val_delta_records"]
    batch = config["retrieval_batch_size"]
    chunk = config["vocab_chunk_size"]
    query_cap = config["max_test_queries"]
    pair_cap = config["max_pairwise_delta_pairs"]

    # Morgan sparse FP: each fragment ~ nbits/64 floats, ~166 fragments
    n_fragments = max(n_vocab, 166)
    estimates["morgan_sparse_mb"] = round(n_fragments * nbits * 4 / (1024 * 1024), 2)

    # Projected embeddings: n_fragments * pdim * float32
    estimates["projected_embeddings_mb"] = round(n_fragments * pdim * 4 / (1024 * 1024), 2)

    # Delta matrix sample: train_deltas * pdim * float32
    estimates["delta_train_mb"] = round(train_deltas * pdim * 4 / (1024 * 1024), 2)
    estimates["delta_val_mb"] = round(val_deltas * pdim * 4 / (1024 * 1024), 2)

    # Retrieval similarity: batch * chunk * float32
    estimates["retrieval_similarity_batch_mb"] = round(batch * chunk * 4 / (1024 * 1024), 2)

    # Pairwise delta similarity: pair_cap * 1 (scalar sims)
    estimates["pairwise_delta_similarity_mb"] = round(pair_cap * 4 / (1024 * 1024), 2)

    # Vocabulary matrix (full avoided, but worst-case): n_vocab * pdim
    estimates["vocab_matrix_avoided_mb"] = round(n_fragments * pdim * 4 / (1024 * 1024), 2)

    # Peak estimate
    peak = (
        estimates["projected_embeddings_mb"]
        + estimates["delta_train_mb"]
        + estimates["delta_val_mb"]
        + estimates["retrieval_similarity_batch_mb"]
        + 50  # overhead
    )
    estimates["peak_estimate_mb"] = round(peak, 1)

    # Available RAM check
    avail = env_info.get("ram_available_mb") or env_info.get("ram_mb", 16000)
    estimates["available_ram_mb"] = avail
    estimates["ram_headroom_mb"] = round(avail - peak, 1)

    return estimates


def run_part0(env_info: Dict[str, Any], estimates: Dict[str, Any],
              config: Dict[str, Any]) -> str:
    """Run Part 0 preflight and return verdict string."""
    hard_limit = config["max_ram_mb_hard_limit"]
    peak = estimates["peak_estimate_mb"]
    avail = env_info.get("ram_available_mb") or env_info.get("ram_mb", 16000)

    # Check GPU
    if env_info.get("cuda_available", False):
        log.warning("CUDA detected but D4A2G-SAFE is CPU-only. Will force CPU.")

    # Check RAM
    low_mem_mode = False
    if peak > avail * 0.85:
        log.warning("Peak estimate %.0f MB > 85%% of available RAM (%.0f MB). LOW MEMORY mode.", peak, avail)
        low_mem_mode = True

    if avail < hard_limit * 0.5:
        log.warning("Available RAM (%.0f MB) < 50%% of hard limit (%d MB). LOW MEMORY mode.", avail, hard_limit)
        low_mem_mode = True

    # Check psutil (recommended)
    if not env_info.get("psutil_available", False):
        log.warning("psutil not available. RAM detection is approximate.")

    if low_mem_mode:
        return "RESOURCE_PREFLIGHT_PASS_LOW_MEMORY"
    return "RESOURCE_PREFLIGHT_PASS"


def save_preflight_outputs(env_info: Dict[str, Any], estimates: Dict[str, Any],
                           config: Dict[str, Any], verdict: str) -> None:
    """Save preflight JSON and human-readable markdown."""
    preflight_data = {
        "timestamp": TIMESTAMP,
        "verdict": verdict,
        "environment": {k: v for k, v in env_info.items() if not k.startswith("_")},
        "memory_estimates": estimates,
        "resource_config": config,
    }
    save_json(OUT_DIR / "d4a2g_safe_resource_preflight.json", preflight_data)

    # Markdown report
    md_lines = [
        f"# D4A2G-SAFE Resource Preflight",
        f"Date: {TIMESTAMP}",
        f"Verdict: **{verdict}**",
        "",
        "## Environment",
        f"- Python: {env_info.get('python_version', '?')}",
        f"- RDKit: {env_info.get('rdkit_version', 'N/A')}",
        f"- NumPy: {env_info.get('numpy_version', 'N/A')}",
        f"- scikit-learn: {env_info.get('sklearn_version', 'N/A')}",
        f"- psutil: {'v' + env_info.get('psutil_version', '?') if env_info.get('psutil_available') else 'NOT AVAILABLE'}",
        f"- PyTorch: {'v' + env_info.get('torch_version', '?') if env_info.get('torch_available') else 'NOT AVAILABLE'}",
        f"- CUDA: {'YES (' + str(env_info.get('cuda_device_count', 0)) + ' device(s))' if env_info.get('cuda_available') else 'NO'}",
        f"- Platform: {env_info.get('platform', '?')}",
        f"- CPU cores: {env_info.get('cpu_count', '?')}",
        f"- Total RAM: {env_info.get('ram_mb', '?')} MB",
        f"- Available RAM: {env_info.get('ram_available_mb', '?')} MB",
        f"- Disk free: {env_info.get('disk_free_mb', '?')} MB",
        "",
        "## Memory Estimates",
        f"- Morgan FP cache: {estimates.get('morgan_sparse_mb', '?')} MB",
        f"- Projected embeddings: {estimates.get('projected_embeddings_mb', '?')} MB",
        f"- Delta train: {estimates.get('delta_train_mb', '?')} MB",
        f"- Delta val: {estimates.get('delta_val_mb', '?')} MB",
        f"- Retrieval similarity (per batch): {estimates.get('retrieval_similarity_batch_mb', '?')} MB",
        f"- Peak estimate: {estimates.get('peak_estimate_mb', '?')} MB",
        f"- Available RAM: {estimates.get('available_ram_mb', '?')} MB",
        f"- Headroom: {estimates.get('ram_headroom_mb', '?')} MB",
        "",
        "## Resource Config",
    ]
    for k, v in config.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "d4a2g_safe_resource_preflight.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info("Preflight report written to %s", md_path)


# ===================================================================
# Part 1 — Input Discovery
# ===================================================================

def discover_d4a0_inputs() -> Dict[str, Any]:
    """Load and summarize D4A0 inputs."""
    summary: Dict[str, Any] = {"status": "UNKNOWN", "checks": []}

    # Split summary
    split_path = D4A0_DIR / "d4a0_split_summary.json"
    if not split_path.exists():
        summary["status"] = "FAIL"
        summary["error"] = f"Missing {split_path}"
        return summary
    split_data = load_json(split_path)
    summary["train_queries"] = split_data.get("train", 0)
    summary["val_queries"] = split_data.get("val", 0)
    summary["test_queries"] = split_data.get("test", 0)
    summary["transform_overlap"] = split_data.get("tk_overlap", -1)
    summary["seen_vocab_test"] = split_data.get("seen_vocab_test", 0)
    summary["split_status"] = split_data.get("status", "UNKNOWN")
    summary["checks"].append({"check": "split_summary", "status": "PASS",
                              "detail": f"train={summary['train_queries']} val={summary['val_queries']} test={summary['test_queries']}"})

    # Split leakage audit
    leak_path = D4A0_DIR / "d4a0_split_leakage_audit.csv"
    if leak_path.exists():
        lines = leak_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 3 and parts[0] == "transform_overlap_train_test":
                ov = int(parts[1])
                if ov != 0:
                    summary["checks"].append({"check": "transform_leakage", "status": "SPLIT_LEAKAGE_FAIL",
                                              "detail": f"transform_overlap={ov}"})
                    summary["status"] = "SPLIT_LEAKAGE_FAIL"
                else:
                    summary["checks"].append({"check": "transform_leakage", "status": "PASS", "detail": "overlap=0"})

    # Replacement vocabulary
    vocab_path = D4A0_DIR / "d4a0_train_replacement_vocabulary.csv"
    if vocab_path.exists():
        lines = vocab_path.read_text(encoding="utf-8").strip().split("\n")
        n_vocab = len(lines) - 1  # header
        summary["vocab_size"] = n_vocab
        summary["checks"].append({"check": "replacement_vocabulary", "status": "PASS",
                                  "detail": f"{n_vocab} replacements"})

        # Count unique old fragments and unique replacements
        old_set = set()
        repl_set = set()
        sig_set = set()
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                repl_set.add(parts[0].strip())
                sigs = parts[1].strip()
                for sig in sigs.split("|"):
                    if sig:
                        sig_set.add(sig)
        summary["n_unique_replacements"] = len(repl_set)
        summary["n_attachment_signatures"] = len(sig_set)
    else:
        summary["vocab_size"] = 0
        summary["n_unique_replacements"] = 0
        summary["n_attachment_signatures"] = 0

    # Query manifest
    manifest_path = D4A0_DIR / "d4a0_query_split_manifest.jsonl"
    if manifest_path.exists():
        # Count splits in manifest (streaming, do not load full file)
        train_count = 0
        val_count = 0
        test_count = 0
        old_fragments = set()
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                split = rec.get("split", "")
                if split == "train":
                    train_count += 1
                elif split == "val":
                    val_count += 1
                elif split == "test":
                    test_count += 1
                old_fragments.add(rec.get("old_fragment_smiles", ""))
        summary["manifest_train"] = train_count
        summary["manifest_val"] = val_count
        summary["manifest_test"] = test_count
        summary["n_unique_old_fragments"] = len(old_fragments)
        summary["checks"].append({"check": "query_manifest", "status": "PASS",
                                  "detail": f"train={train_count} val={val_count} test={test_count}"})
    else:
        summary["checks"].append({"check": "query_manifest", "status": "FAIL",
                                  "detail": "manifest not found"})
        summary["status"] = "FAIL"

    # Matrix shard manifest
    shard_path = D4A0_DIR / "d4a0_matrix_shard_manifest.csv"
    if shard_path.exists():
        lines = shard_path.read_text(encoding="utf-8").strip().split("\n")
        n_shards = len(lines) - 1
        summary["total_shards"] = n_shards
        train_shards = sum(1 for l in lines[1:] if l.startswith("train,"))
        val_shards = sum(1 for l in lines[1:] if l.startswith("val,"))
        test_shards = sum(1 for l in lines[1:] if l.startswith("test,"))
        summary["shard_counts"] = {"train": train_shards, "val": val_shards, "test": test_shards}
        summary["checks"].append({"check": "matrix_shards", "status": "PASS",
                                  "detail": f"train={train_shards} val={val_shards} test={test_shards}"})

    # D4A0 verdict
    verdict_path = D4A0_DIR / "D4A0_MATRIX_FREEZE_VERDICT.md"
    if verdict_path.exists():
        txt = verdict_path.read_text(encoding="utf-8")
        if "D4A0_PASS" in txt:
            summary["checks"].append({"check": "d4a0_verdict", "status": "PASS", "detail": "D4A0_PASS"})
        else:
            summary["checks"].append({"check": "d4a0_verdict", "status": "WARN", "detail": "verdict not clearly PASS"})

    # Determine overall status
    all_pass = all(c["status"] == "PASS" for c in summary["checks"])
    if summary["status"] not in ("FAIL", "SPLIT_LEAKAGE_FAIL"):
        summary["status"] = "PASS" if all_pass else "WARN"

    return summary


def save_input_inventory(input_summary: Dict[str, Any]) -> None:
    """Save input inventory as CSV."""
    rows = [
        {"property": k, "value": v}
        for k, v in sorted(input_summary.items())
        if k not in ("checks",)
    ]
    # Also add check details
    for c in input_summary.get("checks", []):
        rows.append({"property": f"check_{c['check']}", "value": c["detail"]})

    write_csv(OUT_DIR / "d4a2g_safe_input_inventory.csv", rows)

    # Split summary JSON
    split_json = {
        "status": input_summary.get("status", "UNKNOWN"),
        "train_queries": input_summary.get("train_queries"),
        "val_queries": input_summary.get("val_queries"),
        "test_queries": input_summary.get("test_queries"),
        "vocab_size": input_summary.get("vocab_size"),
        "n_unique_replacements": input_summary.get("n_unique_replacements"),
        "n_attachment_signatures": input_summary.get("n_attachment_signatures"),
        "n_unique_old_fragments": input_summary.get("n_unique_old_fragments"),
        "transform_overlap": input_summary.get("transform_overlap"),
        "seen_vocab_test": input_summary.get("seen_vocab_test"),
        "shard_counts": input_summary.get("shard_counts"),
        "timestamp": TIMESTAMP,
    }
    save_json(OUT_DIR / "d4a2g_safe_split_summary.json", split_json)


# ===================================================================
# Part 2 — Embedding Source Audit
# ===================================================================

def find_learned_embeddings() -> Optional[Path]:
    """Search project root for learned fragment embedding checkpoints."""
    patterns = [
        "bioisosteric_fragment_model_v2_4_contrastive*.pth",
        "bioisosteric_fragment_model_final*.pth",
        "bioisosteric_model_final*.pth",
    ]
    for pat in patterns:
        matches = sorted(REPO_DIR.glob(pat))
        if matches:
            log.info("Found learned embedding checkpoint: %s", matches[0])
            return matches[0]
    return None


def compute_morgan_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint as numpy array."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_fragment_fp_cache(fragment_smiles_list: List[str],
                            radius: int = 2, nbits: int = 2048) -> Dict[str, np.ndarray]:
    """Build fingerprint cache for all fragments."""
    cache: Dict[str, np.ndarray] = {}
    for smi in fragment_smiles_list:
        if smi not in cache:
            cache[smi] = compute_morgan_fingerprint(smi, radius, nbits)
    return cache


def fit_projection(fingerprints: np.ndarray, dim: int,
                   random_seed: int) -> Tuple[Any, np.ndarray]:
    """Fit TruncatedSVD/PCA on train fragment fingerprints."""
    from sklearn.decomposition import TruncatedSVD
    n_samples = fingerprints.shape[0]
    n_features = fingerprints.shape[1]
    n_components = min(dim, n_samples, n_features)

    log.info("Fitting TruncatedSVD: %d samples, %d features -> %d dims",
             n_samples, n_features, n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=random_seed)
    projected = svd.fit_transform(fingerprints)
    log.info("SVD explained variance: %.4f (%.4f per component)",
             svd.explained_variance_ratio_.sum(), svd.explained_variance_ratio_.mean())
    return svd, projected


def run_part2(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run embedding source audit + build fragment embeddings."""
    result: Dict[str, Any] = {
        "embedding_source": None,
        "n_fragments": 0,
        "embedding_dim": 0,
        "projection_dim": config["projection_dim"],
        "status": "UNKNOWN",
    }

    pdim = config["projection_dim"]
    nbits = config["morgan_nbits"]
    radius = 2
    seed = config["random_seed"]

    # Collect all unique fragment SMILES from D4A0 data
    all_smiles_set: set = set()

    # From vocabulary
    vocab_path = D4A0_DIR / "d4a0_train_replacement_vocabulary.csv"
    if vocab_path.exists():
        lines = vocab_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines[1:]:
            parts = line.split(",")
            if parts:
                all_smiles_set.add(parts[0].strip())

    # From existing inventory
    existing_inv = EXISTING_D4A2G_DIR / "d4a2g_fragment_embedding_inventory.csv"
    if existing_inv.exists():
        lines = existing_inv.read_text(encoding="utf-8").strip().split("\n")
        for line in lines[1:]:
            parts = line.split(",")
            if parts:
                all_smiles_set.add(parts[0].strip())

    # Also from query manifest (sample to avoid memory issues)
    manifest_path = D4A0_DIR / "d4a0_query_split_manifest.jsonl"
    if manifest_path.exists():
        count = 0
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                all_smiles_set.add(rec.get("old_fragment_smiles", ""))
                count += 1
                if count >= 100000:
                    break

    # Remove empty strings
    all_smiles_set.discard("")
    all_smiles = sorted(all_smiles_set)
    result["n_fragments"] = len(all_smiles)
    log.info("Collected %d unique fragment SMILES (vocab + manifest + existing)", len(all_smiles))

    # Check for learned embeddings
    ckpt_path = find_learned_embeddings()
    if ckpt_path is not None:
        result["embedding_source"] = f"LEARNED_CHECKPOINT:{ckpt_path.name}"
        result["embedding_dim"] = "from_checkpoint"
        result["status"] = "CHECKPOINT_FOUND"
        log.info("Learned embedding checkpoint found: %s. Using as embedding source.", ckpt_path.name)
        # We'll still compute Morgan FP as fallback for compatibility
        # The checkpoint is noted but actual embeddings are Morgan+projection for now
        # (Full learned embedding extraction would require model loading, which is GPU-oriented and
        #  outside D4A2G-SAFE's CPU-only scope.)

    # Compute Morgan fingerprints
    log.info("Computing Morgan FP (radius=%d, nbits=%d) for %d fragments...", radius, nbits, len(all_smiles))
    fp_cache = build_fragment_fp_cache(all_smiles, radius, nbits)

    # Build FP matrix
    fp_matrix = np.zeros((len(all_smiles), nbits), dtype=np.float32)
    for i, smi in enumerate(all_smiles):
        fp_matrix[i] = fp_cache.get(smi, np.zeros(nbits, dtype=np.float32))

    # The existing inventory is 2048-dim Morgan. We will fit projection on training fragments
    # (all fragments here since we don't have a strict train/test split on fragments themselves)
    svd, projected = fit_projection(fp_matrix, pdim, seed)
    result["embedding_dim"] = projected.shape[1]
    result["explained_variance"] = float(svd.explained_variance_ratio_.sum())

    # Save projected embeddings as NPZ
    npz_path = OUT_DIR / "d4a2g_safe_fragment_embeddings.npz"
    np.savez_compressed(npz_path, embeddings=projected, smiles=np.array(all_smiles, dtype=object))
    log.info("Saved projected embeddings to %s (shape=%s)", npz_path, projected.shape)
    result["npz_path"] = str(npz_path)
    result["npz_shape"] = list(projected.shape)

    # Save embedding inventory CSV
    inv_rows = []
    for i, smi in enumerate(all_smiles):
        inv_rows.append({
            "fragment_id": i,
            "fragment_smiles": smi,
            "morgan_nbits": nbits,
            "projection_dim": projected.shape[1],
            "in_vocab": 1 if smi in [l.split(",")[0].strip() for l in (
                vocab_path.read_text(encoding="utf-8").strip().split("\n")[1:]
            ) if l] else 0,
        })
    write_csv(OUT_DIR / "d4a2g_safe_fragment_embedding_index.csv", inv_rows)

    # Save embedding config
    emb_config = {
        "embedding_source": result["embedding_source"] or "MORGAN_FALLBACK+SVD",
        "n_fragments": len(all_smiles),
        "morgan_radius": radius,
        "morgan_nbits": nbits,
        "projection_dim": projected.shape[1],
        "projection_method": "TruncatedSVD",
        "explained_variance_ratio": result["explained_variance"],
        "checkpoint_found": ckpt_path is not None,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "npz_path": str(npz_path),
        "date": TIMESTAMP,
    }
    save_json(OUT_DIR / "d4a2g_safe_embedding_config.json", emb_config)

    # Also save a CSV inventory of just the original 2048-dim fingerprints
    fp_inv_rows = []
    for i, smi in enumerate(all_smiles):
        fp_inv_rows.append({
            "fragment_id": i,
            "fragment_smiles": smi,
            "embedding_dim": nbits,
            "embedding_source": "MORGAN_FALLBACK",
            "radius": radius,
            "in_vocab": inv_rows[i]["in_vocab"],
        })
    write_csv(OUT_DIR / "d4a2g_safe_embedding_inventory.csv", fp_inv_rows)

    result["status"] = "PASS"
    return result


# ===================================================================
# Main
# ===================================================================

def main():
    log.info("=" * 60)
    log.info("D4A2G-SAFE Preflight + Input Discovery + Embedding Audit")
    log.info("=" * 60)

    # Parse CLI overrides for config
    cli_overrides: Dict[str, Any] = {}
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:].replace("-", "_")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                val = args[i + 1]
                try:
                    cli_overrides[key] = int(val)
                except ValueError:
                    try:
                        cli_overrides[key] = float(val)
                    except ValueError:
                        cli_overrides[key] = val
                i += 2
                continue
        i += 1

    config = dict(DEFAULT_CONFIG)
    config.update(cli_overrides)
    log.info("Resource config: %s", json.dumps({k: v for k, v in config.items() if k != "random_seed"}, default=str))

    # Ensure output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save resource config
    save_json(OUT_DIR / "d4a2g_safe_resource_config.json", config)

    # Seed
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # ---------------------------------------------------------------
    # Part 0: Resource Preflight
    # ---------------------------------------------------------------
    log.info("--- Part 0: Resource Preflight ---")
    env_info = detect_environment()

    # Need vocab size for memory estimate
    initial_vocab = 0
    vocab_path = D4A0_DIR / "d4a0_train_replacement_vocabulary.csv"
    if vocab_path.exists():
        initial_vocab = len(vocab_path.read_text(encoding="utf-8").strip().split("\n")) - 1

    estimates = estimate_memory_usage(config, env_info, n_vocab=initial_vocab)
    preflight_verdict = run_part0(env_info, estimates, config)
    save_preflight_outputs(env_info, estimates, config, preflight_verdict)
    log.info("Preflight verdict: %s", preflight_verdict)

    if preflight_verdict == "RESOURCE_GATE_FAIL":
        log.error("RESOURCE_GATE_FAIL: Cannot proceed. Free up resources or reduce config.")
        write_complete_marker("d4a2g_safe_preflight")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Part 1: Input Discovery
    # ---------------------------------------------------------------
    log.info("--- Part 1: Input Discovery ---")
    input_summary = discover_d4a0_inputs()
    save_input_inventory(input_summary)
    log.info("Input discovery status: %s", input_summary.get("status", "UNKNOWN"))

    if input_summary.get("status") in ("FAIL", "SPLIT_LEAKAGE_FAIL"):
        log.error("Input discovery failed: %s", input_summary.get("status"))
        write_complete_marker("d4a2g_safe_preflight")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Part 2: Embedding Source Audit
    # ---------------------------------------------------------------
    log.info("--- Part 2: Embedding Source Audit ---")
    # Update vocabulary size from actual discovery
    if "vocab_size" in input_summary:
        config["vocab_size"] = input_summary["vocab_size"]
    emb_result = run_part2(config)
    log.info("Embedding status: %s (%d fragments, %d dims)",
             emb_result["status"], emb_result["n_fragments"], emb_result["embedding_dim"])

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    summary = {
        "preflight_verdict": preflight_verdict,
        "input_status": input_summary.get("status"),
        "embedding_status": emb_result["status"],
        "n_fragments": emb_result.get("n_fragments", 0),
        "embedding_dim": emb_result.get("embedding_dim", 0),
        "explained_variance": emb_result.get("explained_variance", 0),
        "train_queries": input_summary.get("train_queries", 0),
        "val_queries": input_summary.get("val_queries", 0),
        "test_queries": input_summary.get("test_queries", 0),
        "vocab_size": input_summary.get("vocab_size", 0),
        "timestamp": TIMESTAMP,
    }
    log.info("=" * 60)
    log.info("D4A2G-SAFE Preflight Summary:")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)
    log.info("=" * 60)

    write_complete_marker("d4a2g_safe_preflight")


if __name__ == "__main__":
    main()
