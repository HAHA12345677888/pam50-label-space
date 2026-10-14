"""
Step 15 — Build a cross-cohort harmonised feature space (METABRIC <-> TCGA-BRCA).

The published external validation was mRNA-only. The reason recorded in the
manuscript is that TCGA "lacks METABRIC-compatible clinical and methylation
features". That is true at the level of the full 69-dimension clinical block,
whose fields (NPI, GRADE, INTCLUST_*, THREEGENE_*, COHORT_*) are METABRIC
constructions with no TCGA counterpart. It is not true for a reduced block.

This step defines the largest clinical sub-block that exists in both cohorts:

    continuous   AGE, TUMOUR_STAGE          -> within-cohort rank -> normal scores
    binary       ER, PR, HER2 (neg/pos)     -> 6 indicators
                 MENOPAUSAL_STATE (pre/post) -> 2 indicators

and the two molecular layers both cohorts share:

    mRNA         412 genes                  -> within-cohort rank -> normal scores
    methylation  232 genes                  -> within-cohort rank -> normal scores

Normalisation is per-cohort, unsupervised, and label-free, so it uses no
information the model would not have at deployment. A gene unmeasured on the
target platform normalises to 0 (the normal-score mean), i.e. contributes
nothing, and the count of such genes is reported rather than hidden.

Outputs
-------
data/metabric_harmonized.npz   mods (clinical_h, mrna_h, meth_h), labels, splits
data/tcga_brca/tcga_harmonized.npz  same layout for the target cohort
results/route_a/harmonization.json  coverage report
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TCGA = os.path.join(DATA, "tcga_brca")
RESULTS = os.path.join(ROOT, "results", "route_a")

# METABRIC one-hot column names for the harmonised clinical block.
MET_CLIN = {
    "age": "AGE_AT_DIAGNOSIS",
    "stage": "TUMOR_STAGE",
    "er_neg": "ER_STATUS_Negative", "er_pos": "ER_STATUS_Positive",
    "her2_neg": "HER2_STATUS_Negative", "her2_pos": "HER2_STATUS_Positive",
    "pr_neg": "PR_STATUS_Negative", "pr_pos": "PR_STATUS_Positive",
    "meno_pre": "INFERRED_MENOPAUSAL_STATE_Pre",
    "meno_post": "INFERRED_MENOPAUSAL_STATE_Post",
}
CLIN_NAMES = ["AGE", "STAGE", "ER_NEG", "ER_POS", "HER2_NEG", "HER2_POS",
              "PR_NEG", "PR_POS", "MENO_PRE", "MENO_POST"]

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def rank_normalise(X):
    """Column-wise rank -> normal scores. All-NaN column -> zeros (unmeasured)."""
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = np.isfinite(col)
        if ok.sum() == 0:
            continue                      # unmeasured gene: neutral value
        if ok.sum() == 1 or np.all(col[ok] == col[ok][0]):
            out[ok, j] = 0.0              # no variance: neutral
            continue
        r = rankdata(col[ok], method="average")
        out[ok, j] = norm.ppf((r - 0.5) / ok.sum())
    return out.astype(np.float32)


def tcga_stage_to_ordinal(s):
    """'Stage IIB' -> 2 ; 'Stage X'/NaN -> NaN."""
    if not isinstance(s, str) or "X" in s:
        return np.nan
    tok = s.replace("Stage", "").strip()
    for r in ("IV", "III", "II", "I"):
        if tok.startswith(r):
            return float(_ROMAN[r])
    return np.nan


def build_tcga_clinical(patient_ids, pc):
    """Harmonised 10-dim clinical block for TCGA, matching METABRIC's encoding."""
    pc = pc.set_index("patientId")
    n = len(patient_ids)
    C = np.zeros((n, len(CLIN_NAMES)), dtype=np.float64)
    raw_age = np.full(n, np.nan)
    raw_stage = np.full(n, np.nan)

    def pos_indicator(v, pos_label="Positive"):
        return (v == pos_label), (v == "Negative")

    for i, pid in enumerate(patient_ids):
        if pid not in pc.index:
            continue
        r = pc.loc[pid]
        if pd.notna(r.get("AGE")):
            raw_age[i] = float(r["AGE"])
        raw_stage[i] = tcga_stage_to_ordinal(r.get("AJCC_PATHOLOGIC_TUMOR_STAGE"))
        er, ern = pos_indicator(r.get("ER_STATUS_BY_IHC"))
        pr, prn = pos_indicator(r.get("PR_STATUS_BY_IHC"))
        her2, her2n = pos_indicator(r.get("IHC_HER2"))
        C[i, 2] = 1.0 if ern else 0.0
        C[i, 3] = 1.0 if er else 0.0
        C[i, 4] = 1.0 if her2n else 0.0
        C[i, 5] = 1.0 if her2 else 0.0
        C[i, 6] = 1.0 if prn else 0.0
        C[i, 7] = 1.0 if pr else 0.0
        m = r.get("MENOPAUSE_STATUS")
        if isinstance(m, str):
            if m.startswith("Pre") or m.startswith("Peri"):
                C[i, 8] = 1.0
            elif m.startswith("Post"):
                C[i, 9] = 1.0
    C[:, 0] = raw_age
    C[:, 1] = raw_stage
    return C, raw_age, raw_stage


def main():
    os.makedirs(RESULTS, exist_ok=True)
    fn = json.load(open(os.path.join(DATA, "feature_names.json")))
    met_names = fn["clinical_features"]
    met_meth_genes = fn["meth_probes"]
    met_mrna_genes = fn["mrna_genes"]

    # ---------------- METABRIC ----------------
    splits, labels, pids = {}, {}, {}
    for s in ("train", "val", "test"):
        d = np.load(os.path.join(DATA, f"{s}.npz"), allow_pickle=True)
        splits[s] = d
        labels[s] = d["labels"]
        pids[s] = d["patient_ids"].astype(str)

    idx = {k: met_names.index(v) for k, v in MET_CLIN.items() if v in met_names}
    missing_met = [v for k, v in MET_CLIN.items() if v not in met_names]
    if missing_met:
        raise SystemExit(f"METABRIC clinical fields absent: {missing_met}")

    def met_clin_block(d):
        C = d["mod_0"]
        out = np.zeros((C.shape[0], len(CLIN_NAMES)), dtype=np.float64)
        out[:, 0] = C[:, idx["age"]]
        out[:, 1] = C[:, idx["stage"]]
        for j, k in enumerate(["er_neg", "er_pos", "her2_neg", "her2_pos",
                               "pr_neg", "pr_pos", "meno_pre", "meno_post"]):
            out[:, 2 + j] = C[:, idx[k]]
        return out

    met = {}
    for s in ("train", "val", "test"):
        d = splits[s]
        met[s] = dict(
            clinical=rank_normalise(met_clin_block(d)),
            mrna=rank_normalise(d["mod_1"]),
            meth=rank_normalise(d["mod_3"]),
        )

    # ---------------- TCGA ----------------
    v = np.load(os.path.join(TCGA, "tcga_validation.npz"), allow_pickle=True)
    m = np.load(os.path.join(TCGA, "tcga_methylation.npz"), allow_pickle=True)
    pc = pd.read_csv(os.path.join(TCGA, "patient_clinical.csv"), low_memory=False)
    pc["patientId"] = pc["patientId"].astype(str)

    # patients with mRNA + label, keyed on 12-char participant barcode
    v_pid = np.array([str(x) for x in v["patient_ids"]])
    m_pid = np.array([str(s)[:12] for s in m["sample_ids"]])

    # align methylation columns to METABRIC's gene order
    m_genes = [str(g) for g in m["gene_symbols"]]
    col_of = {g: i for i, g in enumerate(m_genes)}
    keep = [p for p in met_meth_genes if p in col_of]
    absent = [p for p in met_meth_genes if p not in col_of]
    M = m["methylation"][:, [col_of[p] for p in keep]]
    Meth_full = np.full((M.shape[0], len(met_meth_genes)), np.nan, dtype=np.float64)
    Meth_full[:, [met_meth_genes.index(p) for p in keep]] = M
    met_meth_idx = np.array([m_pid.tolist().index(p) for p in m_pid])

    # align TCGA mRNA columns to METABRIC's gene order
    em = np.load(os.path.join(TCGA, "mrna_expression.npz"), allow_pickle=True)
    em_genes = [str(g) for g in em["gene_symbols"]]
    em_col = {g: i for i, g in enumerate(em_genes)}
    mrna_keep = [g for g in met_mrna_genes if g in em_col]
    mrna_absent = [g for g in met_mrna_genes if g not in em_col]

    # index TCGA rows to the intersection (mRNA+label AND methylation)
    meth_by_pid = {p: i for i, p in enumerate(m_pid)}
    rows, t_pid = [], []
    for i, p in enumerate(v_pid):
        if p in meth_by_pid:
            rows.append(i)
            t_pid.append(p)
    rows = np.array(rows)
    t_pid = np.array(t_pid)
    t_mid = np.array([meth_by_pid[p] for p in t_pid])

    # mRNA: pull by sample_id (15-char) -> map participant -> expression row
    v_sid = np.array([str(x) for x in v["sample_ids"]])
    em_sid = {str(s): i for i, s in enumerate(em["sample_ids"])}
    mrna_rows = np.array([em_sid.get(s, -1) for s in v_sid])
    if (mrna_rows < 0).any():
        raise SystemExit("TCGA mRNA missing for some validation samples")

    MRNA = np.full((len(t_pid), len(met_mrna_genes)), np.nan, dtype=np.float64)
    src = em["expression"][mrna_rows[rows]][:, [em_col[g] for g in mrna_keep]]
    MRNA[:, [met_mrna_genes.index(g) for g in mrna_keep]] = src

    clin, raw_age, raw_stage = build_tcga_clinical(t_pid, pc)

    tcga = dict(
        clinical=rank_normalise(clin),
        mrna=rank_normalise(MRNA),
        meth=rank_normalise(Meth_full[t_mid]),
        labels=v["labels"][rows],
        patient_ids=t_pid,
    )

    # ---------------- save ----------------
    np.savez_compressed(
        os.path.join(DATA, "metabric_harmonized.npz"),
        **{f"{s}_{k}": met[s][k] for s in ("train", "val", "test")
           for k in ("clinical", "mrna", "meth")},
        **{f"{s}_labels": labels[s] for s in ("train", "val", "test")},
    )
    np.savez_compressed(
        os.path.join(TCGA, "tcga_harmonized.npz"),
        clinical=tcga["clinical"], mrna=tcga["mrna"], meth=tcga["meth"],
        labels=tcga["labels"], patient_ids=tcga["patient_ids"],
    )

    u, c = np.unique(tcga["labels"], return_counts=True)
    # Report *measurement* coverage, not symbol coverage: a gene can be present
    # in the target platform's gene list yet carry no data, and rank
    # normalisation then emits a neutral (zero) column either way.
    meth_unmeasured = int(np.isnan(Meth_full).all(0).sum())
    raw_mrna_const = int((np.nanstd(em["expression"], axis=0) == 0).sum())
    report = {
        "harmonized_clinical_dims": CLIN_NAMES,
        "metabric": {s: int(len(labels[s])) for s in ("train", "val", "test")},
        "tcga_n": int(len(tcga["labels"])),
        "tcga_class_counts": {int(k): int(v_) for k, v_ in zip(u, c)},
        "tcga_classes_present": [int(x) for x in u],
        "tcga_classes_absent": [k for k in range(6) if k not in set(u.tolist())],
        "methylation_genes_in_panel": len(met_meth_genes),
        "methylation_genes_measured_on_target": len(met_meth_genes) - meth_unmeasured,
        "methylation_genes_unmeasured_on_target": meth_unmeasured,
        "mrna_genes_in_panel": len(met_mrna_genes),
        "mrna_genes_symbol_matched": len(mrna_keep),
        "mrna_genes_constant_on_target": raw_mrna_const,
        "tcga_age_missing": int(np.isnan(raw_age).sum()),
        "tcga_stage_missing": int(np.isnan(raw_stage).sum()),
        "neutral_value_policy": "unmeasured or constant feature -> normal score 0",
        "note": "rank->normal scores computed within each cohort, label-free",
    }
    json.dump(report, open(os.path.join(RESULTS, "harmonization.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
