#!/usr/bin/env python3
"""
METABRIC preprocessing v3 — Tier 1 upgrades:
  1. One-hot encode ALL categorical clinical variables (17→~45 features)
  2. Standardize continuous clinical variables (Z-score)
  3. Keep mRNA at full 412 genes (small panel — no variance filtering needed)
  4. CNA / Methylation reduced to 256 genes/probes each (balanced with mRNA)
  5. Mutation binarized (presence/absence)

Output: train.npz, val.npz, test.npz with expanded clinical features.
"""

import os, json, glob
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

METABRIC_DIR = "/Users/liushuxing8888student.usm.my/data/METABRIC"
OUT_DIR = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer/data"
RANDOM_STATE = 42

# ──────────────────────────────────────────────
# Gene mapping
# ──────────────────────────────────────────────
def load_gene_mapping_all(batch_files):
    """Load EntrezGeneId → symbol mapping from all mRNA batches."""
    gene_id_to_symbol = {}
    for bf in batch_files:
        with open(bf) as f:
            batch = json.load(f)
        for rec in batch:
            gid = rec.get('entrezGeneId')
            gene_info = rec.get('gene', {})
            symbol = gene_info.get('hugoGeneSymbol', '') if isinstance(gene_info, dict) else ''
            if gid and symbol:
                gene_id_to_symbol[gid] = symbol
    return gene_id_to_symbol

def process_batches(batch_files, gene_mapping, valid_patients, target_dim=None):
    """Build sample × gene matrix from batch JSON files."""
    sample_gene_dict = defaultdict(dict)
    for bf in batch_files:
        with open(bf) as f:
            batch = json.load(f)
        for rec in batch:
            sid = rec.get('sampleId', '')
            if sid not in valid_patients:
                continue
            gid = rec.get('entrezGeneId')
            symbol = gene_mapping.get(gid, '')
            value = rec.get('value', 0)
            if symbol:
                sample_gene_dict[sid][symbol] = value

    all_genes = sorted(set(g for sid in sample_gene_dict for g in sample_gene_dict[sid]))
    sample_ids = sorted(s for s in sample_gene_dict if s in valid_patients)

    mat = np.zeros((len(sample_ids), len(all_genes)), dtype=np.float32)
    for i, sid in enumerate(sample_ids):
        for j, gene in enumerate(all_genes):
            mat[i, j] = sample_gene_dict[sid].get(gene, 0.0)

    if target_dim is not None and mat.shape[1] > target_dim:
        variances = np.nanvar(mat, axis=0)
        top_idx = np.argsort(variances)[::-1][:target_dim]
        mat = mat[:, top_idx]
        all_genes = [all_genes[i] for i in top_idx]

    return mat, sample_ids, all_genes


def process_mutations(mut_file, valid_patients, top_k=173):
    with open(mut_file) as f:
        mut = json.load(f)

    sample_gene_counts = defaultdict(lambda: defaultdict(int))
    gene_total = defaultdict(int)

    for rec in mut:
        sid = rec.get('sampleId', '')
        if sid not in valid_patients:
            continue
        gene_info = rec.get('gene', {})
        gene = gene_info.get('hugoGeneSymbol', '') if isinstance(gene_info, dict) else ''
        if gene:
            sample_gene_counts[sid][gene] += 1
            gene_total[gene] += 1

    top_genes = sorted(gene_total.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_gene_names = [g for g, _ in top_genes]

    sample_ids = sorted(set(sample_gene_counts.keys()) & valid_patients)
    mat = np.zeros((len(sample_ids), len(top_gene_names)), dtype=np.float32)

    for i, sid in enumerate(sample_ids):
        for j, gene in enumerate(top_gene_names):
            mat[i, j] = sample_gene_counts[sid].get(gene, 0)

    return mat, sample_ids, top_gene_names


# ──────────────────────────────────────────────
# Clinical feature engineering (ONE-HOT version)
# ──────────────────────────────────────────────
def build_clinical_features_onehot(patient_df, sample_df, common_samples):
    """
    Build one-hot encoded clinical feature matrix (vectorized).
    Returns:
        clinical_mat: (N, D) float32 array (continuous Z-scored + one-hot categorical)
        feature_names: list of feature names
        labels, durations, events
    """
    # Merge patient + sample by position (same row order)
    merged = pd.concat([patient_df.reset_index(drop=True),
                        sample_df.reset_index(drop=True)], axis=1)
    merged['_key'] = merged['patientId'].astype(str)

    subtype_map = {'LumA': 0, 'LumB': 1, 'Her2': 2, 'claudin-low': 3, 'Basal': 4, 'Normal': 5}
    merged['label'] = merged['CLAUDIN_SUBTYPE'].map(subtype_map)

    # Keep only rows with valid labels AND in common_samples
    merged = merged[merged['label'].notna() & merged['_key'].isin(common_samples)].copy()
    merged = merged.sort_values('_key').reset_index(drop=True)
    print(f"  Clinical rows matched to common samples: {len(merged)}")

    # --- Continuous features (Z-scored) ---
    continuous_cols = [
        'AGE_AT_DIAGNOSIS', 'LYMPH_NODES_EXAMINED_POSITIVE', 'NPI',
        'TUMOR_SIZE', 'TUMOR_STAGE', 'GRADE',
        'MUTATION_COUNT', 'TMB_NONSYNONYMOUS',
    ]

    cont_raw = merged[continuous_cols].replace({np.nan: 0.0}).astype(np.float32)
    scaler = StandardScaler()
    cont_scaled = scaler.fit_transform(cont_raw).astype(np.float32)

    # --- Categorical features (one-hot) ---
    categorical_cols = [
        'CELLULARITY', 'BREAST_SURGERY', 'CHEMOTHERAPY', 'HORMONE_THERAPY',
        'RADIO_THERAPY', 'INFERRED_MENOPAUSAL_STATE',
        'ER_STATUS', 'HER2_STATUS', 'PR_STATUS',
        'LATERALITY', 'THREEGENE', 'HISTOLOGICAL_SUBTYPE',
        'INTCLUST', 'CANCER_TYPE_DETAILED', 'COHORT',
    ]
    cat_df = merged[categorical_cols].fillna('NA').astype(str)
    cat_encoded = pd.get_dummies(cat_df, prefix=categorical_cols, dtype=np.float32)

    # Drop perfectly constant columns (e.g., single-value categories)
    constant_cols = [c for c in cat_encoded.columns if cat_encoded[c].nunique() <= 1]
    if constant_cols:
        cat_encoded = cat_encoded.drop(columns=constant_cols)

    clinical_mat = np.hstack([cont_scaled, cat_encoded.values]).astype(np.float32)
    feature_names = list(continuous_cols) + list(cat_encoded.columns)

    print(f"  Clinical feature dim: {clinical_mat.shape[1]} "
          f"({len(continuous_cols)} continuous + {cat_encoded.shape[1]} categorical)")

    # --- Labels & survival ---
    labels = merged['label'].astype(np.int64).values

    durations = []
    events = []
    for _, row in merged.iterrows():
        rfs = row.get('RFS_MONTHS')
        rfs_status = row.get('RFS_STATUS')
        os_mon = row.get('OS_MONTHS')
        os_status = row.get('OS_STATUS')
        dur = float(rfs) if pd.notna(rfs) else (float(os_mon) if pd.notna(os_mon) else 0.0)
        ev = 1 if (pd.notna(rfs_status) and '1' in str(rfs_status)) or \
                   (pd.notna(os_status) and 'DECEASED' in str(os_status)) else 0
        durations.append(dur)
        events.append(ev)

    durations_arr = np.array(durations, dtype=np.float32)
    events_arr = np.array(events, dtype=np.int64)

    return clinical_mat, feature_names, labels, durations_arr, events_arr, merged['_key'].tolist()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load clinical data
    print("[1/7] Loading clinical data...")
    patient_df = pd.read_csv(os.path.join(METABRIC_DIR, 'clinical_patient_wide.csv'))
    sample_df = pd.read_csv(os.path.join(METABRIC_DIR, 'clinical_sample_wide.csv'))

    subtype_map = {'LumA': 0, 'LumB': 1, 'Her2': 2, 'claudin-low': 3, 'Basal': 4, 'Normal': 5}
    patient_df['_label'] = patient_df['CLAUDIN_SUBTYPE'].map(subtype_map)
    valid_patient_ids = set(patient_df[patient_df['_label'].notna()]['patientId'].dropna().astype(str))
    print(f"  Valid patients with labels: {len(valid_patient_ids)}")

    # 2. Load gene mapping from ALL mRNA batches
    print("\n[2/7] Loading gene mappings...")
    mrna_batches = sorted(glob.glob(os.path.join(METABRIC_DIR, 'mrna_batch*.json')))
    gene_mapping = load_gene_mapping_all(mrna_batches)
    print(f"  Total gene mappings: {len(gene_mapping)}")

    # 3. Process mRNA — keep ALL genes (only ~412 in this panel)
    print("\n[3/7] Processing mRNA (keeping all 412 genes)...")
    mrna_mat, mrna_samples, mrna_genes = process_batches(
        mrna_batches, gene_mapping, valid_patient_ids, target_dim=None)
    # Standardize mRNA
    from sklearn.preprocessing import StandardScaler
    mrna_mat = StandardScaler().fit_transform(mrna_mat).astype(np.float32)
    print(f"  mRNA: {mrna_mat.shape}, samples: {len(mrna_samples)}, genes: {len(mrna_genes)}")

    # 4. Process CNA — reduce to 256 by variance
    print("\n[4/7] Processing CNA...")
    cna_batches = sorted(glob.glob(os.path.join(METABRIC_DIR, 'cna_batch*.json')))
    cna_mat, cna_samples, cna_genes = process_batches(
        cna_batches, gene_mapping, valid_patient_ids, target_dim=256)
    cna_mat = StandardScaler().fit_transform(cna_mat).astype(np.float32)
    print(f"  CNA: {cna_mat.shape}, samples: {len(cna_samples)}, genes: {len(cna_genes)}")

    # 5. Process methylation — reduce to 256 by variance
    print("\n[5/7] Processing Methylation...")
    meth_batches = sorted(glob.glob(os.path.join(METABRIC_DIR, 'methylation_batch*.json')))
    meth_mat, meth_samples, meth_genes = process_batches(
        meth_batches, gene_mapping, valid_patient_ids, target_dim=256)
    meth_mat = StandardScaler().fit_transform(meth_mat).astype(np.float32)
    print(f"  Methylation: {meth_mat.shape}, samples: {len(meth_samples)}, probes: {len(meth_genes)}")

    # 6. Process mutations
    print("\n[6/7] Processing Mutations...")
    mut_mat, mut_samples, mut_genes = process_mutations(
        os.path.join(METABRIC_DIR, 'mutations_fetch.json'), valid_patient_ids, top_k=173)
    # Binarize: count → presence/absence
    mut_mat = (mut_mat > 0).astype(np.float32)
    print(f"  Mutations: {mut_mat.shape}, samples: {len(mut_samples)}, genes: {len(mut_genes)}")

    # 7. Align all modalities
    print("\n[7/7] Aligning modalities and building clinical features...")
    common_samples = sorted(set(mrna_samples) & set(cna_samples) & set(meth_samples) & set(mut_samples))
    print(f"  Common samples: {len(common_samples)}")

    def align_matrix(mat, all_samples, common_samples):
        sample_to_idx = {s: i for i, s in enumerate(all_samples)}
        aligned = np.zeros((len(common_samples), mat.shape[1]), dtype=np.float32)
        for i, sid in enumerate(common_samples):
            if sid in sample_to_idx:
                aligned[i] = mat[sample_to_idx[sid]]
        return aligned

    mrna_aligned = align_matrix(mrna_mat, mrna_samples, common_samples)
    cna_aligned = align_matrix(cna_mat, cna_samples, common_samples)
    meth_aligned = align_matrix(meth_mat, meth_samples, common_samples)
    mut_aligned = align_matrix(mut_mat, mut_samples, common_samples)

    # Build one-hot clinical features
    clinical_mat, feat_names, labels, durations, events, matched_samples = \
        build_clinical_features_onehot(patient_df, sample_df, set(common_samples))

    # Re-align molecular data to matched clinical samples
    final_samples = set(matched_samples)
    mrna_final = align_matrix(mrna_aligned, common_samples, matched_samples)
    cna_final = align_matrix(cna_aligned, common_samples, matched_samples)
    meth_final = align_matrix(meth_aligned, common_samples, matched_samples)
    mut_final = align_matrix(mut_aligned, common_samples, matched_samples)

    print(f"\n  Final samples: {len(labels)}")
    print(f"  Label distribution: {np.bincount(labels, minlength=6)}")
    print(f"  Clinical: {clinical_mat.shape}")
    print(f"  mRNA: {mrna_final.shape}")
    print(f"  CNA: {cna_final.shape}")
    print(f"  Methylation: {meth_final.shape}")
    print(f"  Mutations: {mut_final.shape}")

    # Save feature names for reference
    with open(os.path.join(OUT_DIR, 'feature_names.json'), 'w') as f:
        json.dump({
            'clinical_features': feat_names,
            'mrna_genes': mrna_genes,
            'cna_genes': cna_genes,
            'meth_probes': meth_genes,
            'mut_genes': mut_genes,
        }, f, indent=2)

    # 8. Split
    print("\n[8/8] Splitting and saving...")
    train_idx, temp_idx = train_test_split(
        range(len(labels)), test_size=0.30, stratify=labels, random_state=RANDOM_STATE)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=RANDOM_STATE)

    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    for name, idx in splits.items():
        np.savez(os.path.join(OUT_DIR, f'{name}.npz'),
                 mod_0=clinical_mat[idx],
                 mod_1=mrna_final[idx],
                 mod_2=cna_final[idx],
                 mod_3=meth_final[idx],
                 mod_4=mut_final[idx],
                 labels=labels[idx],
                 durations=durations[idx],
                 events=events[idx],
                 patient_ids=np.array([matched_samples[i] for i in idx]))
        print(f"  {name}.npz: {len(idx)} samples, labels: {np.bincount(labels[idx], minlength=6)}")

    print(f"\nDone! Saved to {OUT_DIR}")
    print(f"Clinical features: {len(feat_names)} dims (was 17 dims)")


if __name__ == '__main__':
    main()
