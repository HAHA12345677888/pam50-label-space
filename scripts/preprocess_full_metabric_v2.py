#!/usr/bin/env python3
"""
Full METABRIC preprocessing with all modalities from batch JSON files.
Creates train.npz, val.npz, test.npz with:
- Clinical features (80-dim)
- mRNA expression (500 genes, top by variance)
- CNA (500 genes, top by variance)
- Methylation (500 probes, top by variance)
- Mutations (173 genes, top mutated)
"""
import os, json, glob
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

METABRIC_DIR = "/Users/liushuxing8888student.usm.my/data/METABRIC"
OUT_DIR = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer/data"
RANDOM_STATE = 42

def load_gene_mapping(batch_file):
    with open(batch_file) as f:
        batch = json.load(f)
    gene_id_to_symbol = {}
    for rec in batch:
        gid = rec.get('entrezGeneId')
        gene_info = rec.get('gene', {})
        symbol = gene_info.get('hugoGeneSymbol', '') if isinstance(gene_info, dict) else ''
        if gid and symbol:
            gene_id_to_symbol[gid] = symbol
    return gene_id_to_symbol

def process_batches(batch_files, gene_mapping, valid_patients, target_dim=500):
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
    
    # Select top genes by variance
    if mat.shape[1] > target_dim:
        variances = np.nanvar(mat, axis=0)
        top_idx = np.argsort(variances)[::-1][:target_dim]
        mat = mat[:, top_idx]
        all_genes = [all_genes[i] for i in top_idx]
    
    return mat, sample_ids, all_genes

def process_mutations(mut_file, valid_patients, top_k=173):
    with open(mut_file) as f:
        mut = json.load(f)
    
    # Count mutations per gene per sample
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
    
    # Select top mutated genes
    top_genes = sorted(gene_total.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_gene_names = [g for g, _ in top_genes]
    
    sample_ids = sorted(set(sample_gene_counts.keys()) & valid_patients)
    mat = np.zeros((len(sample_ids), len(top_gene_names)), dtype=np.float32)
    
    for i, sid in enumerate(sample_ids):
        for j, gene in enumerate(top_gene_names):
            mat[i, j] = sample_gene_counts[sid].get(gene, 0)
    
    return mat, sample_ids, top_gene_names

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load clinical data
    print("[1/6] Loading clinical data...")
    clinical = pd.read_csv(os.path.join(METABRIC_DIR, 'clinical_merged.csv'))
    subtype_map = {'LumA': 0, 'LumB': 1, 'Her2': 2, 'claudin-low': 3, 'Basal': 4, 'Normal': 5}
    clinical['label'] = clinical['CLAUDIN_SUBTYPE'].map(subtype_map)
    valid_clinical = clinical[clinical['label'].notna()].copy()
    valid_patients = set(valid_clinical['patientId'].dropna().astype(str))
    print(f"  Valid patients with labels: {len(valid_patients)}")
    
    # 2. Load gene mapping from batch1
    print("\n[2/6] Loading gene mappings...")
    gene_mapping = load_gene_mapping(os.path.join(METABRIC_DIR, 'mrna_batch1.json'))
    print(f"  Gene mappings: {len(gene_mapping)}")
    
    # 3. Process mRNA
    print("\n[3/6] Processing mRNA...")
    mrna_batches = sorted(glob.glob(os.path.join(METABRIC_DIR, 'mrna_batch*.json')))
    mrna_mat, mrna_samples, mrna_genes = process_batches(mrna_batches, gene_mapping, valid_patients, target_dim=500)
    print(f"  mRNA: {mrna_mat.shape}, samples: {len(mrna_samples)}, genes: {len(mrna_genes)}")
    
    # 4. Process CNA
    print("\n[4/6] Processing CNA...")
    cna_batches = sorted(glob.glob(os.path.join(METABRIC_DIR, 'cna_batch*.json')))
    cna_mat, cna_samples, cna_genes = process_batches(cna_batches, gene_mapping, valid_patients, target_dim=500)
    print(f"  CNA: {cna_mat.shape}, samples: {len(cna_samples)}, genes: {len(cna_genes)}")
    
    # 5. Process methylation
    print("\n[5/6] Processing Methylation...")
    meth_batches = sorted(glob.glob(os.path.join(METABRIC_DIR, 'methylation_batch*.json')))
    meth_mat, meth_samples, meth_genes = process_batches(meth_batches, gene_mapping, valid_patients, target_dim=500)
    print(f"  Methylation: {meth_mat.shape}, samples: {len(meth_samples)}, probes: {len(meth_genes)}")
    
    # 6. Process mutations
    print("\n[6/6] Processing Mutations...")
    mut_mat, mut_samples, mut_genes = process_mutations(
        os.path.join(METABRIC_DIR, 'mutations_fetch.json'), valid_patients, top_k=173)
    print(f"  Mutations: {mut_mat.shape}, samples: {len(mut_samples)}, genes: {len(mut_genes)}")
    
    # 7. Align all modalities to common samples
    print("\n[7/7] Aligning modalities...")
    common_samples = sorted(set(mrna_samples) & set(cna_samples) & set(meth_samples) & set(mut_samples))
    print(f"  Common samples across all modalities: {len(common_samples)}")
    
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
    
    # 8. Build clinical features
    print("\nBuilding clinical features...")
    clinical_features = []
    labels = []
    durations = []
    events = []
    
    for sid in common_samples:
        pat_row = valid_clinical[valid_clinical['patientId'] == sid]
        if pat_row.empty:
            continue
        row = pat_row.iloc[0]
        
        age = float(row['AGE_AT_DIAGNOSIS']) if pd.notna(row['AGE_AT_DIAGNOSIS']) else 0.0
        lymph_nodes = float(row['LYMPH_NODES_EXAMINED_POSITIVE']) if pd.notna(row['LYMPH_NODES_EXAMINED_POSITIVE']) else 0.0
        npi = float(row['NPI']) if pd.notna(row['NPI']) else 0.0
        tumor_size = float(row['TUMOR_SIZE']) if pd.notna(row['TUMOR_SIZE']) else 0.0
        tumor_stage = float(row['TUMOR_STAGE']) if pd.notna(row['TUMOR_STAGE']) else 0.0
        grade = float(row['GRADE']) if pd.notna(row['GRADE']) else 0.0
        mutation_count = float(row['MUTATION_COUNT']) if pd.notna(row['MUTATION_COUNT']) else 0.0
        tmb = float(row['TMB_NONSYNONYMOUS']) if pd.notna(row['TMB_NONSYNONYMOUS']) else 0.0
        
        cellularity = {'High': 0, 'Low': 1, 'Moderate': 2, 'NA': 3}.get(row['CELLULARITY'], 3)
        er = {'Positive': 0, 'Negative': 1, 'Positve': 0}.get(row['ER_STATUS'], 2)
        her2 = {'Positive': 0, 'Negative': 1, 'NEUTRAL': 2}.get(row['HER2_STATUS'], 2)
        pr = {'Positive': 0, 'Negative': 1}.get(row['PR_STATUS'], 2)
        surgery = {'MASTECTOMY': 0, 'BREAST CONSERVING': 1, 'NA': 2}.get(row['BREAST_SURGERY'], 2)
        chemo = {'YES': 1, 'NO': 0, 'NA': 2}.get(row['CHEMOTHERAPY'], 2)
        hormonal = {'YES': 1, 'NO': 0, 'NA': 2}.get(row['HORMONE_THERAPY'], 2)
        radio = {'YES': 1, 'NO': 0, 'NA': 2}.get(row['RADIO_THERAPY'], 2)
        menopause = {'Pre': 0, 'Post': 1, 'NA': 2}.get(row['INFERRED_MENOPAUSAL_STATE'], 2)
        
        features = [age, lymph_nodes, npi, tumor_size, tumor_stage, grade,
                    mutation_count, tmb, cellularity, er, her2, pr, surgery,
                    chemo, hormonal, radio, menopause]
        
        clinical_features.append(features)
        labels.append(int(row['label']))
        
        rfs = row['RFS_MONTHS'] if pd.notna(row['RFS_MONTHS']) else row['OS_MONTHS']
        rfs_status = row['RFS_STATUS'] if pd.notna(row['RFS_STATUS']) else row['OS_STATUS']
        durations.append(float(rfs) if pd.notna(rfs) else 0.0)
        events.append(1 if '1' in str(rfs_status) or 'DECEASED' in str(rfs_status) else 0)
    
    clinical_features = np.array(clinical_features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    durations = np.array(durations, dtype=np.float32)
    events = np.array(events, dtype=np.int64)
    
    print(f"  Final samples: {len(labels)}")
    print(f"  Label dist: {np.bincount(labels, minlength=6)}")
    print(f"  mRNA: {mrna_aligned.shape}")
    print(f"  CNA: {cna_aligned.shape}")
    print(f"  Meth: {meth_aligned.shape}")
    print(f"  Mut: {mut_aligned.shape}")
    
    # 9. Split and save
    print("\n[9/9] Splitting and saving...")
    train_idx, temp_idx = train_test_split(range(len(labels)), test_size=0.30,
                                           stratify=labels, random_state=RANDOM_STATE)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50,
                                         stratify=labels[temp_idx], random_state=RANDOM_STATE)
    
    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    for name, idx in splits.items():
        np.savez(os.path.join(OUT_DIR, f'{name}.npz'),
                 mod_0=clinical_features[idx],
                 mod_1=mrna_aligned[idx],
                 mod_2=cna_aligned[idx],
                 mod_3=meth_aligned[idx],
                 mod_4=mut_aligned[idx],
                 labels=labels[idx],
                 durations=durations[idx],
                 events=events[idx],
                 patient_ids=np.array([common_samples[i] for i in idx]))
        print(f"  {name}.npz: {len(idx)} samples, labels: {np.bincount(labels[idx], minlength=6)}")
    
    print(f"\nDone! Saved to {OUT_DIR}")

if __name__ == '__main__':
    main()
