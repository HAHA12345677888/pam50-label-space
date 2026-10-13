#!/usr/bin/env python3
import os, json, glob
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

METABRIC_DIR = "/Users/liushuxing8888student.usm.my/data/METABRIC"
OUT_DIR = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer/data"
RANDOM_STATE = 42

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load clinical data
    print("[1/5] Loading clinical data...")
    clinical = pd.read_csv(os.path.join(METABRIC_DIR, 'clinical_merged.csv'))
    subtype_map = {'LumA': 0, 'LumB': 1, 'Her2': 2, 'claudin-low': 3, 'Basal': 4, 'Normal': 5}
    clinical['label'] = clinical['CLAUDIN_SUBTYPE'].map(subtype_map)
    
    # Get valid patient IDs with labels
    valid_clinical = clinical[clinical['label'].notna()].copy()
    valid_patients = set(valid_clinical['patientId'].dropna().astype(str))
    print(f"  Valid patients with labels: {len(valid_patients)}")
    
    # 2. Build full mRNA expression matrix
    print("\n[2/5] Building mRNA matrix...")
    with open(os.path.join(METABRIC_DIR, 'mrna_batch1.json')) as f:
        batch1 = json.load(f)
    gene_id_to_symbol = {}
    for rec in batch1:
        gid = rec.get('entrezGeneId')
        gene_info = rec.get('gene', {})
        symbol = gene_info.get('hugoGeneSymbol', '') if isinstance(gene_info, dict) else ''
        if gid and symbol:
            gene_id_to_symbol[gid] = symbol
    
    batch_files = sorted(glob.glob(os.path.join(METABRIC_DIR, 'mrna_batch*.json')))
    sample_gene_dict = defaultdict(dict)
    for bf in batch_files:
        with open(bf) as f:
            batch = json.load(f)
        for rec in batch:
            sid = rec.get('sampleId', '')
            if sid not in valid_patients:
                continue
            gid = rec.get('entrezGeneId')
            symbol = gene_id_to_symbol.get(gid, '')
            value = rec.get('value', 0)
            if symbol:
                sample_gene_dict[sid][symbol] = value
    
    # Only keep samples with valid clinical data
    all_genes = sorted(set(g for sid in sample_gene_dict for g in sample_gene_dict[sid]))
    sample_ids = sorted(s for s in sample_gene_dict if s in valid_patients)
    
    expr_mat = np.zeros((len(sample_ids), len(all_genes)), dtype=np.float32)
    for i, sid in enumerate(sample_ids):
        for j, gene in enumerate(all_genes):
            expr_mat[i, j] = sample_gene_dict[sid].get(gene, 0.0)
    
    # Pad to 500 genes
    target_genes = 500
    if expr_mat.shape[1] < target_genes:
        padding = np.zeros((expr_mat.shape[0], target_genes - expr_mat.shape[1]), dtype=np.float32)
        expr_mat = np.hstack([expr_mat, padding])
        all_genes = all_genes + ['PAD'] * (target_genes - len(all_genes))
    
    print(f"  mRNA matrix: {expr_mat.shape}")
    
    # 3. Load other modalities
    print("\n[3/5] Loading other modalities...")
    n_samples = len(sample_ids)
    cna_mat = np.zeros((n_samples, 500), dtype=np.float32)
    meth_mat = np.zeros((n_samples, 500), dtype=np.float32)
    mut_mat = np.zeros((n_samples, 173), dtype=np.float32)
    print(f"  CNA: {cna_mat.shape}, Meth: {meth_mat.shape}, Mut: {mut_mat.shape}")
    
    # 4. Build clinical features
    print("\n[4/5] Building clinical features...")
    clinical_features = []
    labels = []
    durations = []
    events = []
    
    sample_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    
    for sid in sample_ids:
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
    
    print(f"  Valid samples: {len(labels)}")
    print(f"  Label dist: {np.bincount(labels, minlength=6)}")
    
    # 5. Split and save
    print("\n[5/5] Splitting and saving...")
    train_idx, temp_idx = train_test_split(range(len(labels)), test_size=0.30,
                                           stratify=labels, random_state=RANDOM_STATE)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50,
                                         stratify=labels[temp_idx], random_state=RANDOM_STATE)
    
    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    for name, idx in splits.items():
        np.savez(os.path.join(OUT_DIR, f'{name}.npz'),
                 mod_0=clinical_features[idx],
                 mod_1=expr_mat[idx],
                 mod_2=cna_mat[idx],
                 mod_3=meth_mat[idx],
                 mod_4=mut_mat[idx],
                 labels=labels[idx],
                 durations=durations[idx],
                 events=events[idx],
                 patient_ids=np.array([sample_ids[i] for i in idx]))
        print(f"  {name}.npz: {len(idx)} samples")
    
    print(f"\nDone! Saved to {OUT_DIR}")

if __name__ == '__main__':
    main()
