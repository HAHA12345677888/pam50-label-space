# Breast Cancer Multimodal PAM50 Subtyping — Label-Space Mismatch and Cross-Cohort Evaluation

Analysis code, results, and manuscript for a study of how **annotation-space differences** affect
cross-cohort evaluation of breast cancer molecular-subtype classifiers, and of whether a multimodal
advantage survives external validation.

The headline results are mostly negative, and they are reported as such.

## What this repository contains

| Path | Contents |
|---|---|
| `scripts/route_a_*.py` | The analysis pipeline, one script per experiment step (see below) |
| `results/route_a/*.json` | The JSON outputs behind every number reported in the manuscript |
| `paper/manuscript.tex` | Manuscript source (single-column) |
| `paper/manuscript_ieee.tex` | Manuscript source (IEEEtran, two-column) |
| `paper/references.bib` | Bibliography |
| `paper/figures/` | Figure PDFs, embedded by the manuscript |
| `paper/build.sh` | Build script (see the build-order note below) |
| `paper/submission_checklist.md` | Submission readiness checklist |
| `data/` | Preprocessed METABRIC arrays and external-cohort data |
| `code/`, `models/` | The earlier five-modality Transformer, retained for provenance |

## Key findings

1. **The annotation space, not the model, dominated reported cross-cohort performance.**
   TCGA-BRCA accuracy was 73.76% uncorrected and 80.65% after restricting predictions to the five
   PAM50 classes with EM prior correction. GSE21653 went from 63.16% to 74.06%. These are changes
   in *evaluation rules*, not improvements in the learned representation.

2. **Unlabeled predictions alone did not identify the absent class.** Unconstrained EM assigned the
   absent claudin-low class a prior of 0.0751 on TCGA-BRCA. Two tested detectors also failed
   (geometric F1 0.047; an LRT procedure selected a non-empty absent set in 90% of clean null runs).

3. **The multimodal advantage did not survive external validation.** In a harmonized three-layer
   space that both METABRIC and TCGA-BRCA carry (`multimodal_external.json`), adding methylation
   was flat under cross-validation and lowered TCGA accuracy by 1.9 points; the harmonized clinical
   block raised held-out METABRIC test accuracy by 2.4 points but moved TCGA accuracy by 0.3 points
   or less; and no multimodal configuration differed significantly from mRNA alone
   (paired McNemar, *p* = 0.13 to 1.00).

4. **The best internal model is not transportable.** The 713-dimension flagship reaches 75.92%
   cross-validated accuracy, against 73.17% for the harmonized tri-modal model. The 2.75-point gap
   is contributed by METABRIC-specific clinical fields (NPI, GRADE, INTCLUST, THREEGENE, cohort
   indicator) that have no counterpart in the target cohort.

5. **A fusion Transformer did not beat the linear reference.** 67.3% vs 75.3% cross-validated
   accuracy. Adding a second training cohort moved the Transformer by +1.11 points, within 0.11
   points of the learning-curve prediction — the deficit is a bias limitation, not a data one.

6. **Treatment-response prediction was null.** In GSE20194 (n=278), the genomic panel did not
   significantly improve pCR prediction over clinical variables (ΔAUC = +0.0045; 95% CI,
   −0.060 to +0.061).

## Pipeline

The scripts are numbered by execution order. All write to `results/route_a/`.

| Step | Script | What it does |
|---|---|---|
| — | `route_a_harness.py` | Shared loaders: `load_split`, `pool_splits`, `stack`, `make_logreg` |
| 1 | `route_a_step1_refs.py` | Rebuild the reference numbers |
| 2 | `route_a_step2_diagnosis.py` | Transformer gap: optimization or data? |
| 3 | `route_a_step3_regularize.py` | Regularization and modality dropout |
| 4 | `route_a_step4_pretrain.py` | Self-supervised encoder pretraining |
| 5 | `route_a_step5_tcga.py` | External validation on TCGA-BRCA (n=987) |
| 5b/5c | `route_a_step5b_tcga_survival.py`, `route_a_step5c_tcga_tdc.py` | Survival analysis |
| 6 | `route_a_step6_treatment.py` | Treatment-response prediction |
| 7 | `route_a_step7_gse21653.py` | Second training cohort |
| 7b | `route_a_step7b_multicohort.py` | Multi-cohort training test |
| 8 | `route_a_step8_shift.py` | Cross-cohort shift decomposition |
| 9 | `route_a_step9_pcr.py` | pCR prediction in GSE20194 |
| 10 | `route_a_step10_sim.py` | Label-space mismatch simulation |
| 11 | `route_a_step11_decompose.py` | Label-space vs prior shift |
| 12 | `route_a_step12_detector.py` | Geometric absence detector |
| 13 | `route_a_step13_detector_cal.py` | Calibrated detector |
| 14 | `route_a_step14_lrt_detector.py` | Likelihood-ratio detector |
| 15 | `route_a_step15_harmonize.py` | Build the cross-cohort harmonized space |
| 16 | `route_a_step16_multimodal_external.py` | **Modality ablation in the harmonized space** |
| 17 | `route_a_step17_detector_v2.py` | **Calibrated absence detector, re-evaluated** |
| — | `fetch_tcga_methylation.py` | Fetch TCGA-BRCA HM450 gene-level methylation from cBioPortal |

Steps 15–17 require the TCGA methylation file. Fetch it first:

```bash
python fetch_tcga_methylation.py     # writes data/tcga_brca/tcga_methylation.npz
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11. The pipeline was developed against torch 2.12, scikit-learn 1.9,
lifelines 0.30, numpy 2.4, pandas 2.3, scipy 1.17.

## Reproducing the manuscript numbers

```bash
python scripts/route_a_step1_refs.py          # etc., in numeric order
```

Each script prints its metrics and writes a JSON into `results/route_a/`, which is the source of
truth for the manuscript's tables.

## Building the manuscript

`latexmk` runs BibTeX before regenerating the `.aux` when `references.bib` is newer, which yields
0 bibitems. Use the explicit order:

```bash
cd paper
pdflatex -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

## Data sources

- **METABRIC** — Curtis et al., *Nature* 486:346–352, 2012; Pereira et al., *Nature Communications*
  7:11479, 2016. Preprocessed arrays in `data/{train,val,test}.npz`.
- **TCGA-BRCA** — mRNA and PAM50 labels in `data/tcga_brca/`; gene-level HM450 methylation fetched
  from cBioPortal (`brca_tcga_methylation_hm450`, 788 samples).
- **GSE21653**, **GSE20194** — GEO.

## Reproducibility notes

- Continuous features are rank-transformed to normal scores **within each cohort**, a label-free
  operation. No target labels are used in any normalization or detector step.
- Unmeasured or invariant features are assigned the neutral score 0 rather than imputed.
- Cross-validated scores use 5-fold stratified CV on pooled METABRIC train+val (n=1,163).
  The joblib process backend is unavailable in some sandboxes; scripts use `n_jobs=1`.
- `figures/figA`–`figF` correspond to the supplementary experiments (learning curves, capacity
  sweep, regularization, pretraining, TCGA validation, treatment AUC).


## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py \
    --train_npz data/train.npz \
    --val_npz data/val.npz \
    --output_dir checkpoints \
    --epochs 100 \
    --pathway_mask_path data/soft_module_mask.npz
```

## Evaluation

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pt \
    --data_path data/test.npz \
    --output_dir results \
    --num_classes 6
```

## Visualization

```bash
python visualize.py \
    --model_path checkpoints/best_model.pt \
    --data_npz data/test.npz \
    --out_dir figures \
    --num_classes 6
```

## Paper

See `paper_cn_v2.pdf` for the full paper draft.

## Data Availability

Preprocessed data files are available upon request.
Original data: [cBioPortal METABRIC](https://www.cbioportal.org/study/brca_metabric).

## Citation

```bibtex
@article{liu2025breast,
  title={A Unified Multimodal Transformer with Pathway-Guided Attention for Breast Cancer Molecular Subtype Classification and Survival Prediction},
  author={Liu, Shuxing},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License