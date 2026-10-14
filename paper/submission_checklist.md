# Paper Submission Checklist

## Paper: Label-Space Mismatch Distorts Cross-Cohort Evaluation in Breast Cancer Molecular Subtyping
**Target journal**: IEEE Journal of Biomedical and Health Informatics (JBHI) — JCR Q1 (Computer Science, Information Systems), IF ~6.7 (2023)
**Fallbacks**: Briefings in Bioinformatics / Bioinformatics / Frontiers in Genetics

> **Journal fit note.** Do not submit to *International Journal of Mechanical Sciences*. IJMS is a
> mechanical/civil/materials engineering journal (WoS categories ENGINEERING, MECHANICAL and
> MECHANICS; scope = rigid-body and structural mechanics, metal forming, fluid mechanics,
> tribology, thermodynamics, materials processing). Its own guide for authors states it "does not
> accept for publication material which quite specifically belongs with other established
> specialist journals," and that "design-based studies... are outside our scope." A breast cancer
> PAM50 genomics paper would be desk-rejected out of scope, without review — this is a scope
> mismatch, not a quality judgement, and no amount of revision can fix it.

> **Page budget.** JBHI regular papers are 8–12 double-column pages (over-length charges apply
> beyond 12). Current manuscript: 36 single-column double-spaced pages ≈ 8,450 body words plus
> 15 figures and 10 tables. In IEEE two-column format that lands at roughly 11–14 pages, so the
> figure set needs trimming — move the Transformer-investigation figures (11–15) and the
> lower-value ablations to the supplement.

---

## Files

### Main Manuscript
- [x] `paper/manuscript.tex` — LaTeX source; all numbers traced to a `results/` file
- [x] Table `tab:tcga_confusion` — raw 5x6 TCGA confusion matrix (987 samples, diag 728); the
      claudin-low column (75 predictions) is printed explicitly because it is the mechanism behind
      the label-space argument and is otherwise invisible in per-class metrics
- [x] `paper/references.bib` — **84 BibTeX references**, every one cited, every one resolved.
      All keys were audited against the body text: 59 citations previously failed to match their
      entries (the `.bib` keys carried an appended entry-index suffix, e.g. `mao2025a21` vs
      `\cite{mao2025a}`), so only 24 of 84 entries ever reached the PDF. Keys are now canonical and
      unique; duplicate (`garg2022domain7` = `garg2022open`) and dangling entries (author-name typo
      `miao2025a` -> `mao2025a`; two headings with no matching work -> replaced by Steyaert *et al.*
      *Nat. Mach. Intell.* 2023 and Wu *et al.* *NAR Genom. Bioinform.* 2023) were fixed rather than
      dropped. All 8 added/verified DOIs were checked against CrossRef; non-ASCII author accents are
      stored as LaTeX escapes because BibTeX splits multi-byte UTF-8 when abbreviating given names.
- [x] `paper/make_figH.py` — label-shift figure generator (panels A–C)
- [x] `paper/make_figI.py` — treatment-response figure generator (panels A–B)
- [x] `paper/generate_figures.py` — main figure set (reads measured values, no synthetic data)

### Supplementary Materials
- [x] `paper/supplementary/supplementary.tex` — S1–S8 sections with complete tables

### Figures — 16 PDFs on disk, all 16 embedded
- [x] `figures/fig1_modality_ablation.pdf` — Modality combination bars with dimension counts
- [x] `figures/fig2_confusion_metabric.pdf` — METABRIC test confusion matrix (real counts)
- [x] `figures/fig3_tcga_results.pdf` — TCGA-BRCA confusion matrix + KM curves (real)
- [x] `figures/fig4_shap_analysis.pdf` — SHAP modality pie + top-10 feature bar
- [x] `figures/fig5_model_comparison.pdf` — Model comparison horizontal bars
- [x] `figures/fig6_cross_platform.pdf` — mRNA-only baseline across the three cohorts
- [x] `figures/figA_learning_curves.pdf` — Transformer learning curve vs LogReg reference
- [x] `figures/figB_capacity_sweep.pdf` — 12-configuration capacity sweep
- [x] `figures/figC_regularization.pdf` — Single-axis regularization sweep
- [x] `figures/figD_pretraining.pdf` — Masked-feature pretraining effect
- [x] `figures/figE_tcga_validation.pdf` — Internal vs external accuracy + external C-index
- [x] `figures/figF_treatment_auc.pdf` — METABRIC treated-subgroup 5-year outcome AUC
- [x] `figures/figG_second_cohort.pdf` — Three-panel cohort figure (GSE21653)
- [x] `figures/figH_label_shift.pdf` — label-shift decomposition / identifiability / simulation
- [x] `figures/figI_pcr.pdf` — pCR AUCs + pCR rate by predicted subtype (GSE20194)
- [x] `figures/figJ_multimodal_external.pdf` — modality ablation in the harmonized
      METABRIC/TCGA space (internal vs external, 4 configurations)

### Data/Code
- [x] `best_model/` — trained model, prediction script, documentation
- [x] `results/paper_final_numbers.md` — verified-numbers registry
- [x] `results/route_a/` — 26 JSON result files (see Route A section below)
- [x] `results/tcga_validation/paper_numbers.json` — TCGA external-validation numbers (n=987)
- [x] `scripts/route_a_step*.py` — one script per experiment, each writing its JSON
- [x] `scripts/route_a_step15_harmonize.py` — builds the 3-layer METABRIC/TCGA shared space
- [x] `scripts/route_a_step16_multimodal_external.py` — modality ablation in that space (Fig. J)
- [x] `scripts/route_a_step17_detector_v2.py` — calibrated absent-class detector, per-cell scores
- [x] `scripts/fetch_tcga_methylation.py` — cBioPortal HM450 gene-level fetch (TCGA-BRCA)
- [x] `paper/make_figJ.py` — harmonized-ablation figure generator
- [x] `README.md` — rewritten: pipeline table, key findings, build order, data sources

---

## Key Numbers (Verified)

| Metric | Value | Source |
|--------|-------|--------|
| METABRIC 5-fold CV (3-modality) | 75.3% ± 2.6% | paper_final_numbers.md |
| METABRIC Val accuracy | 79.0% | paper_final_numbers.md |
| METABRIC Test accuracy (3-mod) | 73.8% | figure_data.json |
| METABRIC Test accuracy (mRNA-only) | 73.79% (macro F1 0.7060) | mrna_only_crossplatform.json |
| TCGA-BRCA accuracy | 73.76% | figure_data.json |
| TCGA macro F1 (6-class, absent claudin-low scored 0) | 0.5847 | figure_data.json |
| TCGA macro F1 (5 PAM50 classes actually present) | 0.7017 | recomputed from confusion in figure_data.json |
| Label-space penalty on identical predictions | 0.117 (= 0.7017 − 0.5847) | figure_data.json |
| **TCGA after label-space correction** | **80.65%** (macro F1 0.7367) | crosscohort_shift_robust.json |
| GSE21653 accuracy | 63.16% | gse21653_validation.json |
| **GSE21653 after correction** | **74.06%** (macro F1 0.6834) | crosscohort_shift_robust.json |
| Bootstrap gain | +6.88 pts, 95% CI [+4.96, +8.81] | crosscohort_shift_robust.json |
| EM claudin-low prior on TCGA (true 0) | 0.0751 | shift_decomposition.json |
| Detector calibration-set F1 (precision / recall) | 0.367 (0.257 / 0.643) | detector_sim.json |
| Geometry detector, mean simulated F1 vs EM | 0.047 vs 0.549 | detector_calibrated.json |
| Geometry detector, τ* / null max median / max | 1.0349 / 1.0184 / 1.0493 | detector_calibrated.json |
| Geometry detector, gain vs uncorrected model | mean +3.34 pts, min −2.77 (worst cell a clean cohort, 73.98%→71.22%) | detector_calibrated.json |
| Geometry detector, gain vs EM | mean −0.37 pts, max +0.40, min −2.96 | detector_calibrated.json |
| Geometry detector, cells vs EM | 11 win, 9 lose, 10 tie (of 30) | detector_calibrated.json |
| BIC detector, mean simulated F1 vs EM | 0.209 vs 0.549 | detector_lrt.json |
| BIC detector, cells beating EM | 0/30 win, 30/30 lose, mean −17.6 pts | detector_lrt.json |
| BIC detector, null false-positive rate | 0.90 (27/30 runs non-empty; 23 claim two classes) | detector_lrt.json |
| Detector-v2, cohorts simulated | 720 (20 seeds x 6 absent sets x 3 cov shifts x 2 prior shifts) | detector_v2.json |
| Uncalibrated EM, micro P / R / F1 | 0.920 / 0.685 / 0.786 | detector_v2.json |
| Uncalibrated EM, exact-set match | 0.550 | detector_v2.json |
| Uncalibrated EM, null false positives | 17/120 | detector_v2.json |
| Calibrated EM, micro P / R / F1 | 0.577 / 0.876 / 0.696 | detector_v2.json |
| Calibrated EM, null false positives | 73/120 (20/20 at every cov shift when prior shift present) | detector_v2.json |
| Importance-weighted EM, micro F1 | 0.542 (null FP 56/120) | detector_v2.json |
| Probability-frequency, micro F1 | 0.682 (null FP 87/120; 20/20 under prior shift) | detector_v2.json |
| Geometric coherence, micro F1 | 0.232 (null FP 27/120) | detector_v2.json |
| Naive (never predicted), micro P / exact | 1.000 / 0.167 (null FP 0/120) | detector_v2.json |
| Uncalibrated EM, mean F1 no shifts -> max shift | 0.995 -> 0.543 | detector_v2.json |
| TCGA C-Index | 0.577 ($p=0.047$) | route_a/tcga_survival.json |
| TCGA C-Index, time-dependent (3y/5y/10y) | 0.587 / 0.574 / 0.577 | route_a/tcga_survival.json |
| TCGA confusion matrix, diagonal | 728 / 987 | independently recomputed; Table `tab:tcga_confusion` |
| TCGA samples predicted claudin-low (a class absent from TCGA) | 75 (7.6%) | recomputed; = 15 LumA + 3 LumB + 3 Her2 + 38 Basal + 16 Normal |
| GSE21653 C-Index | 0.584 ($p=0.011$) | gse21653_validation.json |
| METABRIC C-Index | 0.598, log-rank $p=0.0008$ | paper_final_numbers.md |
| pCR AUC clinical / genomic / combined | 0.791 / 0.773 / 0.794 | pcr_gse20194.json |
| pCR Δ genomic over clinical | +0.0045, 95% CI [−0.060, +0.061] (NS) | pcr_gse20194.json |
| SHAP mRNA / methylation / clinical | 63.3% / 26.7% / 10.0% | shap_by_subtype.png |
| **Harmonized multi-layer TCGA cohort** | **n=694** (5 PAM50 classes; claudin-low absent) | harmonization.json |
| Harmonized mRNA-only TCGA accuracy | 80.55%, 95% CI [77.5, 83.4] | multimodal_external.json |
| Harmonized +clinical TCGA accuracy | 80.40%, 95% CI [77.4, 83.3]; McNemar *p*=1.00 | multimodal_external.json |
| Harmonized +methylation TCGA accuracy | 78.67%, 95% CI [75.5, 81.7]; McNemar *p*=0.13 | multimodal_external.json |
| Harmonized tri-modal TCGA accuracy | 79.11%, 95% CI [75.9, 82.0]; McNemar *p*=0.30 | multimodal_external.json |
| Harmonized tri-modal METABRIC CV / test | 73.17% / 74.76% | multimodal_external.json |
| Original 713-dim flagship METABRIC CV | 75.92% (not transportable) | multimodal_external.json |
| TCGA methylation genes measured | 209 of 232 (23 absent from array) | harmonization.json |
| TCGA C-Index, superseded | ~~0.586~~ (p=0.013) | not reproducible; see Open items |

---

## Before Submission

- [x] Compile LaTeX — **DONE on the host**, 36 pages, 0 undefined citations.
      The blocker was only ever the platform sandbox. The host has a working TinyTeX
      (TeX Live 2026) at `~/Library/TinyTeX/bin/universal-darwin/`, which is *not* on PATH —
      so a bare `pdflatex` appears missing. Run `cd paper && ./build.sh`; it sets PATH itself
      and refuses to ship if any citation/reference is undefined.
- [x] Generate all publication figures from `make_fig*.py` / `generate_figures.py`
- [x] Convert figures to PDF/vector format, all 15 embedded and verified present
- [x] Cross-reference integrity: 0 dangling `\ref`, 0 duplicate `\label`, 0 double-embedded figures
- [ ] Remove placeholder author names/affiliations
- [ ] Add ORCID and corresponding author email
- [ ] Replace the repository URL placeholder (`github.com/[username]/...`) in the abstract
- [ ] Write cover letter — lead with the label-space finding, not the classifier
- [ ] Prepare highlights (3–5 bullets)
- [ ] Consider submitting `data/` gene panels as supplementary data
- [ ] Check JBHI-specific formatting: it is a two-column IEEEtran template, current source is
      single-column `article`. Template conversion is required before submission and will need
      figure widths re-tuned.

---

## Experimental Completeness Summary

| Requirement | Status | Notes |
|-------------|--------|-------|
| Internal validation (METABRIC test) | ✅ | 73.8% acc, per-class F1 in Table 3 |
| External validation (TCGA-BRCA) | ✅ | n=987, 73.76% acc, per-class F1 in Table 4 |
| Second external cohort (GSE21653) | ✅ | n=266, 63.16% acc, independent platform |
| **Label-space correction** | ✅ | +6.88 pts TCGA, +10.90 pts GSE21653, label-free by construction |
| **Identification of correction non-identifiability** | ✅ | EM retains 0.0751 false prior; recovers 2.2 of 6.9 pts |
| **Controlled simulation of shift components** | ✅ | 3×3×2 factorial, n=20 seeds |
| **Detector null calibration (negative result)** | ✅ | near-zero power; reported as a negative result |
| Survival analysis | ✅ | C-Index 0.598 (METABRIC), 0.577 / 0.584 (external) |
| SHAP interpretability | ✅ | Modality + feature level |
| Model ablation (6 algorithms) | ✅ | LogReg best, Table 2 |
| Modality ablation (10 combos) | ✅ | 3-modality best, Table 1 |
| Feature selection | ✅ | 200 genes sufficient, Supp. S3 |
| Treatment-subgroup prognosis | ✅ | METABRIC, 3 subgroups, 5-year outcome |
| **Treatment-response prediction (pCR)** | ✅ | GSE20194 n=278; panel adds no significant increment |
| Cross-platform normalization | ✅ | Within-platform rank, Methods 2.2 |

---

## Route A Methodological Investigation

**Purpose**: test whether cross-modal attention fusion beats the linear reference, and whether any
internal gain survives external validation. Outcome: a documented negative result, plus the
label-space finding that became the paper's main line.

### Results files (`results/route_a/`)
- [x] `route_a_summary.json` — consolidated verdict
- [x] `reference_check.json` — harness rebuild; LogReg 713d CV reproduced exactly
- [x] `diagnosis.json` — capacity sweep + learning curve + bias/variance verdict
- [x] `regularized_cv.json` — 11 single-axis configs + best combination
- [x] `pretrain_cv.json` — scratch vs masked-feature pretrained (per-fold, no leakage)
- [x] `tcga_best.json`, `tcga_survival.json` — external accuracy and C-index, three models
- [x] `treatment_response.json` — treated-subgroup prognosis
- [x] `crosscohort_shift.json`, `crosscohort_shift_robust.json` — label-shift correction
- [x] `shift_decomposition.json` — base / mask_only / em_pam50 / em_full decomposition
- [x] `sim_mismatch.json` — 3-factor controlled simulation
- [x] `detector_sim.json`, `detector_lrt.json`, `detector_calibrated.json` — failed detector
- [x] `detector_v2.json` (78 KB) + `detector_v2_raw.npz` — empty-set-convention re-scoring; raw sets kept so the metric can be re-examined without re-running the simulation
- [x] `gse21653_validation.json`, `multicohort_cv.json` — second cohort
- [x] `pcr_gse20194.json` — treatment response
- [x] `figure_data.json` — governing source for Figures 2/3/6

### Headline numbers
| Quantity | Value |
|---|---|
| LogReg reference CV | 75.26% |
| Best Transformer CV (pretrained) | 70.04% |
| Internal gap | 5.22 points |
| LogReg TCGA accuracy | 73.76% |
| Best Transformer TCGA accuracy | 62.72% |
| External gap | 11.04 points |
| **TCGA after label-space reconciliation** | **80.65%** |
| **GSE21653 after label-space reconciliation** | **74.06%** |
| TCGA C-index (linear / TF / TF+pre) | 0.577 / 0.551 / 0.530 |

### Manuscript integration
- [x] Title and abstract rewritten around label-space mismatch
- [x] New Results section on label-space mismatch, decomposition, simulation (with `figH`)
- [x] New Methods subsections: label-space reconciliation, controlled simulation, absence detector
- [x] New Results section: pCR treatment response (with `figI`)
- [x] Transformer negative result demoted to supporting evidence (Section 7)
- [x] Tables: `tab:route_a`, `tab:treatment`, `tab:tcga_per_class`, `tab:multicohort`, `tab:cox`
- [x] Structural validation: balanced environments, 0 dangling refs, 15/15 figures present

---

## Open items

- [x] **RESOLVED — the manuscript compiles.** Earlier note said "cannot be compiled"; that was
      true only of the *platform sandbox*, whose `tex` conda env ships TeX Live 2026 binaries over
      a truncated macro tree (`texmf-dist` 7.3 MB, no `tex/latex`, no `article.cls`, no
      `pdflatex.fmt`). The **host is fine**: TinyTeX 2026 lives at
      `~/Library/TinyTeX/bin/universal-darwin/` (off PATH, which is why `which pdflatex` fails and
      it looks like LaTeX is missing) with a complete `texmf-dist`. `./build.sh` →
      `build/manuscript.pdf`, **36 pages, 883,769 bytes, 0 undefined citations/references**.
      A second, larger defect surfaced once BibTeX actually ran: **59 of 84 citations never
      resolved**, because the `.bib` keys carried the entry's index suffix while the body text used
      the bare author-year form (`\cite{mao2025a}` vs key `mao2025a21`). Only 24 references reached
      the PDF. Fixed by canonicalising all keys against the citations (see Files -> `references.bib`).
      Confirmed clean: `manuscript_ieee.tex` builds to **17 pages, 550,775 bytes, 84 references,
      0 undefined citations, 0 undefined references, 0 LaTeX errors**; the companion
      `manuscript.tex` builds to 36 pages, 0 undefined citations. tectonic is still not an option
      here — its bundle host `relay.fullyjustified.net` is not on the sandbox allowlist.
- [x] `figF` caption corrected. It plots 5-year *outcome* among treated METABRIC patients, not
      pCR. The pCR claim now has its own figure (`figI`, from `pcr_gse20194.json`) with a caption
      matching its content, and `figF` was restored to the treated-subgroups section with its
      `\label{fig:treatment}` so the `\ref` at that section resolves.
- [x] **TCGA survival figures unified.** The original analysis reported C-index 0.586 ($p=0.013$)
      under a risk-score construction not preserved in the codebase. The manuscript reports the
      reproducible pipeline (Cox fit on METABRIC train, applied to TCGA) — 0.577 ($p=0.047$),
      time-dependent 0.587/0.574/0.577 — with an explicit disclosure sentence. Do not
      reintroduce 0.586.
- [x] **RESOLVED — two-column IEEEtran manuscript exists.** `paper/manuscript_ieee.tex`
      (`\documentclass[10pt,journal,compsoc,twocolumn]{IEEEtran}`) is a full JBHI-format build:
      17 pages, 0 undefined citations/references, 0 LaTeX errors. Every `\includegraphics` width was
      re-tuned to the 3.5 in single-column measure via the `\fitcol` helper, and the wide tables were
      scaled with `\AtBeginEnvironment{table}{\small}`. `IEEEtran.cls` + `IEEEtran.bst` are installed
      in `~/Library/TinyTeX/texmf-local/` (not in a repo `texmf/`), so the class resolves out of the box.
      Note the build must run `pdflatex -> bibtex -> pdflatex -> pdflatex` in that order; `latexmk`
      mis-sequences BibTeX against the freshly-written `.aux` on this TinyTeX install.
- [x] **RESOLVED — affiliation/author block carries real values**: Faculty of Medicine and Health
      Sciences, Universiti Sains Malaysia, Kubang Kerian, Kelantan, Malaysia; corresponding author
      Shuxing Liu. Data-availability (cBioPortal/GEO), code-availability (GitHub + Zenodo DOI pending),
      author-contributions, funding, competing-interests and ethics statements are all present in the
      Declarations block.

## Pre-submission residue
- [ ] Zenodo DOI is promised in the code-availability statement but not yet minted; mint it from the
      archived GitHub snapshot (commit `24c3ed9`) before submitting, or soften the sentence.

