# Candidate external cohorts for multi-cohort training — VERIFIED

**Provenance.** Every accession below was resolved against NCBI GEO on 2026-08-06 via
E-utilities (`esearch`/`esummary`, db=gds) and by streaming the series-matrix headers from
`ftp.ncbi.nlm.nih.gov/geo/series/.../matrix/`. Sample counts and platform IDs are read from
those records, not from memory. Annotation status is read from the `!Sample_characteristics_ch*`
lines of the series matrix.

**Why this file exists.** An earlier recommendation named `GSE96058` and `GSE20685` as additional
training cohorts from recall alone. Re-checking showed one of those two does not carry PAM50
labels at all, so it could not have served as a supervised training cohort. Do not cite a GEO
accession as a training cohort without checking its `!Sample_characteristics_ch*` lines first —
"series exists" and "series is PAM50-labelled" are different claims.

## Verified accessions

| Accession | Real | Platform | n samples | PAM50 subtype labels in GEO? | Usable as |
|---|---|---|---|---|---|
| GSE96058 (SCAN-B) | yes | RNA-seq — GPL11154 (HiSeq 2000) + GPL18573 (NextSeq 500), split matrices | 3,409 | **No** — characteristics are only `scan-b external id` and `instrument model`. PAM50 for this cohort comes from the source publication, not the GEO record. | External cohort *if* labels are obtained from the publication or re-derived |
| GSE21653 | yes | microarray | 266 | **Yes** — `molecular subtype: LuminalA / LuminalB / ERBB2 / Normal / Basal` | Supervised training or validation cohort (5-class, matches our label set minus claudin-low) |
| GSE20685 | yes | microarray | 327 | **No** — `subtype:` field holds `type I / II / III / IV / VI`, a private clustering, **not** PAM50 | Nothing for subtyping. Usable only for unsupervised checks. |
| GSE20711 | yes | methylation + expression | 90 | not verified | — |
| GSE2034 | yes | GPL96 (Affymetrix U133A) | 286 | **No** — relapse-site fields only (`bone relapses`, etc.) | Survival endpoint only, not subtyping |

## Consequence for the manuscript

1. **`paper_cn_v2.tex` line 496** lists `GSE2034` among cohorts for external validation of
   subtype generalization. That dataset has no subtype labels — it supports a survival/prognosis
   claim only. Reword or drop it if that draft is used.
2. **`GSE21653` (n=266)** is the one candidate here with PAM50-style labels already in GEO, so it
   is the lowest-friction second training cohort. Caveat: it is microarray, so it enters through
   the same within-platform rank normalization already used for TCGA, and it is small relative to
   METABRIC (1,974) — expect it to shift CV by roughly a point, not transform performance.
3. **`GSE96058` (n=3,409)** is the largest and most attractive cohort, but it is only usable if
   PAM50 labels are sourced from the SCAN-B publication or re-derived from expression. That is
   real work, not a download, and should be scoped before it is promised in a revision plan.
4. None of these cohorts carry the `Claudin-low` class, which METABRIC has and TCGA lacks. Adding
   a cohort with a different label vocabulary than the training set changes the classification
   task and must be handled explicitly rather than pooled silently.
