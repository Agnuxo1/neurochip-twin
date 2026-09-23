# External Dataset Due-Diligence Register

**Snapshot:** 2026-09-23
**Status:** candidate discovery and public-source review only; no external biological measurements were downloaded, inspected, trained on, or evaluated. No preregistration exists.

This register distinguishes promising data sources from evidence actually produced by NeuroChip Twin. Public availability alone is not proof of reuse rights, and organoid data are not organ-on-chip data.

## Highest-priority candidate: human midbrain-organoid neurotoxicity

Monzel et al. describe high-content imaging and machine-learning-assisted analysis of human midbrain organoids exposed to 6-hydroxydopamine (6-OHDA). Their report discusses dopaminergic-neuron counts and neuronal complexity and states that data are openly available. It reports organoids from three independent human iPSC lines. The public University of Luxembourg directory contains an HCS imaging folder plus `Data.csv`, `Data_PD.csv`, `Data_TOX.csv`, and overview spreadsheets. The paper's described classifier used repeated random five-fold splits; the reviewed sources do not establish a held-out organoid-batch or experiment-level test.

| Due-diligence item | Current finding | Remaining gate |
|---|---|---|
| Biological relevance | Human neural midbrain organoids and a neurotoxic perturbation; closer to neural toxicity than non-neural OoC quality images or rat DRG electrophysiology | This is an organoid study, not an organ-on-chip model, and does not validate NeuroChip Twin's OoC response predictions |
| Available material | The public HCS directory lists a `StitchedImages/` folder, three small CSV tables (29–33 KB), and two 20 KB overview workbooks | Full image inventory, dimensions, labels, and provenance have not been checked |
| Reuse rights | The public code repository displays Apache-2.0; the data directory/paper reviewed here do not state a separate dataset license | Obtain and document explicit data reuse rights before downloading, redistribution, or analysis; the code license does not license the data |
| Experimental units | The paper reports three independent iPSC lines | Line count does not prove independent differentiation, plate, organoid, or imaging-run groups; a valid held-out grouping has not been verified |
| Published evaluation | The article/preprint describes ten repetitions of random five-fold cross-validation | It does not establish an untouched experiment-held-out evaluation; reproduce no published score and do not assume its folds prevent leakage |
| Current project status | Discovery candidate only; not included in any NeuroChip Twin metric or model | No biological model fitting or evaluation until rights, experimental-unit metadata, and a locked preregistration are in place |

**Decision:** prioritize read-only metadata and rights due diligence for this candidate. Do not present it as validation, merge it with the synthetic benchmark or other unpaired assays, or use its paper-reported cross-validation score as a NeuroChip Twin result. Even a future grouped organoid audit would be neural-organoid evidence only—not neural organ-on-chip validation.

## Other candidates and boundaries

| Source | Potential value | Main limitation for this project | Status |
|---|---|---|---|
| [CellPhe](https://datadryad.org/dataset/doi:10.5061/dryad.4xgxd25f0) | Temporal microscopy and a paper-described separate-day split for one 2D cell-phenotyping task | 19.32 GB; cancer-cell phenotype/drug classification, not neural toxicity or OoC | Candidate only; not evaluated here |
| [S-BIAD616](https://www.ebi.ac.uk/biostudies/studies/S-BIAD616) | Phase imaging of treated breast-cancer organoids | Not neural or OoC; very large image archive and only limited independent run groups documented | Candidate only; not evaluated here |
| [OoC image-quality dataset](https://zenodo.org/records/10203721) | Real OoC brightfield images for sample-quality triage | Six non-neural cell lines, no chip/experiment IDs, and conflicting source license statements; quality labels are not response labels | Existing audit is quality-control only |
| [EPA acute embryonic-rat DRG workbook](https://catalog.data.gov/dataset/kodavantip-acute_hsab_aop_neurotox_science-hub-data-090319) | Separate neural electrophysiology and neurite endpoints | Rat assay, not a human neural OoC or paired imaging-response dataset | Existing exploratory assay audit only |
| [Han et al. human sensory-neuron MPS study](https://doi.org/10.3390/toxics12110809) | Closest located human neural-MPS analog: image-based separation of neurite toxicity and cytotoxicity, with NF-L as an orthogonal assay | Peripheral sensory neurons, not midbrain/CNS organoids; the paper does not link a reusable training-image archive or independent experiment-held-out evaluation | Literature only; data are described as included in the article, with further inquiries to the corresponding author; not evaluated here |
| [BBBC038](https://bbbc.broadinstitute.org/BBBC038) | Instance-mask images for segmentation checks | Nuclei segmentation only; no treatment response or OoC evidence | Segmentation audit only |
| [GSE255606](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255606) | Neural spinal-cord-chip omics candidate | Separate ALS/control transcriptomics task, not paired image-based toxicity response | Candidate only; not evaluated here |

## Gates before any biological evaluation

1. Verify a data-specific license and any restrictions on research use, derived features, and redistribution. Do not infer a data license from the software repository.
2. Review the data dictionary and provenance to identify the biological replicate, iPSC donor/line, organoid differentiation batch, plate/well, acquisition run, and treatment grouping. Keep every lower-level image or field from a held-out experiment on the same side of the split.
3. Define a task the source actually supports (for example, a separately labelled 6-OHDA perturbation-classification audit). Do not relabel it as organ-on-chip response prediction or combine it with unpaired OoC-quality, EPA, BBBC038, or synthetic rows.
4. Lock a prospective analysis plan before measurement-level use: cohort/version, inclusion rules, feature construction, grouping unit, untouched test groups, model/baseline, metrics, uncertainty, and stop criteria. Record the actual registration identifier and timestamp; until then, do not download or fit/evaluate biological measurements.
5. If all gates pass, report the result as a separate neural-organoid audit, with the number of independent groups and limitations explicit. It cannot by itself establish neural OoC or clinical validity.

## Primary references

- Monzel et al., 2020, [author preprint and institutional record](https://orbilu.uni.lu/handle/10993/43214), [PubMed record](https://pubmed.ncbi.nlm.nih.gov/32534431/), DOI [10.1016/j.parkreldis.2020.05.011](https://doi.org/10.1016/j.parkreldis.2020.05.011).
- Han et al., 2024, [human iPSC sensory-neuron MPS study](https://doi.org/10.3390/toxics12110809); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598741/). This is a directly relevant design precedent, not a reusable external benchmark: its data statement points to material in the article and author inquiry rather than a linked image-data repository.
- [University of Luxembourg public data directory](https://webdav.lcsb.uni.lu/public/data/machine-learning-assisted-neurotoxicity-prediction-in-human-midbrain-organoid/) and its [HCS imaging file listing](https://webdav.lcsb.uni.lu/public/data/machine-learning-assisted-neurotoxicity-prediction-in-human-midbrain-organoid/HCS%20imaging%20data/).
- [Authors' analysis-code repository](https://github.com/LCSB-DVB/Monzel_2020) (Apache-2.0 code license; not a data license).
