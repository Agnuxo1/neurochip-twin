# External Dataset Due-Diligence Register

**Snapshot:** 2026-09-23
**Status:** candidate discovery and public-source review only; no external biological measurements were downloaded, inspected, trained on, or evaluated. No preregistration exists.

This register distinguishes promising data sources from evidence actually produced by NeuroChip Twin. Public availability alone is not proof of reuse rights, and organoid data are not organ-on-chip data.

## Highest-priority candidate: human dual neural-organoid drug-screen images

Lu et al. report human peripheral-central dual neural organoids (hPCOs) that
co-develop sensory-ganglion-like and dorsal-spinal-cord-like regions. The
article abstract describes image analysis for teratogenic-risk assessment;
ShanghaiTech's research summary says the team used bright-field morphology and
AI image recognition to screen FDA-approved drugs, with valproic acid and
thalidomide disrupting peripheral-region development. A Mendeley Data record
links raw bright-field drug-screen images to the article and states a CC BY 4.0
license. Its description lists six archive batches spanning treatment IDs
1–220, arranged as two batches for each of three treatment-ID ranges.

This is the strongest newly identified candidate for a *separate human neural
organoid image-analysis audit*, not for validation of organ-on-chip response
prediction. No images, per-image labels, or measurements have been downloaded
or examined. The landing-page metadata do not establish what the treatment IDs
mean, which outcomes are paired to each image, or whether archive batches map
to independent differentiations/acquisitions.

| Due-diligence item | Current finding | Remaining gate |
|---|---|---|
| Biological relevance | Human model with both CNS-like and PNS-like neural regions; bright-field morphology and drug-screening context | It is an organoid model, not an organ-on-chip, and teratogenic-development risk is not interchangeable with acute neural OoC toxicity or response |
| Available material | Mendeley record lists raw bright-field images in six batches for treatment-ID ranges 1–80, 81–160, and 161–220 | Image dimensions, per-image target labels, treatment mapping, and whether images are single time points or longitudinal are not verified |
| Reuse rights | The dataset landing page states CC BY 4.0 | Attribute the dataset and article; confirm the applicable license notice on the exact version used and do not infer rights to unrelated materials or the organoid technology |
| Experimental units | The record distinguishes six archive batches and two batches per treatment-ID range | Batch labels do not prove independent biological replicates, differentiation runs, or acquisition runs; establish group IDs before defining any split |
| Published evaluation | PubMed abstract and institutional research summary describe image-based teratogenicity assessment and AI-based drug screening | Exact target construction, held-out grouping, and performance estimates have not been verified from the full methods or data dictionary |
| Current project status | Metadata-only candidate discovery; no images or measurements used in NeuroChip Twin | No biological model fitting or evaluation until labels, grouping, task scope, and a locked preregistration are verified |

**Decision:** prioritize read-only source/method metadata review and, only after
the preregistration gate, consider a separate static bright-field hPCO
morphology audit. A plausible hybrid design would represent the two
morphologically distinguishable regions separately and fuse their features
only after checking paired labels and acquisition structure. This is a
proposal, not implemented or evaluated. Do not call the archive OoC data,
time-lapse data, or project validation.

**Method-design precedent:** Metzger et al. used a supervised CNN to measure
known control-versus-disease phenotype rescue in micropatterned neural
organoids. For adverse effects whose possible phenotypes were not known in
advance, they instead used a convolutional autoencoder, removed the defined
disease direction from its latent space, and measured Mahalanobis distance
from the control distribution; they compared that readout with an MTT assay.
This supports keeping *known-label phenotype/rescue* and *open-set deviation*
as distinct, separately validated outputs in any future hybrid design. It is
not a result for hPCOs or this project: the study used a different neural
organoid and immunofluorescence images, while the candidate archive is
bright-field and its targets and independent groups remain unverified. The
paper's data are available on request, not a public benchmark for our model.

## Other candidates and boundaries

| Source | Potential value | Main limitation for this project | Status |
|---|---|---|---|
| [CellPhe](https://datadryad.org/dataset/doi:10.5061/dryad.4xgxd25f0) | Temporal microscopy and a paper-described separate-day split for one 2D cell-phenotyping task | 19.32 GB; cancer-cell phenotype/drug classification, not neural toxicity or OoC | Candidate only; not evaluated here |
| [S-BIAD616](https://www.ebi.ac.uk/biostudies/studies/S-BIAD616) | Phase imaging of treated breast-cancer organoids | Not neural or OoC; very large image archive and only limited independent run groups documented | Candidate only; not evaluated here |
| [Monzel et al. human midbrain-organoid study](https://doi.org/10.1016/j.parkreldis.2020.05.011) | HCS and machine-learning-assisted analysis of 6-OHDA-exposed human midbrain organoids | Separate dataset license is unresolved; the reviewed article describes repeated random folds rather than a verified experiment-held-out evaluation | Candidate only; not evaluated here; lower priority until rights and groups are clear |
| [OoC image-quality dataset](https://zenodo.org/records/10203721) | Real OoC brightfield images for sample-quality triage | Six non-neural cell lines, no chip/experiment IDs, and conflicting source license statements; quality labels are not response labels | Existing audit is quality-control only |
| [EPA acute embryonic-rat DRG workbook](https://catalog.data.gov/dataset/kodavantip-acute_hsab_aop_neurotox_science-hub-data-090319) | Separate neural electrophysiology and neurite endpoints | Rat assay, not a human neural OoC or paired imaging-response dataset | Existing exploratory assay audit only |
| [Han et al. human sensory-neuron MPS study](https://doi.org/10.3390/toxics12110809) | Closest located human neural-MPS analog: image-based separation of neurite toxicity and cytotoxicity, with NF-L as an orthogonal assay | Peripheral sensory neurons, not midbrain/CNS organoids; the paper does not link a reusable training-image archive or independent experiment-held-out evaluation | Literature only; data are described as included in the article, with further inquiries to the corresponding author; not evaluated here |
| [BBBC038](https://bbbc.broadinstitute.org/BBBC038) | Instance-mask images for segmentation checks | Nuclei segmentation only; no treatment response or OoC evidence | Segmentation audit only |
| [GSE255606](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255606) | Neural spinal-cord-chip omics candidate | Separate ALS/control transcriptomics task, not paired image-based toxicity response | Candidate only; not evaluated here |

## Gates before any biological evaluation

1. Verify a data-specific license and any restrictions on research use, derived features, and redistribution. Do not infer a data license from the software repository.
2. Review the data dictionary and provenance to identify the biological replicate, iPSC donor/line, organoid differentiation batch, plate/well, acquisition run, and treatment grouping. Keep every lower-level image or field from a held-out experiment on the same side of the split.
3. Define a task the source actually supports (for example, an hPCO teratogenicity-morphology audit or a separately labelled 6-OHDA perturbation-classification audit). Do not relabel either as organ-on-chip response prediction or combine it with unpaired OoC-quality, EPA, BBBC038, or synthetic rows.
4. Lock a prospective analysis plan before measurement-level use: cohort/version, inclusion rules, feature construction, grouping unit, untouched test groups, model/baseline, metrics, uncertainty, and stop criteria. Record the actual registration identifier and timestamp; until then, do not download or fit/evaluate biological measurements.
5. If all gates pass, report the result as a separate neural-organoid audit, with the number of independent groups and limitations explicit. It cannot by itself establish neural OoC or clinical validity.

## Primary references

- Lu et al., 2026, [Developmental Cell article record](https://pubmed.ncbi.nlm.nih.gov/42759501/), DOI [10.1016/j.devcel.2026.08.013](https://doi.org/10.1016/j.devcel.2026.08.013); [Mendeley Data record](https://data.mendeley.com/datasets/bm3726x9kp/1), DOI [10.17632/bm3726x9kp.1](https://doi.org/10.17632/bm3726x9kp.1), CC BY 4.0; [ShanghaiTech research summary](https://www.shanghaitech.edu.cn/en/2026/0921/c1260a1127488/page.htm).
- Metzger et al., 2022, [open-access neural-organoid phenotypic-screen study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9500000/), DOI [10.1016/j.crmeth.2022.100297](https://doi.org/10.1016/j.crmeth.2022.100297). Its images/data are not a public benchmark; its open article license is CC BY-NC-ND 4.0.
- Monzel et al., 2020, [author preprint and institutional record](https://orbilu.uni.lu/handle/10993/43214), [PubMed record](https://pubmed.ncbi.nlm.nih.gov/32534431/), DOI [10.1016/j.parkreldis.2020.05.011](https://doi.org/10.1016/j.parkreldis.2020.05.011).
- Han et al., 2024, [human iPSC sensory-neuron MPS study](https://doi.org/10.3390/toxics12110809); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598741/). This is a directly relevant design precedent, not a reusable external benchmark: its data statement points to material in the article and author inquiry rather than a linked image-data repository.
- [University of Luxembourg public data directory](https://webdav.lcsb.uni.lu/public/data/machine-learning-assisted-neurotoxicity-prediction-in-human-midbrain-organoid/) and its [HCS imaging file listing](https://webdav.lcsb.uni.lu/public/data/machine-learning-assisted-neurotoxicity-prediction-in-human-midbrain-organoid/HCS%20imaging%20data/).
- [Authors' analysis-code repository](https://github.com/LCSB-DVB/Monzel_2020) (Apache-2.0 code license; not a data license).
