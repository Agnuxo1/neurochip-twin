# NeuroChip Twin: an interpretable temporal digital twin for organ-on-chip phenotype screening

**Submission category:** End-to-End System  
**Author:** Francisco Angulo de Lafuente (`Agnuxo1`)  
**Challenge:** AI4S Open Innovation: AI for Life Science  
**Status:** reproducible research prototype; synthetic validation proxy

## Abstract

Organ-on-chip experiments produce time-resolved microscopy and experimental metadata, but many analysis pipelines collapse the movie into a single endpoint. NeuroChip Twin keeps the temporal signal, adds hydrodynamic exposure context, and exposes the interaction between the two. It segments cell-like objects, associates them between frames, extracts interpretable phenotype trajectories, and combines them with a fixed recurrent reservoir plus a compact physics-informed multimodal readout. Separate heads estimate toxicity, viability and IC50, and a flow counterfactual makes the digital-twin idea testable. The repository contains a deterministic image generator, analysis code, tests, figures, and an HTML demo. On the fixed-seed synthetic benchmark, the multimodal model obtains ROC-AUC 0.998, F1 0.984, viability R² 0.970 and IC50 R² 0.973. These are synthetic stress-test numbers, not clinical performance or real organ-on-chip validation. The project is designed as a license-clean bridge to later evaluation on authorized organ-on-chip data and public microscopy resources such as BBBC, RxRx1, and Cell Painting.

## 1. Problem and impact

Microphysiological systems can reduce animal experimentation and help evaluate drug response in human-relevant environments. The practical bottleneck is often not the absence of images but the loss of dynamics: cell count, morphology, movement, and survival change over time, and a single endpoint can hide transient toxicity or delayed response.

The target workflow is:

`microscopy time-lapse + treatment metadata → cell tracks → phenotype trajectories → response risk + audit trail`

The intended users are OoC researchers and assay developers who need a fast screening signal and an explanation of which measurable phenotype changed. The system is not a diagnostic device and does not replace experimental controls, biological replicates, or statistical review.

## 2. Prior work and contribution

The design reuses ideas from the author's public work while disclosing the adaptation:

- **QESN_MABe_V2_REPO2:** source of the temporal-dynamics and recurrent-state inspiration; it is not copied as a microscopy detector.
- **CAJAL:** source of the scientific-audit and reproducibility orientation; it is not used to generate or alter benchmark results.
- **JEV-Orchestrator:** used as the local decision/control plane for repository-first analysis and evidence checkpoints; no private user data is part of the model.

The new competition-specific contribution is the self-contained cell-phenotype-to-response pipeline, its synthetic stress test, explicit static-vs-temporal ablation, and reproducibility contract.

## 3. Data and compliance

The included benchmark is generated locally by `src/neurochip_twin.py`. It contains no human data, clinical identifiers, copyrighted images, or hidden labels. Each sequence is seeded, and the generator is part of the repository, so the benchmark is reproducible from source.

The generator simulates bright cell-like objects, motion, disappearance, changing morphology, intensity loss, and debris-like background under a treatment perturbation. “Toxicity” is a synthetic latent state used only to test whether the pipeline can recover temporal change.

For a real study, the input contract can be replaced by authorized organ-on-chip microscopy and metadata. The planned public-data validation track is: (1) BBBC or Cell Painting for segmentation/phenotype portability, (2) RxRx1 for perturbation-response representation, and (3) authorized OoC data for the final biological claim. Licences and permitted uses must be checked per dataset and recorded before inclusion.

## 4. System architecture

### 4.1 Pre-processing and segmentation

Each frame is lightly smoothed, thresholded using a robust intensity quantile, and cleaned with binary opening/closing. Connected components become candidate cells. Objects outside an area gate are rejected. For each object we retain centroid, area, mean intensity, and covariance-based elongation.

### 4.2 Tracking

The Hungarian assignment minimizes centroid distance between adjacent frames. A gate rejects assignments beyond 16 pixels. Unmatched objects start new tracks. The result is a table with `(sequence, frame, track_id, centroid, area, intensity, elongation)`.

### 4.3 Phenotype features

For each sequence, the feature vector is:

`[count_ratio, area_ratio, intensity_ratio, elongation_ratio, motility, persistence, count_slope, area_slope, intensity_slope]`

Ratios compare the final and initial frame. Motility is mean centroid displacement, persistence is the fraction of tracks observed for at least half the movie, and slopes are least-squares temporal trends.

### 4.4 Temporal readout

For frame-level phenotype vector `x_t`, the fixed reservoir state is:

`h_t = tanh(W h_(t-1) + W_in x_t)`

`W` is scaled to spectral radius 0.82 and is never trained. A logistic regression readout is trained on the concatenation of the interpretable summary features and the final reservoir state. This separation keeps the temporal memory inspectable and makes the trainable component small.

### 4.5 Physics-informed multimodal fusion

Each sequence also carries dose, flow rate, a documented wall-shear proxy, clearance factor, and effective dose. The final feature vector concatenates temporal phenotype, reservoir state and physics covariates, then adds explicit products between phenotype/reservoir features and shear, clearance, and effective dose. This is a compact, interpretable analogue of multimodal cross-attention: the effect of a phenotype can change with exposure conditions without requiring a large opaque model.

The classifier is accompanied by Ridge regression heads for end-of-sequence viability and IC50. The synthetic generator applies effective exposure attenuation under flow and a high-shear penalty; both are labeled as proxies rather than biological laws.

### 4.6 Uncertainty and auditability

The demo reports a distance-from-0.5 uncertainty proxy for the binary readout. It is deliberately labelled as a proxy, not a confidence interval. A real deployment must replace it with calibration curves, bootstrap intervals, replicate-aware splits, and external validation.

## 5. Experiments

Command:

```powershell
python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180
```

The fixed split has 135 training and 45 held-out sequences. The current generated result is:

| Model | ROC-AUC | Average precision | Balanced accuracy | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| First-frame static baseline | 0.280 | 0.556 | 0.500 | 0.667 | 0.800 |
| Physics-only baseline | 0.998 | 0.999 | 0.983 | 0.978 | 0.983 |
| Temporal fixed reservoir + readout | 0.996 | 0.998 | 0.950 | 0.956 | 0.967 |
| Temporal + physics, no interactions | 0.998 | 0.999 | 0.950 | 0.956 | 0.967 |
| Multimodal physics-informed readout | 0.998 | 0.999 | 0.967 | 0.978 | 0.984 |

The multimodal heads also obtain viability RMSE 5.222/R² 0.970 and IC50 RMSE 0.190/R² 0.973 for this seed. The result is intentionally easy to audit, not presented as a population estimate. The next mandatory experiment is to repeat across at least 10 seeds, add chip-level rather than frame-level splits, and test robustness to blur, illumination drift, object overlap, and missing frames.

The ablation is also a guard against overclaiming: because the current synthetic label is generated directly from effective dose and shear, the physics-only baseline nearly saturates classification. The multimodal architecture is therefore a systems and interpretability upgrade, not yet a demonstrated accuracy improvement. A harder benchmark must introduce compound-specific response variation and nuisance factors before using the score to justify the fusion layer.

The reproducible multi-seed audit now runs ten independent seeds (`python -m src.validation --out outputs/validation --seeds 0 1 2 3 4 5 6 7 8 9`). Multimodal classification gives ROC-AUC mean 0.992 (range 0.973–1.000), average precision mean 0.997, and F1 mean 0.963 (range 0.925–1.000). With the compact, regularized regression heads, viability gives RMSE 5.58 ± 0.79 and R² 0.964 ± 0.013 (minimum 0.932); IC50 gives RMSE 0.203 ± 0.029 and R² 0.968 ± 0.011 (minimum 0.941). The multimodal-versus-temporal ROC-AUC delta averages +0.0053, but ranges from −0.0051 to +0.0177, so the current synthetic benchmark does not justify claiming a universal fusion gain. These are regression-test results only; biological claims require experiment/chip-level grouped splits and external validation.

A grouped acquisition-batch audit (ten seeds, eight sequences per synthetic batch, no batch shared between train and test) gives multimodal ROC-AUC 0.995 ± 0.005 (range 0.985–1.000), F1 0.973 ± 0.017, viability R² 0.962 ± 0.020 (minimum 0.926), and IC50 R² 0.965 ± 0.018 (minimum 0.934). The grouped result is a leakage-resistance diagnostic only: the synthetic batches are not real chips, and the report must not present them as biological validation.

### External front-end portability audit

To test the image-analysis component outside the generator, the repository includes `src.external_validation` for BBBC038v1, a public microscopy dataset with CC0/public-domain images and instance masks. A deterministic sample of 36 cases was split into 12 calibration cases and 24 evaluation cases. The threshold and minimum-area parameters were selected on calibration cases only, then frozen for evaluation. On the 24 held-out images, the unmodified NeuroChip Twin segmentation front-end obtained mean pixel IoU 0.515, Dice 0.573, precision 0.843, recall 0.571, and absolute object-count error 14.9. This is useful evidence that the front-end can be exercised on real microscopy, but it is not organ-on-chip validation, response prediction, or clinical performance. The source page and download URL are persisted in the generated JSON artifact; no external images are committed to the repository.

## 6. Failure modes and safeguards

- **Segmentation bias:** thresholding can merge cells or miss dim cells. The report must show masks and object-count calibration.
- **Synthetic-to-real gap:** generated morphology is not biological ground truth. No clinical or real OoC claim is made.
- **Leakage risk:** sequences from one chip must stay in one split; frame-level random splitting is prohibited for real data.
- **Confounding:** treatment dose can correlate with batch, illumination, or plate position. Real validation must include randomized controls and batch-aware evaluation.
- **Uncertainty misuse:** the current proxy is not a calibrated probability of biological failure.
- **Tracking errors:** crowded fields and divisions can break one-to-one assignment. A real extension needs division-aware or transformer-based association.
- **Synthetic physics:** flow, clearance and shear are transparent proxies; they must be calibrated against measured chip geometry and experimental observations.

## 7. Reproduction and extension

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180
```

All generated outputs are disposable and can be regenerated. The repository does not require an API key, cloud service, proprietary hardware, or private data.

## 8. Planned real-data validation

1. Add an adapter for a legally redistributable microscopy benchmark and preserve raw-data provenance.
2. Validate segmentation against object masks using IoU/F1 and report per-condition variance.
3. Use chip/experiment-level grouped splits and never mix adjacent frames across train/test.
4. Compare static morphology, temporal reservoir, 3D CNN/UNet, and a simple dose-only baseline.
5. Calibrate probabilities and report bootstrap intervals across experiments.
6. Fit dose-response curves and inspect flow counterfactuals against controls.
7. Obtain domain review before interpreting a phenotype as toxicity.

## 9. Conclusion

NeuroChip Twin makes a narrow but testable claim: preserving temporal microscopy dynamics can improve response-state discrimination over a first-frame static baseline in a controlled synthetic proxy. Its value for organ-on-chip research will depend on the real-data validation protocol, not on the synthetic score. The project therefore prioritizes traceability, interpretable features, public reproduction, and explicit limits.
