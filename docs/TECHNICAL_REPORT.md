# NeuroChip Twin: an interpretable temporal digital twin for organ-on-chip phenotype screening

**Submission category:** End-to-End System  
**Author:** Francisco Angulo de Lafuente (`Agnuxo1`)  
**Challenge:** AI4S Open Innovation: AI for Life Science  
**Status:** reproducible research prototype; synthetic validation proxy

## Abstract

Organ-on-chip experiments produce time-resolved microscopy and experimental metadata, but many analysis pipelines collapse the movie into a single endpoint. NeuroChip Twin keeps the temporal signal, adds hydrodynamic exposure and compound context, and exposes the interaction between the two. It segments cell-like objects, associates them between frames, extracts interpretable phenotype trajectories, and combines them with a fixed recurrent reservoir plus a compact physics-informed multimodal readout. Separate heads estimate toxicity, viability and IC50, and a flow counterfactual makes the digital-twin idea testable. The repository contains a deterministic compound-specific stress-test generator, analysis code, tests, figures, and an HTML demo. On the fixed-seed synthetic benchmark, the multimodal model obtains ROC-AUC 0.990, F1 0.958, viability R² 0.957 and IC50 R² 0.957. These are synthetic stress-test numbers, not clinical performance or real organ-on-chip validation. The project is designed as a license-clean bridge to later evaluation on authorized organ-on-chip data and public microscopy resources such as BBBC, RxRx1, and Cell Painting.

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

The new competition-specific contribution is the self-contained cell-phenotype-to-response pipeline, its compound-specific synthetic stress test, explicit static-vs-temporal/multimodal ablations, and reproducibility contract.

## 3. Data and compliance

The included benchmark is generated locally by `src/neurochip_twin.py`. It contains no human data, clinical identifiers, copyrighted images, or hidden labels. Each sequence is seeded, and the generator is part of the repository, so the benchmark is reproducible from source.

The generator simulates bright cell-like objects, motion, disappearance, changing morphology, intensity loss, and debris-like background under a treatment perturbation. The primary `compound_specific` scenario adds seeded compound descriptors and a latent sequence-level susceptibility that is observable through the time series but not exposed directly to the physics-only ablation. The older `exposure_only` scenario remains available as a control. “Toxicity” is a synthetic latent state used only to test whether the pipeline can recover temporal change.

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

Each sequence carries dose, flow rate, a documented wall-shear proxy, clearance factor, effective dose, and three seeded compound-context descriptors. The final feature vector concatenates temporal phenotype, reservoir state and context covariates, then adds explicit products between phenotype/reservoir features and shear, clearance, effective dose, and compound context. This is a compact, interpretable analogue of multimodal cross-attention: the effect of a phenotype can change with exposure conditions and compound context without requiring a large opaque model.

The classifier is accompanied by Ridge regression heads for end-of-sequence viability and IC50. The synthetic generator applies effective exposure attenuation under flow and a high-shear penalty; both are labeled as proxies rather than biological laws.

### 4.6 Uncertainty and auditability

The demo reports a distance-from-0.5 uncertainty proxy for the binary readout and stores Brier score plus ten-bin expected calibration error (ECE) in `metrics.json`. For seed 42 the multimodal readout gives Brier 0.041 and ECE 0.064; the ten-seed audit averages 0.053 and 0.069, and the grouped audit averages 0.048 and 0.071. These are calibration diagnostics for the synthetic proxy, not confidence intervals. A real deployment must replace them with calibration curves, bootstrap intervals, replicate-aware splits, and external validation.

## 5. Experiments

Command:

```powershell
python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180 --scenario compound_specific
```

The fixed split has 135 training and 45 held-out sequences. The current generated result is:

| Model | ROC-AUC | Average precision | Balanced accuracy | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| First-frame static baseline | 0.567 | 0.620 | 0.580 | 0.600 | 0.700 |
| Physics-only baseline | 0.692 | 0.778 | 0.676 | 0.667 | 0.634 |
| Temporal fixed reservoir + readout | 0.968 | 0.972 | 0.908 | 0.911 | 0.920 |
| Temporal + compound context, no interactions | 0.990 | 0.991 | 0.955 | 0.956 | 0.958 |
| Multimodal physics-informed readout | 0.990 | 0.991 | 0.955 | 0.956 | 0.958 |

The multimodal heads obtain viability RMSE 5.856/R² 0.957 and IC50 RMSE 0.216/R² 0.957 for this seed. The result is intentionally a stress test, not a population estimate. The temporal model improves sharply over physics-only because the latent susceptibility is only visible through phenotype dynamics; compound context then provides a small additional gain over the temporal representation.

The `exposure_only` control remains intentionally easy and demonstrates why a dose/shear-only synthetic label is insufficient. The primary compound-specific scenario avoids presenting that control as the main result: physics-only ROC-AUC is 0.692, temporal ROC-AUC is 0.968, and multimodal ROC-AUC is 0.990 for seed 42. This is still synthetic evidence and must not be confused with biological generalization.

The primary ten-seed audit runs `python -m src.validation --out outputs/validation --seeds 0 1 2 3 4 5 6 7 8 9 --scenario compound_specific`. Multimodal classification gives ROC-AUC mean 0.987 (range 0.968–0.998), average precision mean 0.992, and F1 mean 0.936. Physics-only ROC-AUC averages 0.696, while the temporal reservoir averages 0.985; the multimodal-versus-temporal delta averages +0.0023. With the compact, regularized regression heads, viability gives RMSE 8.53 ± 4.21 and R² 0.902 ± 0.103; IC50 gives RMSE 0.310 ± 0.135 and R² 0.908 ± 0.085. These are regression-test results only; biological claims require experiment/chip-level grouped splits and external validation.

A grouped compound-specific audit (ten seeds, eight sequences per synthetic batch, no batch shared between train and test) gives multimodal ROC-AUC 0.983 ± 0.023, F1 0.944 ± 0.051, viability R² 0.945 ± 0.019, and IC50 R² 0.942 ± 0.027. Physics-only ROC-AUC is 0.683 and the temporal reservoir 0.980. The grouped result is a leakage-resistance diagnostic only: the synthetic batches are not real chips, and the report must not present them as biological validation.

### External front-end portability audit

To test the image-analysis component outside the generator, the repository includes `src.external_validation` for BBBC038v1, a public microscopy dataset with CC0/public-domain images and instance masks. A deterministic sample of 36 cases was split into 12 calibration cases and 24 evaluation cases. Threshold, minimum-area, and morphology parameters were selected on calibration cases only, then frozen for evaluation. On the 24 held-out images, the NeuroChip Twin segmentation front-end obtained mean pixel IoU 0.520, Dice 0.575, precision 0.842, recall 0.578, and absolute object-count error 14.2. This is useful evidence that the front-end can be exercised on real microscopy, but it is not organ-on-chip validation, response prediction, or clinical performance. The source page and download URL are persisted in the generated JSON artifact; no external images are committed to the repository.

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
python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180 --scenario compound_specific
python -m src.neurochip_twin --out outputs/demo_control --seed 42 --samples 180 --scenario exposure_only
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
