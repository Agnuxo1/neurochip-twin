# AI4S Open Innovation — Final-Round Scientific Positioning
## NeuroChip Twin v2 | From measured neural-network dynamics to uncertainty-aware decisions in vitro

**Submission category:** End-to-End System  
**Applicant and sole human team member:** Francisco Angulo de Lafuente  
**Core submission:** [NeuroChip Twin v2](https://github.com/Agnuxo1/neurochip-twin)  
**Prepared:** 22 October 2026  
**Document status:** Optional supporting context for the final-round presentation and Q&A. Not a change to the evaluated submission. Only submit it if the organisers explicitly permit supplementary material.

### The proposition

**Neural-network formation is a time-dependent biological process. NeuroChip Twin v2 turns real electrophysiology measurements into calibrated forecasts and decision support for in-vitro neural studies.**

The scientific objective is specific: make better use of scarce, expensive and time-consuming measurements in neural microphysiological systems by predicting dose-dependent network-development trajectories for **unseen chemicals**, expressing uncertainty, recognising when to abstain, and testing whether those forecasts can support laboratory decisions.

This is the AI4S contribution. It is not a claim to have reproduced a human brain, grown synthetic biological neurons, or built a general-purpose brain simulator.

### Why it fits AI + organ-on-a-chip

A neural culture's response to chemical exposure is not adequately represented by a single endpoint. It evolves across days and involves multiple interacting network features. Laboratories need models that can cope with missing concentrations, generalise to new chemicals and disclose uncertainty before their outputs influence what is measured next.

NeuroChip Twin v2 follows a traceable workflow:

**Real rat MEA recordings → provenance-tracked features and chemical-level data splits → few-shot trajectory forecasting → uncertainty calibration and abstention → decision-support evaluation.**

A **separate**, descriptive layer analyses directed connectivity from recordings of a real four-compartment hippocampal MEA chip. It provides a bridge to compartmental neural systems without claiming that its chip dataset and the chemical-exposure dataset are one experimentally calibrated platform.

### The evidence that matters to the jury

The following values are reported in the submitted project's [README](https://github.com/Agnuxo1/neurochip-twin/blob/main/README.md), [technical report](https://github.com/Agnuxo1/neurochip-twin/releases/download/v2.0.0/neurochip_twin_v2.pdf) and checked `results/*.json` records. They are **not new experimental findings** of this supporting document.

| Judge's question | Submitted evidence | Interpretation |
|---|---|---|
| Does it predict network formation for chemicals excluded from training? | At three observed concentrations, curve MAE **1.180** vs **1.325** for the strongest reported comparator | Positive on the held-out EPA NFA chemical evaluation |
| Does it report usable uncertainty? | Nominal 90% intervals: empirical coverage **89.9%**; **10.7%** abstention | Calibrated and selective **within the tested domain** |
| Can predictions help a practical laboratory decision? | Three-dose hazard classification: balanced accuracy **86.4%** vs **82.6–85.9%** EPA-rule comparators | Promising but modest; only one of three paired tests significant |
| Is next-dose selection demonstrably better? | DoseCompass **ties** a log-spaced strategy (and outperforms random in the reported comparison) | No claim of superiority over practical log-spaced selection |
| Are failure modes investigated? | Model-derived potency underperforms interpolation; cytotoxic selectivity AUROC **0.51** | Unsuccessful components are disclosed; limitations bound the claims |

**Reproducibility:** public [code](https://github.com/Agnuxo1/neurochip-twin), [technical report](https://github.com/Agnuxo1/neurochip-twin/releases/download/v2.0.0/neurochip_twin_v2.pdf), [demonstration video](https://youtu.be/yLzpvGqP_2A), [online demo](https://agnuxo1.github.io/neurochip-twin/) and [Kaggle notebook](https://www.kaggle.com/code/franciscoangulo/neurochip-twin-v2-demo). The repository exposes a CPU demonstration, traceable result files, validation scripts, model-bundle hashes and explicit dataset provenance.

**Scientific boundaries:** the biological measurements come from rat cultures; they do not establish human clinical validity. The four-compartment chip dataset is undosed and separate from the chemically exposed EPA cohort. Predictions are not validated across all chip architectures, species or cellular systems. Claims of experimental savings, physical device performance or clinical utility would require additional prospective studies.

### The research line behind the submission

A consistent long-term question connects the applicant's wider work:

> **How can the spatial organisation, changing activity and signal propagation of biological neural networks inspire computational systems whose states and predictions can be inspected, tested and eventually implemented in alternative substrates?**

This is a research direction, **not** a claim that the distinct projects below share a trained model, dataset, equivalent biological dynamics, or validated end-to-end pipeline.

The stages must be distinguished:

1. **Biological evidence — delivered and evaluated here.** NeuroChip Twin v2 models neural-network measurements from living rat cultures. Its relevant numerical claims are independently inspectable in the competition submission.
2. **Spatial computation — exploratory, separate research.** [NeuroPixel](https://github.com/Agnuxo1/NeuroPixel) explores neural state updates organised over a two-dimensional grid, with local interactions and inspectable intermediate states. Pixels are **computational cells in a model**, not living neurons or verified equivalents of biological cells. This project contributes experience with spatial representations, testing and visual interpretation; it is **not a component of NeuroChip Twin v2**.
3. **Alternative optical implementation — long-term methodology, not biological evidence.** [Neuro3D Lab](https://extensions.blender.org/add-ons/neuro3d-lab/) is a publicly distributed **virtual optical bench** for scalar coherent propagation and diffractive networks; the [source](https://github.com/Agnuxo1/Neuro3D-Lab) includes numerical validation and explicitly reported limitations. [Neuro3D](https://github.com/Agnuxo1/Neuro3D) investigates optical networks derived from Blender geometry. [Silice-Neuro3D-Cube](https://github.com/Agnuxo1/Silice-Neuro3D-Cube) examines a prospective path toward waveguide circuits in fused silica. These are **optical and photonic research**, not organ-on-a-chip experiments or demonstrated copies of biological neural networks; no fabricated, experimentally validated silica neural processor is claimed.

**Strategic relevance:** These additional projects demonstrate a consistent practice of building transparent numerical models, measuring error and exposing limitations. They do **not** add biological validation or raise the evidence level of the competition submission. Their names and links belong in **one optional backup slide or in answers to questions about technical capabilities**, not in the main results slides.

### Recommended one-sentence description of the wider programme

> “Our research programme studies neural-network dynamics across biological measurement and computational modelling; NeuroChip Twin v2 is its experimentally grounded life-science application, while our spatial-computation and optical-simulation work explores separate longer-term methods for interpretable, testable neural-inspired systems.”

### Presentation discipline

**Show first:** the laboratory problem; real neural MEA data; frozen data splits; the unseen-chemical example; comparison against baselines; calibrated uncertainty; reproducibility; the precise laboratory decision supported.

**Show only when directly relevant:** the publicly listed Neuro3D Lab Blender extension and one sentence about NeuroPixel's inspectable spatial computation. Other optical/silica projects should remain in this background note unless the jury specifically asks about the long-term research programme.

**Do not imply:** that each pixel is a living cell; that current optical simulations reconstruct a biological neural network; that a glass-based processor exists; that NeuroPixel trained or improved NeuroChip Twin v2; that a non-significant or negative result demonstrates efficacy; or that rat data validate human clinical decisions.

### Committee-facing 35-second introduction

> “NeuroChip Twin v2 tackles a concrete life-science problem: how to infer the development of a neural network under chemical exposure when only a few concentrations have been measured. We use real rat microelectrode-array recordings, evaluate on chemicals held out from training, and attach calibrated uncertainty so predictions can be interpreted responsibly. At three observed doses, our reported trajectory error is 1.180 versus 1.325 for the strongest evaluated baseline. Some decision tasks remain inconclusive, and we document those limits. The accompanying open code, report and demonstration make the findings inspectable. This is a focused step towards trustworthy digital-twin assistance in neural organ-on-a-chip research.”

### Technical evidence and primary links

- [Main NeuroChip Twin v2 submission source](https://github.com/Agnuxo1/neurochip-twin)
- [Submitted technical report](https://github.com/Agnuxo1/neurochip-twin/releases/download/v2.0.0/neurochip_twin_v2.pdf)
- [Interactive demonstrator](https://agnuxo1.github.io/neurochip-twin/)
- [Submitted video demonstration](https://youtu.be/yLzpvGqP_2A)
- [Kaggle executable notebook](https://www.kaggle.com/code/franciscoangulo/neurochip-twin-v2-demo)
- [Official Blender Extensions listing — Neuro3D Lab](https://extensions.blender.org/add-ons/neuro3d-lab/)
- [NeuroPixel research source](https://github.com/Agnuxo1/NeuroPixel)
- [Neuro3D optical research source](https://github.com/Agnuxo1/Neuro3D)
- [Silice-Neuro3D-Cube research source](https://github.com/Agnuxo1/Silice-Neuro3D-Cube)

### Release gate

The competition's official submission is the **Kaggle Writeup**. The jury's final-round presentation format and any permission to provide updated or supplementary materials must come from the organising committee. Do not upload this document, send it by email, merge it into the frozen public submission, or represent it as accepted additional evidence without that confirmation. This note is prepared for controlled use during final presentation and Q&A.
