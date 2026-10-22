# Supplementary Research Portfolio — AI4S Open Innovation Final

**Primary submission:** NeuroChip Twin v2 — A Digital Twin of Neural-Network Formation  
**Researcher:** Francisco Angulo de Lafuente  
**Prepared:** 22 October 2026  
**Status:** Optional supporting material for an online final presentation and technical Q&A. **Not an amended competition submission or a claim that the projects below have been integrated.**

## Executive summary

NeuroChip Twin v2 is the **sole evaluated AI4S project**. It addresses few-shot, uncertainty-aware prediction of neural-network formation and toxicological responses using real, publicly available rat microelectrode-array (MEA) measurements. The author's related open-source research portfolio demonstrates complementary competencies in numerical simulation, experimentally disciplined benchmarking, GPU computation, visualisation, geometric models, and transparent reporting of failed hypotheses.

The portfolio is **supporting evidence of research and engineering capability**, not additional validation data for NeuroChip Twin v2. In particular, optical fields and photonic neural-network simulations are **not** the biological neural-network-formation measurements used by NeuroChip Twin v2.

## 1. Main project: NeuroChip Twin v2

**Repository:** https://github.com/Agnuxo1/neurochip-twin  
**Technical report:** https://github.com/Agnuxo1/neurochip-twin/releases/download/v2.0.0/neurochip_twin_v2.pdf  
**Interactive demonstration:** https://agnuxo1.github.io/neurochip-twin/  
**Kaggle notebook:** https://www.kaggle.com/code/franciscoangulo/neurochip-twin-v2-demo  
**Demonstration video:** https://youtu.be/yLzpvGqP_2A

The primary system meta-learns across 243 chemicals from the US EPA Network Formation Assay to forecast responses for held-out chemicals from a small number of measured concentrations. It predicts 17 network features across days in vitro (DIV) 5, 7, 9 and 12, and includes uncertainty calibration, domain-aware abstention, decision support and a separate descriptive layer fitted on real four-compartment hippocampal MEA recordings.

The following figures are **reported by the main repository and its linked results files**, rather than independently remeasured for this portfolio:

| Evaluation | Reported result | Interpretation |
| --- | --- | --- |
| Held-out chemical trajectory forecasting, 3 observed doses | MAE 1.180 versus 1.325 for the best reported baseline | Positive within the stated held-out NFA evaluation |
| Nominal 90% conformal intervals | Coverage 0.899 (reported interval 0.891–0.906); 10.7% abstention | Calibration supported **within** the evaluated domain |
| Three-dose hazard classification | Balanced accuracy 86.4%; reference comparators 82.6–85.9% | Modest evidence; only one of three paired comparisons significant |
| Next-dose selection | Tied with log-spaced selection; better than random in the reported experiment | No superiority over the strongest reported practical dose-selection baseline |
| Model-derived benchmark concentration | Worse than interpolation of measured doses | Negative result; no predictive potency superiority claimed |
| Specific versus cytotoxic selectivity | AUROC 0.51 | Negative result |

**Scope:** The measured biological data are from rat cortical and hippocampal cultures. The chip recordings are from a different, undosed experiment; the repository explicitly does not claim cross-platform calibration or human/clinical validity. The source of truth for published numbers is `results/*.json`, with scripts and number-tracing checks in the primary repository.

## 2. Complementary scientific research and software

### A. Neuro3D Lab — published Blender virtual optical laboratory

**Official Blender Extensions listing:** https://extensions.blender.org/add-ons/neuro3d-lab/  
**Source and methods:** https://github.com/Agnuxo1/Neuro3D-Lab

Neuro3D Lab is a virtual bench for **coherent scalar optics and diffractive optical neural-network** experiments. A Blender add-on constructs source planes, phase layers and detectors, exports geometry and height maps, runs numerical propagation, and imports detector intensity for inspection. Its public extension version **0.2.1** is listed as approved in the official Blender Extensions catalogue (checked 22 October 2026).

Research capabilities include a NumPy angular-spectrum core, an optional finite-support model with numerical refinement, independently compared numerical references, recorded settings and file hashes, testable Blender installation, and GPU-oriented development. The project publicly retains failed convergence and benchmark tests. For example, its own README reports that historically sampled phase-only D2NN classifications **did not outperform** the CNN baseline, and that one sample per subwavelength pixel was not a converged description of the fabricated physical element.

**Contribution to the AI4S presentation:** tangible third-party software publication, numerical method verification, instrument-building, reproducibility and a visual demonstration of research workflow.

**Boundary:** This is an **optical** laboratory in simulation. It contains no measured biological chip data, is not a physical optical device, and is not currently an organ-on-a-chip simulator.

### B. NeuroPixel — inspectable spatial neural computation

**Repository:** https://github.com/Agnuxo1/NeuroPixel

NeuroPixel represents learned recurrent neural states on a spatial canvas and includes GPU graphics-based state updates, image-based processing, a shared dictionary decoder and visible intermediate activity. According to its README, the project reports **136 GPU engineering checks** on an RTX 3090 and retention of **320 test decisions** across five learned models. Its bounded external UCI optical-digits comparison reports **94.36%** versus **90.84%** for a paired CNN comparator; its own robustness tests also show a pronounced performance collapse under a one-column image shift.

**Contribution:** inspectable intermediate states, GPU computation, comparative benchmarking and explicit negative controls. The architecture offers research ideas for interpretability and computational efficiency, **not evidence that NeuroChip's toxicological forecasts improve by adopting it**.

**Boundary:** These are engineering and image-recognition results, not demonstrations of biological neural behaviour, cross-assay transfer or general architectural superiority.

### C. Neuro3D / OpticNeuroBlender — trainable optical geometry

**Repository:** https://github.com/Agnuxo1/Neuro3D

Neuro3D studies a trainable scalar optical network whose response derives from evaluated Blender geometry. Its README distinguishes the graphics-engine part (triangle capture, geometrical distance, phase and derivatives) from topology, coherent fusion and optimisation still handled on CPU. It documents comparison with independent numerical solvers, geometric consistency checks, actual training runs and failed acceleration hypotheses.

Reported bounded examples include a captured scene with **104 objects, 6,656 triangles, 133 graph states and 17,060 paths**, and a small Iris task reporting **27/30** test decisions correct in a specified split. These examples are not proof of generalisation to other tasks or of a full GPU-native neural pipeline.

**Contribution:** geometric modelling, coherent field computation, numerical cross-checking and transparent performance evaluation.

**Boundary:** An optical computational model; it has no direct validation against the MEA chemical-exposure data used for NeuroChip Twin v2.

### D. Silice-Neuro3D-Cube — prospective three-dimensional photonic hardware

**Repository:** https://github.com/Agnuxo1/Silice-Neuro3D-Cube

This numerical research programme investigates waveguides and optical transformations in fused silica, with a proposed FEMTO-1 prototype based on femtosecond-laser-written structures. The repository reports a **22.33×** timing improvement for **one specified resident CUDA128 scalar-propagation case**, and **168/168** audited coherent-field representation checks across the tested OpenGL/Vulkan cases.

The repository is unusually explicit about counter-evidence: some proposed energy-efficiency advantages fail under the documented assumptions; the preregistered numerical work narrows the claimed design benefits.

**Contribution:** a prospective hardware pathway, numerical validation discipline, physical assumptions and fabrication-readiness planning.

**Boundary:** **No photonic processor has been fabricated or measured.** Simulated benefits must not be presented as realised hardware performance, nor as a manufactured biological organ-on-a-chip.

## 3. What the four projects demonstrate together

1. **Practical scientific software development:** a Blender extension distributed through Blender's official catalogue, plus documented simulations and demonstrations.
2. **Reproducibility and falsifiability:** protocols, tests, code, tracked result files, numerical comparisons and retention of failed or inconclusive experiments.
3. **Engineering breadth:** GPU numerical methods, visual and geometrical interfaces, neural architectures and prospective optical hardware, without pretending that these fields are experimentally interchangeable.
4. **Research maturity:** clearly identified domains of validity, negative findings and milestones necessary before physical experiments.

These are **capabilities of the researcher and supporting portfolio**, not extra entries in the AI4S scoring dataset.

## 4. Suggested usage at the final

Include **one optional supporting slide** headed *Beyond the Submission: A Reproducible Research Toolchain*, with a screenshot of the officially published Neuro3D Lab extension and a compact link/QR code to this document.

The primary presentation should spend most of its time on the rat MEA data, unseen-chemical cross-validation, effect sizes, uncertainty calibration, decision-making evidence, practical laboratory value and limitations of **NeuroChip Twin v2**. Present the related projects only when asked about technical execution, software dissemination, reproducibility or future research directions.

### Suggested oral answer (approximately 30 seconds)

> “NeuroChip Twin v2 is our life-science entry and its claims stand entirely on the reported rat MEA datasets and frozen evaluations. Separately, I have developed an open scientific software portfolio covering numerical simulation, graphics-based neural computation and photonic research. One tangible outcome is Neuro3D Lab, a virtual optical bench publicly approved in Blender's official extensions catalogue. These projects document my ability to build and test scientific tools. They are complementary research, not additional biological validation or already integrated parts of NeuroChip Twin.”

## 5. Review and submission conditions

- **Use this document as optional context only.** Submit it through Kaggle or any final-round channel **only if the organisers permit updated or supplementary material**.
- Do **not** replace the frozen evaluated submission, edit its reported benchmark results, or claim that the optical tools have been experimentally connected to the biological chip.
- Avoid private or unreleased data, credentials, and third-party assets without appropriate permissions.
- Keep original citations and software licences attached to each repository. The Blender add-on is GPL-3.0-or-later; repository cores may have different licences.
- If the committee requests a PDF or slides, render a separate submission-ready version and confirm the exact deadline and delivery format.

**All quantitative statements above are taken from the linked public project READMEs or extension listing as of 22 October 2026; they are reported research results with the stated limitations, not fresh independent replication.**
