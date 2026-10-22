# NeuroChip Twin v2 — Targeted Final-Round Clarifications
## Technical Innovation and Practical Value | AI4S Open Innovation

**Applicant:** Francisco Angulo de Lafuente  
**Submission:** NeuroChip Twin v2 — *End-to-End System*  
**Status:** Optional supplementary explanation for the final-round talk or Q&A; **not an amended submission, new experiment or additional benchmark**. The original technical report, code and submitted results remain authoritative.

### Executive statement

NeuroChip Twin v2 addresses a focused experimental bottleneck: neural-network responses to chemical exposure unfold over several days, while laboratory dose–response measurements are limited and costly. The submitted system learns from real rat microelectrode-array (MEA) assays to forecast how a previously unseen chemical will affect network formation from only a few measured concentrations, reports the reliability of those forecasts and evaluates decisions a laboratory could make with them.

The final-round clarification strengthens **two specific aspects of the existing evidence**: **(A) why the modelling design is technically distinctive**, and **(B) how a laboratory would use it, with benefits and risks stated at their actual evidence level**. No change to the numerical results or dataset is proposed.

## A. Technical innovation — a testable combination, not a slogan

**Core innovation:** a **chemical-as-few-shot-task** formulation, where a model trained across chemicals receives a small observed dose-response context for a completely held-out compound, and forecasts its joint **dose × developmental day × neural network feature** response.

The submission's distinctive and assessable design elements are:

1. **Joint developmental prediction.** NeuroTrajectory predicts **17 measured network features** on days in vitro (DIV) **5, 7, 9 and 12**, not a single static toxicity endpoint.
2. **Interpolation-informed residual decoder.** Instead of discarding well-understood dose interpolation, the neural model learns per feature and day how strongly to trust measured neighbouring concentrations, and how to correct that interpolation using patterns learned across other chemicals. This is a principled hybrid of an interpretable laboratory baseline and learned generalisation.
3. **Dose-local attention.** Queries attend directly to the actually measured concentration levels, so the influence of nearby evidence can vary with the requested dose.
4. **Bounded trust, not universal certainty.** Calibration, chemical-level cross-validation and abstention supply experimentally assessed safeguards within the studied data domain.

**What directly supports the novelty claim:**

| Test in original submission | Reported evidence | What it establishes |
|---|---|---|
| R2 — held-out chemical prediction at 3 measured concentration levels | Curve MAE **1.180** versus **1.325** for the best reported comparator; paired difference **−0.145 [−0.188, −0.099]** | Out-of-sample predictive improvement against the evaluated methods |
| R9 — remove interpolation-informed decoder at 3 levels | Error rises by **6.9%** in the specified single-seed ablation; reported paired MAE difference **+0.082 [0.062, 0.104]** | The interpolation-aware design contributes measurable value |
| R9 — remove local dose attention at 3 levels | Paired MAE difference **+0.021 [0.008, 0.035]** | Attention adds a smaller, measurable contribution in that setup |
| R7 — nominal 90% prediction intervals | Empirical coverage **89.9%**, with **10.7%** abstention | A quantitative reliability check on held-out NFA data |

**Correct interpretation:** The novelty lies in the *experimentally tested workflow and combination of mechanisms*, not a claim to have invented attention, interpolation, neural processes or conformal prediction individually. Ablation conditions and full baselines are detailed in the original report; distinct evaluations should not be casually conflated.

## B. Practical value — show the decision and the evidence boundary

**Primary user:** an in-vitro neural-screening laboratory or a neural organ-on-a-chip research group able to map its electrophysiology recordings to the required features and assay descriptors.

**A concrete workflow consistent with the submitted implementation:**

1. **Start with real measurements.** Register compound, available concentrations, culture day, assay metadata, provenance and MEA network features.
2. **Forecast unmeasured conditions.** After a small set of observed concentrations, inspect predicted dose–development trajectories for the remaining concentrations, together with calibrated intervals and an abstention flag.
3. **Assess a decision.** Use the evaluated developmental-neurotoxicity hazard classifier, compare against established EPA rules, and choose another observation when uncertainty is high.
4. **Plan the next assay.** Use a sensible fixed log-spaced concentration scheme as the practical default unless a prospectively validated planner can outperform it. DoseCompass is an inspectable candidate, **not** a demonstrated improvement over that strong baseline.
5. **Decide when to measure again.** The separate retrospective early-exit study compares throughput- and safety-oriented gates on DIV 5/7 observations. Any prospective stop rule would require a pre-agreed tolerated false-negative rate.

**How to quantify potential laboratory value without claiming measured savings:**

| Relevant question | Existing evidence | Honest limitation |
|---|---|---|
| Can three measured levels replace a full seven-level profile for modelling? | In the tested retrospective setting, a 3-of-7 design entails **57.1% fewer treated wells for that chemical** at equal replication; R2 tests prediction of the held-out responses | A *design scenario*, not a prospectively observed reduction in cost, time or error at a partner laboratory |
| Can the partial profile inform hazard screening? | R4: **86.4%** balanced accuracy on the specified **97-chemical** comparison versus **82.6–85.9%** for EPA-rule variants | Only **one of three** paired comparisons is statistically significant; **19** reference negatives make specificity estimates uncertain |
| Does adaptive dose selection already outperform a simple lab design? | R6: DoseCompass ties fixed log-spaced selection, while outperforming the mean random design in the primary retrospective test | **No claim of superiority** over fixed log spacing |
| Is early exit potentially useful? | R11: a simple persistence-based gate offers a retrospective throughput operating point; the twin's uncertainty-based gate trades throughput for fewer missed known neurotoxicants under the reported post-hoc reference-label check | Retrospective, reference-dependent; not an approved laboratory stopping policy or a demonstrated animal-test reduction |

**The next real-world validation is precisely defined:** map a partner laboratory's new chip recordings, freeze the prediction model and trial protocol, blind the withheld measurements, compare against log-spaced or current lab procedures and quantify *measured* prediction error, calibration, false negatives, wells used, assay days and decision utility. No external-lab success is claimed before this study.

## C. What a final-round judge should retain

> **“NeuroChip Twin v2 adds value at two levels: it tests a hybrid, interpolation-aware neural forecasting method on unseen chemicals, and it translates those forecasts into bounded, auditable decision support for real in-vitro workflows. Its accuracy and uncertainty improvements are documented; prospective laboratory savings remain the next validation milestone.”**

**Suggested 45-second oral emphasis:**

> “Our innovation is not a biological-looking simulation. A chemical is treated as a few-shot prediction task: we learn across real neural culture experiments, then forecast the development of a new compound from a few measured concentrations. The interpolation-informed decoder is not cosmetic: when removed, our specified ablation reports a 6.9 percent increase in error. With three measured levels, the reported forecasting error is 1.180, versus 1.325 for the strongest tested baseline. For laboratories, the practical opportunity is to target future measurements while seeing uncertainty and a clear abstention signal. We evaluated hazard calls and dose selection against real experimental baselines. We also state where the latter does not yet beat a simple fixed strategy. The next step is prospective validation of actual laboratory savings.”

## D. Relation to the broader research programme — one sentence only

The applicant's parallel work on [NeuroPixel](https://github.com/Agnuxo1/NeuroPixel), [Neuro3D](https://github.com/Agnuxo1/Neuro3D), [Neuro3D Lab](https://extensions.blender.org/add-ons/neuro3d-lab/) and [Silice-Neuro3D-Cube](https://github.com/Agnuxo1/Silice-Neuro3D-Cube) shares an interest in inspectable neural-inspired computation and numerical validation; **those projects are separate and supply no additional biological validation, manufactured photonic hardware or measured performance benefit to NeuroChip Twin v2**. Include this only if directly asked about longer-term research.

## Source of truth and submission gate

The technical proof is in [the submitted report](https://github.com/Agnuxo1/neurochip-twin/releases/download/v2.0.0/neurochip_twin_v2.pdf), particularly Method, **R2**, **R4**, **R6**, **R7**, **R9**, **R11**, Practical Value and Limitations, supported by the [public code and results](https://github.com/Agnuxo1/neurochip-twin), [interactive demo](https://agnuxo1.github.io/neurochip-twin/), [video](https://youtu.be/yLzpvGqP_2A) and [Kaggle notebook](https://www.kaggle.com/code/franciscoangulo/neurochip-twin-v2-demo). Reported figures have not been independently remeasured for this addendum.

**Do not send or upload this note until the organisers confirm that final-round supporting documentation is permitted.** The judged submission and original report remain unchanged.
