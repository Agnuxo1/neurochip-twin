# AI4S Final — Jury-Focused Presentation and Q&A
**Working master in English — not submitted**  
**Candidate:** NeuroChip Twin v2 · **Category:** End-to-End System  
**Competition:** [AI4S Open Innovation: AI for Life Science](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview/challenge-organization)  
**Author:** Francisco Angulo de Lafuente  
**Version:** 22 October 2026

> **Release condition:** The organising committee has not yet provided the confirmed presentation duration, meeting platform or permission for supplementary submissions. This is a modular **working script**, not an assertion that these slides have been approved or delivered. All results below are drawn from the existing public submission; do not substitute new runs or modified numbers without a separate documented verification process.

## Positioning for the judges

**One problem. One delivered solution. One reproducible evidence base.**

The submission concerns **uncertainty-aware prediction of neural network development under chemical exposure, using real in-vitro MEA recordings**. The research philosophy behind this work is to study the evolution and connectivity of neural networks through experimentally grounded models. Separate research into spatial neural computation and photonic devices does not validate the biological predictions and should be presented **only if requested**.

The 22 October final-round notification awarded the following **reviewer means**: technical innovation **26.5/30 (88.3%)**, project quality/performance **22.75/25 (91%)**, practical value **17.5/20 (87.5%)**, completeness **14.25/15 (95%)**, and explainability/reliability **9/10 (90%)**. **Improve only the two lowest relative scores:** practical value and technical innovation. Keep the original report, results, data, software and validated claims unchanged. Do not replace or confuse these reviewer criteria with an earlier overview-page scoring rubric.

## Slide 1 — A neural culture evolves. Our measurements are sparse.

**On-screen headline:** *Can a digital twin forecast neural-network formation from a few measured doses?*

**Visual:** The existing graphical abstract from [the main project](https://github.com/Agnuxo1/neurochip-twin/blob/main/report/figures/visuals/graphical_abstract.png), cropped for clarity. Emphasise the chemical, measured concentrations, developmental days and network features.

**On-screen text:** *Real in-vitro electrophysiology. Few-shot forecasting. Explicit uncertainty.*

**Speaker (30–40 s):** “In an in-vitro neural network, drug effects unfold over days, not just at a single endpoint. Measuring every concentration and every time point is costly. NeuroChip Twin v2 explores whether we can infer the trajectory for a previously unseen chemical when only a small number of concentrations have been measured — and whether the uncertainty is sufficiently explicit to support responsible experimental planning.”

**Claim guardrail:** The system does **not** prove experimental cost reduction or provide human clinical predictions.

## Slide 2 — Where the technical innovation is

**On-screen headline:** *An interpretable baseline, a learned correction, and a testable reliability layer.*

**Visual architecture:**  
`Real EPA NFA MEA → chemical-level held-out splits → measured-dose interpolation + learned residual correction → joint dose × DIV × feature forecast → conformal calibration / abstention`

A small clearly separated inset:  
`Real 4-compartment hippocampal MEA → descriptive connectivity model (different undosed dataset)`

**On-screen text:** *The decoder earns its complexity: +6.9% error when removed (R9, three measured doses). Dose-local attention also contributes: +0.021 paired MAE on removal in the tested setting.*

**Speaker (45–55 s):** “The key novelty is not a claim that we invented neural processes, attention or interpolation. It is their tested combination for a real biological few-shot problem. For each previously unseen chemical, the model uses a few measured concentrations to forecast 17 network features across four culture days. The decoder starts from an interpretable measured-dose interpolation and learns how to correct it across chemicals. Our ablation shows that removing this decoder increases curve error by 6.9 percent at three measured doses, with a smaller independent effect for local attention. That is direct evidence of what each design element contributes.”

**Claim guardrails:** R9 is a specified single-seed ablation, not the identical evaluation object as ensemble R2. Do not imply the model generates physical living neurons, predicts humans, or already integrates the separate chip dataset.

## Slide 3 — Quantitative evidence on held-out chemicals

**On-screen headline:** *Better trajectory accuracy, with calibrated uncertainty.*

**Two big numerical findings** (published, directly attributable):

- **1.180 vs 1.325** — trajectory MAE at three measured concentrations, model versus best reported comparator.
- **89.9%** — observed coverage of nominal 90% conformal intervals, with **10.7% abstention**.

**Small source footer:** `results/trajectory_cv.json`, `results/conformal_r7.json`; technical report R2 and R7.

**Speaker (45–55 s):** “The central result is a controlled forecast for chemicals excluded from training. With three observed concentrations the trajectory error is 1.180, compared with 1.325 for the strongest comparator in our evaluation. The system also quantifies predictive uncertainty: measured interval coverage is 89.9% for a nominal 90% target, with selective abstention. These figures apply to the evaluated cohort and protocol, not to all species or chip technologies.”

**Claim guardrail:** No blanket claim of superiority over all methods or out-of-domain guarantees.

## Slide 4 — Practical value: exactly which laboratory decision?

**On-screen headline:** *Make scarce measurements more informative — and measure the trade-offs.*

**Visual:** A simple bench workflow or an actual demo screen showing **three measured of seven available concentration levels**, the predicted remaining profile and its uncertainty; one adjacent result block for the hazard comparison.

**On-screen text (three evidence-backed lines):**
- **3/7 observed doses → 57.1% fewer treated wells in that design scenario**, with forecasts evaluated against withheld observations (**not** proven prospective cost savings).
- **86.4%** three-dose hazard balanced accuracy vs **82.6–85.9%** EPA-rule comparators on the documented matched cohort.
- **DoseCompass ties a strong fixed log-spaced plan.** It does not yet establish adaptive-planning superiority.

**Speaker (45–55 s):** “The practical workflow is concrete: measure a small number of concentrations, forecast the remaining network-development response with uncertainty, and decide whether another measurement is needed. Three of seven dose levels represent 57.1 percent fewer treated wells within that retrospective design, provided replicate counts are equal — that is an experimental design opportunity, not a cost reduction already proved in a new laboratory. Our three-dose hazard analysis is competitive against reproduced EPA rules, although only one of three paired comparisons is significant. Our dose recommender ties log spacing. These controls show both how the system could help a laboratory and exactly what remains to be demonstrated.”

**Claim guardrails:** Avoid conflating reduced treated wells in a modelled design with total laboratory savings, and do not claim dose scheduling is better than the best fixed comparator. If asked about early exit, distinguish the simple persistence throughput gate from the uncertainty-aware safety gate, noting both are retrospective and not validated stopping policies.

## Slide 5 — Why the results can be trusted and reproduced

**On-screen headline:** *Evidence-linked outputs. Explicit limits. Reproducible code.*

**Visual:** Screenshot of the main [online demo](https://agnuxo1.github.io/neurochip-twin/) alongside code/report/demo links.

**Proof points:** Public code · frozen chemical splits · numerical results under version control · CPU demonstrator · model-bundle SHA-256 checks · disclosed negative outcomes.

**Speaker (35–45 s):** “A result must be independently inspectable. We supply a public repository, a technical report, a working demonstration, runnable reproduction instructions, result ledgers and integrity checks. The review can distinguish data-backed improvements from unsupported claims. The same transparency applies to negative findings: our model-derived potency does not beat interpolation of observed doses, and a reported selectivity score is close to chance.”

**Claim guardrail:** Do not say tests were independently reproduced by an external biology laboratory. A coding-agent review is not an external peer review.

## Slide 6 — Proving laboratory value next

**On-screen headline:** *From retrospective prediction to prospectively measured decision utility.*

**Visual:** Four scientifically checkable milestones:  
`New partner-chip recordings → frozen/blinded forecast → comparison with real laboratory baseline → measured impact & safety audit`

**Speaker (35–45 s):** “The remaining translational step is not speculative hardware. It is an independently defined biological replication: map a partner laboratory's neural chip recordings, freeze our method and the comparator protocol, predict outcomes before withheld observations are revealed, and measure calibration and errors alongside the actual wells, assay days and decisions used. The test should be considered successful only if it beats a meaningful laboratory baseline without unacceptable false negatives. This is how a promising computational finding becomes documented experimental value.”

**Claim guardrail:** Prospective design only; no pre-claimed clinical validation, hardware fabrication, or achieved savings.

### Optional backup slide — The research programme behind the work

**Use only for questions about broader technical expertise. Do not insert into the main scientific narrative by default.**

**Headline:** *One methodological direction: measurable neural dynamics, inspectable computation, verifiable models.*

- **Current AI4S evidence:** NeuroChip Twin v2 — real rat MEA data, generalisation tests and calibrated forecasting.
- **Parallel computational research:** [NeuroPixel](https://github.com/Agnuxo1/NeuroPixel) — spatial recurrent computation with visible internal states.
- **Separately published optical research software:** [Neuro3D Lab on Blender Extensions](https://extensions.blender.org/add-ons/neuro3d-lab/) — publicly distributed virtual optical bench, **not** a biological cell-culture simulator.
- **Longer-horizon photonic research:** [Neuro3D](https://github.com/Agnuxo1/Neuro3D) and [Silice-Neuro3D-Cube](https://github.com/Agnuxo1/Silice-Neuro3D-Cube) — optical simulation and prospective geometry-in-glass concepts; no measured fabricated optical neural chip is claimed.

**Suggested 20 s statement:** “I also investigate representations of neural-inspired computation in spatial and optical media. Those projects demonstrate experience in building numerical tools and checking physical assumptions. They are separate from this submission: all biological evidence and all AI4S performance claims come from NeuroChip Twin v2.”

## Q&A — Prepared answers for rigorous judges

**Q1. Is this a digital twin of a specific human organ-on-a-chip?**  
No. It is a data-grounded computational prototype for rat neural MEA assays, with a separate descriptive model of a rat four-compartment hippocampal chip. There is no claim of a human clinical model or validation on a particular commercial platform.

**Q2. Are the two biological datasets cross-calibrated?**  
No. The chemically exposed cortical NFA data support the dose-development forecasting claims. The separate undosed hippocampal chip recordings support bounded connectivity analysis. Their respective roles and differences are documented.

**Q3. What is the innovation beyond curve fitting?**  
A few-shot unseen-chemical forecasting task spanning multiple developmental time points and neural features, an interpolation-informed decoder, domain-aware conformal calibration/abstention and explicit benchmarking of consequential decisions. The claim is bounded by comparisons and ablation evidence in the technical report.

**Q4. How can 89.9% coverage be trusted?**  
The 89.9% is empirical coverage measured on the stated held-out evaluation under the repository's calibration protocol; it is not a guarantee for arbitrary new assays. Calibration design, strata, split rules and results files should be shown if requested.

**Q5. Do you beat EPA hazard rules?**  
Not consistently enough to claim a robust superiority. The balanced-accuracy point estimate is competitive, but only one of three paired significance tests meets the reported threshold; the negative/reference-class count is limited.

**Q6. Does DoseCompass demonstrably reduce the number of measurements?**  
Not yet. Against the practical log-spaced method it ties in the reported comparison. Demonstrating reduced experimental cost or better decision quality requires prospective laboratory assessment.

**Q7. Can your pixels function like biological cells?**  
In NeuroPixel, the pixels are elements of a recurrent computational state: a loose **analogy to local interacting units**, not living cells or physiological neurons. NeuroPixel is not the biological model used in NeuroChip Twin.

**Q8. Have you reproduced living neural networks inside crystal or silica?**  
No. The parallel photonic work studies numerical wave propagation, interference and proposed waveguide structures. Computing with light and reproducing biological neuronal electrophysiology are different questions. The silica research does not claim a fabricated or biologically equivalent neural processor.

**Q9. Where can a judge reproduce the result without specialised hardware?**  
Start with `python demo.py` and the repository's documented requirements. Consult the frozen model bundle, result ledger and report for full methodological replication. The public web demo and Kaggle notebook provide an accessible entry point.

**Q10. Is generative AI part of the validated prediction system?**  
No. AI coding assistants contributed to development and verification workflows and are disclosed as tools; the delivered scientific pipeline and reported prediction models are described separately in the technical report.

**Q11. What would most improve the evidence?**  
A prospectively defined, independent biological dataset with new compounds and a new chip or assay, plus an assessment of whether predictions and abstention policies improve decisions relative to laboratory baselines.

**Q12. Why should this be awarded under End-to-End System?**  
Because the submission integrates real experimental inputs, a **novelty claim tested by a component ablation**, held-out chemical forecasting, reliability calibration and **retrospectively tested decisions** into a traceable life-science workflow. The original R2/R9 experiments are the innovation evidence; R4/R6 and the practical-value section quantify what a laboratory could use and the limits of current benefit. Prospective cost savings remain unproved.

## Production checklist — before any actual presentation

- [ ] Receive final-round date/time, platform, permitted presentation duration and format.
- [ ] Confirm whether updated slides or supplementary files are accepted; respect cut-off and submission channel.
- [ ] Verify publicly accessible report, demo video, code, notebook and interactive demo immediately before presentation.
- [ ] Extract figures from the *frozen* submitted report/results. Add concise labels and sources; no decorative synthetic evidence.
- [ ] Rehearse with the committee's time allocation and preserve 2–3 supporting results in backup slides.
- [ ] Check language, text legibility, data-rights citations, backup video and screen-sharing permissions.
- [ ] Present the optical/neuromorphic research **only if asked** or if the committee explicitly values broader research capacity.
- [ ] Do not send a second unsolicited email while awaiting the organisers' already-requested final instructions.

**Source of all AI4S quantitative results:** the published NeuroChip Twin v2 README, `results/*.json` and technical report. The wider project links are supplied only for background context. 
