# NeuroChip Twin v2 - interactive app

A Gradio app over the shipped `data_bundle/`. It runs on CPU, offline, with no downloads. For any of the
EPA Network Formation Assay chemicals you choose how many concentrations are "measured" (k = 1-4). The
NeuroTrajectory ensemble then forecasts the whole dose x developmental-time response (17 MEA network
features at DIV 5/7/9/12) with a 90 % band. Press **Reveal** to see the real held-out wells and the error
against the paper's few-shot baselines (log-linear interpolation and analog kNN).

A chemical is always forecast by the models of **its own cross-validation fold**, which never saw it.

| Tab | Content |
|---|---|
| Forecast & reveal | chemical search, k slider, design (`spread` or custom levels), forecast panels, 17 x 4 trajectory heatmap, abstention badge, then the reveal with a metrics table and the matching cross-validation reference |
| Results | key numbers and 95 % CIs, read at runtime from `results/trajectory_cv.json`, `conformal_r7.json` and `epa_baseline_repro.json` |
| Chip layer | directed propagation on the real Brewer 4-compartment MEA (`results/r8_chiplayer.json`, `chiplayer_params.json`, `results/figures/chip_*.png`) |
| About & limits | evidence boundaries, data licences from `data_bundle/manifest.json`, how to reproduce |

No number in the app is typed by hand. The app reads every number from the results JSON files and the manifest.

The page always uses a light theme, because the figures are drawn on white. Colours follow the
colour-blind-safe Okabe-Ito palette: blue forecast and band, near-black measured wells, vermillion revealed
wells. Below 720 px of browser width (phones) the figures switch to a narrow layout, with two DIV panels per
row and the heatmaps stacked. Wide tables scroll inside their own box.

## Run locally

From the repository root:

```bash
pip install -r requirements-app.txt      # numpy, pandas, torch (CPU is enough), matplotlib, gradio
python app/app.py                        # http://127.0.0.1:7860   (--port / --host to change)
```

Other entry points:

```bash
python app/app.py --smoke --chemical Deltamethrin --k 3   # compute core only, prints JSON, no server
python -m pytest tests/test_app.py -q                     # tests of the compute core (skipped without gradio)
```

On Windows, if pytest cannot clean its temporary folder, add `--basetemp=<a folder on a data drive>`.

The compute core is made of plain functions: `run_forecast`, `reveal`, `forecast_figure`,
`trajectory_figure`, `r2_table`, `conformal_table` and the others. They reuse the engine of `demo.py`
(`load_fold_models`, `spread_design`, the ensemble in `forecast`), so the app and the command-line
demo give identical numbers for the same design. A test checks this.

### Abstention flag and calibrated band

`results/conformal_r7.json` defines the rule: abstain when the epistemic score (mean ensemble std of mu
divided by mean predictive scale) is above the 90th percentile of the calibration folds. The per-fold
thresholds are not stored in that file, so the app rebuilds them from the bundle:

- **Automatically.** When the server starts and no valid thresholds file exists, it builds one in a
  background thread. This takes about 20 s on a desktop CPU and a few minutes on a 2-vCPU Space. Until the
  build ends, the badge says "abstention check unavailable" and the band is the parametric one. The next
  forecast after the build uses the thresholds. Turn this off with `--no-auto-thresholds` or
  `NT_APP_AUTO_THRESHOLDS=0`.
- **By hand.** Run `python app/app.py --build-thresholds` (add `--workers N` to use N processes).

The build mirrors `scripts/run_conformal.py`: out-of-fold predictions for k = 1-3 and 3 seeded designs
per chemical, with cross-fitted per-fold thresholds. It also computes the per-fold conformal 90 %
multipliers, pooled over cytotoxicity regimes because the bundle has no viability calls. The command
prints its own abstention rate and the error of retained and abstained forecasts next to the published
values in `conformal_r7.json`, so you can check that they match.

The file goes to `outputs/app/abstention_thresholds.json` (set `NT_APP_CACHE` to write elsewhere). The app
also looks for `data_bundle/abstention_thresholds.json`. The file records a fingerprint of the bundle
manifest. If the bundle changes, the app ignores the stale file and builds a new one.

## Deploy on a free Hugging Face Space (not done yet)

The free "CPU basic" hardware is enough: 3 small models per fold, about 30 MB of weights in total.

1. Optional: build the thresholds locally (above) and copy `outputs/app/abstention_thresholds.json` to
   `data_bundle/abstention_thresholds.json`. The Space then shows the badge from the first request.
   Without the file, the Space builds it by itself a few minutes after it starts.
2. Create a Space with the **Gradio** SDK and **CPU basic** hardware.
3. Upload a folder with this layout:

   ```text
   README.md            # Space card with the front matter below
   requirements.txt     # see below
   demo.py
   app/app.py
   src/neurotwin/...
   data_bundle/         # nfa_tasks.npz, transform.json, manifest.json, models/ (+ abstention_thresholds.json)
   results/             # *.json, trajectory_cv_per_chemical.csv, figures/chip_*.png
   scripts/verify_bundle.py
   scripts/run_conformal.py      # the threshold build reuses its k values
   scripts/run_trajectory_cv.py  # ... and the seeded designs of the paper
   ```

   Space `README.md` front matter:

   ```yaml
   ---
   title: NeuroChip Twin v2
   sdk: gradio
   sdk_version: 6.5.1
   app_file: app/app.py
   python_version: "3.12"
   license: mit
   ---
   ```

   `requirements.txt` for the Space. It uses the CPU-only torch wheel. The SDK installs gradio.

   ```text
   --extra-index-url https://download.pytorch.org/whl/cpu
   torch
   numpy<2.6
   pandas
   matplotlib
   ```

4. Push the folder, for example with `hf upload <user>/<space> ./space . --repo-type space`. The
   Space runs `python app/app.py`. Spaces set `GRADIO_SERVER_NAME=0.0.0.0` and the port, and the app
   reads both.

Only Fran publishes: pushing the Space is an external, public action.

## Evidence boundaries

- The data are rat cortical cultures in 48-well MEA plates (US EPA NFA). This is not a validated twin
  of any commercial organ-on-a-chip.
- Compartment scenarios based on the Brewer parameters are simulations.
- Research demonstration only; not for clinical, regulatory or safety decisions.
