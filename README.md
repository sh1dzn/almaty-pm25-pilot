# Almaty PM2.5 forecasting pilot

Hourly forecasts for one Kazhydromet station in Almaty, kgmt_159. Training: 2022–2023. Validation: 2024. Test: 2025. Models: persistence, ridge regression, random forest, XGBoost and LSTM.

The pilot uses 25,795 measured PM2.5 values after local quality control. Missing targets are excluded, and every model at a given horizon is scored on the same test hours. Stochastic models use seeds 11, 22, 33, 44 and 55.

## Test RMSE

Values are in µg/m³. Stochastic model entries are means over five seeds.

| Horizon | Persistence | Ridge | Random forest | XGBoost | LSTM |
| --- | --- | --- | --- | --- | --- |
| 1 h | 12.90 | 12.04 | 12.17 | 11.95 | 13.35 |
| 6 h | 36.15 | 26.30 | 26.74 | 25.42 | 28.60 |
| 12 h | 45.26 | 29.15 | 29.71 | 28.78 | 33.38 |
| 24 h | 35.77 | 30.53 | 32.50 | 31.15 | 31.72 |

The 2024 validation set selected LSTM at 6, 12 and 24 hours, but this ranking did not carry over to 2025. Saved statistical comparisons use the validation-selected model and seed 11, not the best test seed. See `results/dm_tests.csv` for signed loss differences and Holm-adjusted p values.

## Run

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python fetch_data.py
.venv/bin/python run_pilot.py
.venv/bin/python verify_results.py
```

The recorded run used Python 3.12.14 on an Apple M2 Pro with 16 GiB RAM and four CPU threads. It took 300.7 seconds after dependency installation. PyTorch runs in its own process to avoid a macOS OpenMP conflict with XGBoost.

## Data

Air quality: [AirData.kz open dataset](https://github.com/qazybekb/AirDatakz-OpenData), commit `f23d17308a4b51cf5b2855225bd5e440e6713c47`. Weather: ERA5 through the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). File hashes and source URLs are in `data/`.

AirData.kz data are CC BY-NC 4.0 with attribution; the source license is in `data/LICENSE`. The download script verifies the air-quality snapshot. The exact weather snapshot is included and hash-checked, since providers may revise historical data.

## Outputs

`results/` contains dataset counts, seed-wise and seasonal metrics, predictions, input ablations, worst errors, weekly block-bootstrap intervals and Diebold–Mariano comparisons. `config.json` records the experiment settings. `verify_results.py` recomputes reported RMSE values and checks cohort equality, target observations and split boundaries.

## Additional analyses

```sh
.venv/bin/python revision_analysis.py
.venv/bin/python revision_figures.py
.venv/bin/python revision_verify.py
```

These commands reuse the saved predictions without refitting. `revision_v2/` contains a row audit, recomputed main and seasonal metrics, an always-exceedance classifier, 22 unique DM comparisons, paired calendar-block bootstrap intervals, seed aggregation sensitivity, common-target persistence scores and consecutive-hour error groups. Figures are supplied as PDF and SVG.

The primary bootstrap estimand is mean RMSE over the five individual runs. Seed 11 and the five-forecast ensemble are scored separately. Calendar blocks are anchored at 2025-01-01 00:00 UTC, span 168 elapsed hours, retain partial boundaries and omit empty blocks. All models share 1,000 draws within each horizon, using seed 20261025. DM families use the same fixed pairs for seed 11 and ensemble, with separate Holm corrections over 22 tests. Extra comparisons selected by test RMSE remain exploratory.

The original 16-test family remains in `results/dm_tests.csv`; the extended family is in `revision_v2/dm_comparisons.csv`. For six-hour LSTM versus persistence, seed-11 Holm p changes from 0.2638 to 0.3165. Validation RMSE is logged rather than independently rescored. Ablation forecasts were not saved, so their seed metrics can be aggregated but not independently recomputed. `evidence_status.csv` records these distinctions.

To regenerate the seasonal skill matrices and hourly error diagnostics:

```sh
.venv/bin/python revision_result_visuals.py
```

`revision_v3/` contains two model-by-horizon seasonal skill matrices and 24-hour LSTM seed-11 MAE by local target hour. Hourly group counts and weighted MAE reconcile to the original seasonal metrics. These are descriptive summaries of saved forecasts, without new fitting or additional significance claims. Updated figures are available as PDF and SVG.

## Scope

This is a preliminary single-station experiment. It uses a small validation grid instead of the full planned Optuna search and rolling folds. Weather is retrospective reanalysis, not weather information available at forecast issue time. Boundary-layer height was excluded because 4,368 requested hours were missing; T850, other pollutant predictors and SHAP were not included.

Published PM2.5 coverage was 75.20%; it fell to 73.56% after local quality control. The season flag is a fixed 15 October–14 April proxy. Short input gaps are forward-filled for at most three hours. The full four-method artificial-gap benchmark and spatial generalization test have not been run.
