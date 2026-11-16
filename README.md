# Cloud Workload Forecasting

Short-horizon forecasting of Azure VM workload (2 hours ahead, 5-minute steps),
and a simple capacity rule for an infrastructure team.

Built for the Atomity AI/ML take-home challenge, on the `data-v1` subset of the Microsoft Azure Public Dataset V2.

## Results at a glance

| Target | Model | Validation MAE (7 folds) | Holdout MAE | Persistence holdout MAE |
|---|---|---|---|---|
| `aggregate_cpu_utilization` | Ridge autoregression, 3 lags | 0.198 | 0.098 | 0.147 |
| `active_vm_count` | Persistence | 0 | 0 | 0 |

- The ridge model has 25% lower error than persistence in validation, and 33% lower on the holdout.
  It wins on 4 of 7 validation folds, so the gain is real but not uniform.
- `active_vm_count` is constant (10,000) because the dataset keeps only VMs with a complete trace.
- Capacity rule: plan for forecast + 0.20 CPU points, and raise the headroom to +0.45 points
  when the P95 of `max_cpu` across VMs goes above 70.2%.

The full analysis is in [`reports/final_report.md`](reports/final_report.md).

## Repository structure

```
src/cwfc/
  data.py          load files, validate readings, build targets, train/holdout split
  models.py        persistence, moving average, ridge autoregression
  evaluation.py    rolling-origin backtesting and model comparison
  pipeline.py      command line entry point that writes predictions.csv
  capacity.py      headroom from validation errors, capacity plan scoring
  metrics.py       MAE, RMSE, sMAPE, normalized MAE (from the starter code)
  baselines.py     starter baselines (kept unchanged)
  validation.py    starter split helper (kept unchanged)
notebooks/
  01_eda.ipynb                  data quality and workload analysis
  02_modeling.ipynb             model selection on the training window
  03_holdout_evaluation.ipynb   final score on the holdout and error analysis
  04_capacity.ipynb             capacity recommendation
tests/                          unit tests (pytest)
reports/final_report.md         final report
predictions.csv                 holdout predictions
```

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate
pip install -e ".[dev]"            # pipeline and tests
pip install -e ".[dev,notebooks]"  # also needed to run the notebooks
```

## Data

Download `data-v1.tar.gz` from the `data-v1` release of the challenge repository, then:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
cd challenge_data && sha256sum -c SHA256SUMS && cd ..
```

The data is not committed to this repository.

## Run

One command builds the targets, fits the models on the training window and writes the predictions:

```bash
python -m cwfc.pipeline --data challenge_data --output predictions.csv
```

Options: `--data` (folder with `vm_cpu_readings.csv.gz`), `--output` (CSV path), `--holdout-size` (default 24).
The command checks the output against the submission contract before writing it.
It only needs `vm_cpu_readings.csv.gz`, and it works on any dataset with the same columns.

Tests:

```bash
pytest -q
```

Notebooks: run them in order, from the project root or from `notebooks/`.

## Modelling decisions

- **Valid observation:** a row with a timestamp, a `vm_id`, and `avg_cpu` between 0 and 100.
  Invalid rows are dropped, never corrected. The release has none.
- **Duplicates:** a VM counts once per timestamp; the first row is kept. The release has none.
- **Gaps:** missing timestamps are reported, not filled. Target values are never imputed. The release has none.
- **Split:** the full series is built first, then the last 24 timestamps form the holdout.
  The holdout is never used for fitting, model selection or tuning.
- **Validation:** rolling origin on the 110 training timestamps. Each fold trains on the past and forecasts
  the next 24 steps, the same horizon as the holdout. Origins every 6 steps, 7 folds.
- **Model selection:** 7 candidates (persistence, 3 moving averages, 3 ridge autoregressions).
  `ridge_ar_lags3` has the lowest MAE and RMSE; with 6 lags the score is almost the same, so I keep the simpler model.
- **Features:** only past values of the target. VM metadata is not used as a feature,
  because fields such as the deletion time and VM-level CPU statistics describe the whole VM life, including the future.
- **Capacity:** headroom = 80th percentile of under-forecast errors in validation.
  The level comes from a 5:1 cost ratio (running short vs idle capacity), not from holdout results.

## Assumptions and limitations

- The window is only 11 hours: no daily or weekly seasonality can be learned.
- The dataset has no VM churn, so the VM count forecast is trivial here.
- The holdout is a single 2-hour window with one peak point; validation results are the more reliable measure.
- The 5:1 cost ratio is an assumption and should come from the infrastructure team.

## Use of AI tools

I used Claude (chat assistant) for guidance, code suggestions and review.
It had no access to my repository: I wrote and ran every step myself, checked the results,
and made the analysis and modelling choices.