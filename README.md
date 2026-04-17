# DeepAR probabilistic forecast — US nonfarm payrolls (PAYEMS)

A minimal, reproducible walkthrough of applying [GluonTS DeepAR](https://github.com/awslabs/gluonts)
to a real macroeconomic indicator: BLS/FRED **PAYEMS** (total nonfarm payrolls, monthly, thousands of persons).

## What this demonstrates

- Installing and running GluonTS `gluonts[torch]` on real macro data
- Walk-forward evaluation with a clean train/test split
- Probabilistic output: point forecasts + calibrated uncertainty bands (p10/p50/p90)
- Proper evaluation metrics (MASE, RMSE, MAPE) with an explicit naive benchmark
- A note on time-series indexing notation for anyone coming from non-econometrics backgrounds

## Setup

```bash
pip install "gluonts[torch]"
python forecast.py
```

Requires Python ≥ 3.9.

## Walk-forward design

| Split | Period | Rows |
|-------|--------|------|
| Training | Jan 2010 – Mar 2024 | 171 months |
| Test (hold-out) | Apr 2024 – Mar 2025 | 12 months |

The model is trained once on the training window, then generates
a 12-month probabilistic forecast. No data leakage.

## Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | ~1,095 thousand | Average miss per month-step |
| RMSE | ~1,139 thousand | Penalises large misses more |
| MAPE | ~0.69% | Relative error—small on absolute terms |
| MASE | ~2.31 | Higher than naive benchmark (see note below) |

> **MASE > 1 note:** Nonfarm payrolls is a near-random-walk in the post-COVID recovery
> period (very smooth, slow-moving). A lag-1 naive benchmark on a series this smooth
> is hard to beat without many training series or covariates. The 0.69% MAPE is
> still economically small (~1,100 jobs out of 160 million).

## Time-series indexing notation

A common source of confusion: **when was the forecast made, and for which period?**

Standard econometric notation: **y_{t+h|t}**

- **t** = the forecast origin (last date of training data, "as of" date)
- **h** = horizon (number of steps ahead)
- **y_{t+h|t}** = the forecast for period t+h, made using information up to t

In this project:
- t = **2024-03** (March 2024, last training observation)
- h = 1 → forecast for **2024-04**; h = 12 → forecast for **2025-03**
- When you compare to actuals, you compare **y_{t+h}** (the true value) against **ŷ_{t+h|t}** (the prediction)

**Practical rule**: always name your results file or variable with both the origin
and the target date, e.g. `forecast_origin=2024-03_target=2025-03.json`,
so it's unambiguous which version of the world the model could see when making the call.

## Next release dates for PAYEMS

The BLS releases the Employment Situation summary on the **first Friday of each month**,
covering the prior calendar month. As of April 2025:

- **May 2, 2025** — April 2025 employment data
- **June 6, 2025** — May 2025 employment data

See the [BLS release calendar](https://www.bls.gov/schedule/news_release/empsit.htm).

## Files

| File | Description |
|------|-------------|
| `forecast.py` | Main script — train, evaluate, save results |
| `payems.csv` | PAYEMS data Jan 2010–Mar 2025 (BLS/FRED) |
| `results.json` | Saved forecast output and metrics |

## Data source

Federal Reserve Bank of St. Louis (FRED), series PAYEMS.
BLS Current Employment Statistics (CES) program.
