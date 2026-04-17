"""
DeepAR probabilistic forecast of US nonfarm payrolls (PAYEMS)
FRED series PAYEMS — monthly, thousands of persons
Walk-forward evaluation: train through Mar 2024, forecast Apr 2024–Mar 2025
Metrics: MASE, RMSE, MAPE vs 12-month hold-out actuals
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from gluonts.dataset.pandas import PandasDataset
from gluonts.torch import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions

# ── 1. Load data ──────────────────────────────────────────────────────────────
# Data: BLS/FRED PAYEMS, Jan 2010 – Mar 2025
# Source: https://fred.stlouisfed.org/series/PAYEMS
# To pull live: pip install fredapi; then use fredapi.Fred(api_key='...').get_series('PAYEMS')
df = pd.read_csv('payems.csv', parse_dates=['DATE'], index_col='DATE')
df.index = pd.DatetimeIndex(df.index, freq='MS')

# ── 2. Walk-forward split ─────────────────────────────────────────────────────
# Notation: y_{t+h|t} = forecast made at time t for h steps ahead
# t = T_train (2024-03), h ∈ {1,...,12}, Y_{T_train+h} = held-out actuals
PREDICTION_LENGTH = 12
TRAIN_END = '2024-03'

train_df = df.loc[:TRAIN_END]          # t ≤ T_train
test_df  = df                           # full series (GluonTS needs full context)

train_ds = PandasDataset(train_df, target='PAYEMS', freq='M')
test_ds  = PandasDataset(test_df,  target='PAYEMS', freq='M')

# ── 3. Train DeepAR ──────────────────────────────────────────────────────────
estimator = DeepAREstimator(
    freq='M',
    prediction_length=PREDICTION_LENGTH,
    num_layers=2,
    hidden_size=40,
    trainer_kwargs={'max_epochs': 30}
)
predictor = estimator.train(train_ds)

# ── 4. Forecast ──────────────────────────────────────────────────────────────
forecast_it, _ = make_evaluation_predictions(
    dataset=test_ds, predictor=predictor, num_samples=200
)
fc = list(forecast_it)[0]

# Forecast origin: the last date of the training window
# y_{t+h|t} for t=2024-03, h=1 → forecast for 2024-04
fc_dates = pd.date_range('2024-04-01', periods=PREDICTION_LENGTH, freq='MS')
actuals  = df.loc['2024-04-01':, 'PAYEMS'].values

# ── 5. Metrics ────────────────────────────────────────────────────────────────
# MASE: MAE(forecast) / MAE(naive lag-1 in-sample benchmark)
# Values < 1 mean the model beats a random-walk benchmark
in_sample_naive_mae = np.mean(np.abs(np.diff(train_df['PAYEMS'].values)))
mae  = np.mean(np.abs(actuals - fc.mean))
mase = mae / in_sample_naive_mae
rmse = np.sqrt(np.mean((actuals - fc.mean)**2))
mape = np.mean(np.abs((actuals - fc.mean) / actuals)) * 100

print(f"\nWalk-forward evaluation  (T_train=2024-03, h=1..12)")
print(f"  MAE  = {mae:,.0f} thousand jobs")
print(f"  RMSE = {rmse:,.0f} thousand jobs")
print(f"  MAPE = {mape:.3f}%")
print(f"  MASE = {mase:.4f}  (< 1 beats naive lag-1 benchmark)")
print(f"\nPoint forecast  vs  actuals (Apr 2024 – Mar 2025):")
print(f"{'Month':<10}  {'Actual':>10}  {'Pred':>10}  {'Error':>8}  {'In 80% CI?':>12}")
for i in range(len(actuals)):
    a   = actuals[i]
    p   = fc.mean[i]
    err = a - p
    lo  = fc.quantile(0.10)[i]
    hi  = fc.quantile(0.90)[i]
    covered = 'yes' if lo <= a <= hi else 'NO'
    print(f"{str(fc_dates[i])[:7]:<10}  {a:>10,.0f}  {p:>10,.0f}  {err:>+8,.0f}  {covered:>12}")

# ── 6. Save results ──────────────────────────────────────────────────────────
results = {
    'model': 'DeepAR (GluonTS v0.16.2)',
    'series': 'PAYEMS (BLS/FRED) — total nonfarm payrolls, thousands',
    'train_cutoff': TRAIN_END,
    'forecast_origin_notation': 'y_{t+h|t}: t=2024-03, h=1..12',
    'metrics': {'mae': round(float(mae),1), 'rmse': round(float(rmse),1),
                'mape_pct': round(float(mape),4), 'mase': round(float(mase),4)},
    'forecast': [
        {'date': str(fc_dates[i])[:7],
         'actual': float(actuals[i]) if i < len(actuals) else None,
         'pred_mean': round(float(fc.mean[i]),1),
         'pred_p10':  round(float(fc.quantile(0.10)[i]),1),
         'pred_p90':  round(float(fc.quantile(0.90)[i]),1)}
        for i in range(PREDICTION_LENGTH)
    ]
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved results.json")
