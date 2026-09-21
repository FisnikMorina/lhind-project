# lhind-project

Daily sales forecasting for 70 store-product pairs from 2010 to 2019.

The main model developed in `modeling.ipynb` is a hybrid forecasting model:

1. ARIMA creates the base sales forecast.
2. LightGBM learns the errors made by the ARIMA model.
3. The final prediction is:

   final prediction = ARIMA forecast + predicted error

Grid Search and HyperOpt are used to tune LightGBM, and the experiments are tracked with MLflow.

## Project structure

```
data/
  raw/            original train and test data
  processed/      forecasts, predictions and summary files

notebooks/
  eda.ipynb
  modeling.ipynb
  deep_learning.ipynb

figures/
  charts created by the notebooks

dashboard/
  app.py

src/
  spark_preprocessing.py

requirements.txt
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run the notebooks in order:

1. `notebooks/eda.ipynb` — establishes the findings (per-pair trend, yearly seasonality,
   store/product interaction) that motivate the modeling choices made next.
2. `notebooks/modeling.ipynb` — fits the ARIMA base model, trains LightGBM on its residuals,
   compares Grid Search vs. HyperOpt tuning, evaluates the final hybrid model on the 2019
   held-out year, and logs every run to MLflow.

The EDA is used to understand the data before building the models.

The modeling notebook then:

- fits the ARIMA base models
- creates residuals
- trains regression models on the residuals
- compares Grid Search and HyperOpt
- trains the final model
- evaluates it on 2019
- logs the experiments with MLflow

The ARIMA model is fitted separately for all 70 store-product pairs, so the notebook can take some time to run.

## Results

Scores on the 2019 held-out test year (see `modeling.ipynb` for the full breakdown):

| Model | MAPE | RMSE | R² |
|---|---|---|---|
| Baseline: pair average | 5.14% | 53.5 | 0.936 |
| Base model: ARIMA + yearly waves | 2.64% | 28.7 | 0.982 |
| **Hybrid: base + tuned LightGBM** | **2.10%** | **23.5** | **0.988** |


## MLflow

The project uses MLflow to save model parameters, metrics, charts and the final LightGBM model.

After running modeling.ipynb, start MLflow from the project folder:

export MLFLOW_ALLOW_FILE_STORE=true
mlflow server --backend-store-uri ./mlruns_local --port 5000

Then open:

http://localhost:5000

The MLflow experiment contains runs for:

- the pair-average baseline
- the ARIMA base model
- Grid Search
- HyperOpt
- the final hybrid model

## Dashboard

Run `eda.ipynb` and `modeling.ipynb` first so that the required
statistics and prediction files are available in `data/processed/`.

Then run:

```bash
streamlit run dashboard/app.py
```

This opens a browser dashboard with store/product/date filters and three views: overall trends
and seasonality (heatmaps, day-of-week and monthly patterns), a per-pair explorer, and actual-vs-
predicted charts with anomaly highlighting for the 2018/2019 hybrid model predictions.

## Spark preprocessing

src/spark_preprocessing.py is an optional PySpark version of part of the preprocessing.

It:

- loads the raw CSV files
- checks missing values and duplicates
- creates date features
- creates the store-product pair feature
- calculates descriptive statistics

Run it with:

python src/spark_preprocessing.py

It creates:

data/processed/spark_features.csv
data/processed/spark_series_descriptive_stats.csv

## Deep learning 

`notebooks/deep_learning.ipynb tests a PyTorch neural network instead of LightGBM for predicting the ARIMA residuals.

The same time split is used:

2017 → training
2018 → validation and early stopping
2017 + 2018 → final training
2019 → final test

The neural network uses embeddings for categorical features such as store, product, pair and day of week.

It also uses yearly sine and cosine features to represent the position within the year.

Results:

| Model | 2018 MAPE | 2019 MAPE | 2019 RMSE |
|---|---|---|---|
| Hybrid: base + tuned LightGBM | 1.61% | 2.10% | 23.5 |
| **Hybrid: base + neural network** | **1.53%** | **1.89%** | **19.3** |

The neural network gave the lowest error among the models tested in this project.

The deep learning notebook reuses files created by modeling.ipynb, so run the modeling notebook first.