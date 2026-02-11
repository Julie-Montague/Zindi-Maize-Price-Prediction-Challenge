# Zindi-Maize-Price-Prediction-Challenge
Using historical prices of dry maize in Kenya, this project develops a machine learning solution to predict average weekly prices of maize in the counties of Kiambu, Kirinyaga, Mombasa, Nairobi and Uasin-Gishu.

## REPOSITORY LAYOUT
```
.
├── Modelling_Final.ipynb
├── data/
│   ├── agriBORA_maize_prices.csv
│   ├── agriBORA_maize_prices_weeks_46_to_51.csv
│   ├── kamis_maize_prices.csv
│   └── (optional) kamis_maize_prices_downloaded.csv
└── modelling_results/
    └── <exp_code>/   # run artifacts (if you choose to save models)
```
The notebook creates a run-specific experiment code (`exp_code`) and uses it to build `OUTPUT_DIR`.

## Data requirements

### AgriBORA inputs
The notebook expects at minimum:
- `Date` (parseable date)
- `County`
- `Commodity_Classification`
- `Wholesale` (used as the target price series)

It filters to:
- `COMMODITY_CLASS = "Dry_White_Maize"`
- `TARGET_COUNTIES = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]`

### KAMIS inputs
The cleaned KAMIS file is expected to contain (case-insensitive; notebook lowercases columns):
- `date`
- `county`
- `classification` (used to select `White_Maize`)
- `retail`, `wholesale`
- `supplyvolume`

## Configuration knobs (top of notebook)

Key settings you can edit:

- **Paths**
  - `INPUT_DIR`, `AGRIBORA_FULL_PATH`, `AGRIBORA_RECENT_PATH`, `KAMIS_PATH`

- **Forecast**
  - `FORECAST_DATES` (week-start Mondays)
  - `ANCHOR_DATE` (last observed week used as the forecast base)

- **Backtesting**
  - `N_BACKTEST_CUTOFFS` (number of cutoffs to test)
  - `H_MAX` (how far to extend the weekly grid)

- **Model selection**
  - `best_models = {1: "catboost", 2: "mlp"}` (example in the notebook)
