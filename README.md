# Analysis and Prediction of Climate Change Trends Using NOAA Climate Data

This project downloads NOAA climate datasets, analyzes temperature anomalies and CO₂ concentrations, and produces trend visualizations plus a simple forecast of future trends.

## Project Structure

```
.
├── data
│   └── raw
├── outputs
├── src
│   └── analysis.py
├── requirements.txt
└── README.md
```

## Datasets

- **NOAA Global Surface Temperature** (global land/ocean/combined anomalies):
  https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series
- **NOAA Atmospheric CO₂ Concentration** (global mean monthly):
  https://gml.noaa.gov/ccgg/trends/data.html

The script pulls the CSV endpoints that back these datasets and stores them in `data/raw/`.

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/analysis.py
```

## Outputs

The script writes:

- Cleaned annual temperature anomalies and CO₂ data to `outputs/`.
- Trend visualizations to `outputs/`.
- Forecast charts (next 10 years) for temperature anomalies and CO₂ concentrations.
- A forecast table (next 10 years) for temperature anomalies and CO₂ concentrations.

## Notes

The forecasting approach uses linear regression over historical annual averages. This is meant as a transparent baseline and can be replaced with more advanced models (e.g., ARIMA, Prophet, LSTM) for deeper analysis.
