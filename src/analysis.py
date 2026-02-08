"""Analysis and prediction of climate change trends using NOAA data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURE_SERIES = {
    "combined": {
        "label": "Global Combined (Land + Ocean)",
        "url": (
            "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
            "global/time-series?date=2024-12&parameter=tavg&data=trend&trend=12"
            "&baseline=1901&begyear=1880&endyear=2024&format=csv"
        ),
    },
    "land": {
        "label": "Global Land",
        "url": (
            "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
            "global/time-series?date=2024-12&parameter=tavg&data=trend&trend=12"
            "&baseline=1901&begyear=1880&endyear=2024&format=csv&location=land"
        ),
    },
    "ocean": {
        "label": "Global Ocean",
        "url": (
            "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
            "global/time-series?date=2024-12&parameter=tavg&data=trend&trend=12"
            "&baseline=1901&begyear=1880&endyear=2024&format=csv&location=ocean"
        ),
    },
}

CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ClimateTrendBot/1.0)"
}


def download_file(url: str, destination: Path) -> None:
    """Download a file if it does not already exist."""
    if destination.exists():
        return
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def parse_temperature_csv(path: Path) -> pd.DataFrame:
    """Parse NOAA climate-at-a-glance CSV into a standard format."""
    df = pd.read_csv(path)
    column_map = {c.lower(): c for c in df.columns}

    year_col = column_map.get("year") or column_map.get("date")
    value_col = column_map.get("value") or column_map.get("anomaly")

    if year_col is None or value_col is None:
        raise ValueError(f"Unexpected columns in {path.name}: {df.columns}")

    parsed = df[[year_col, value_col]].rename(columns={year_col: "year", value_col: "anomaly"})
    parsed["year"] = pd.to_numeric(parsed["year"], errors="coerce")
    parsed["anomaly"] = pd.to_numeric(parsed["anomaly"], errors="coerce")
    parsed = parsed.dropna().sort_values("year")
    parsed["year"] = parsed["year"].astype(int)
    return parsed


def parse_co2_csv(path: Path) -> pd.DataFrame:
    """Parse NOAA CO2 CSV into annual averages."""
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=[
            "year",
            "month",
            "decimal_date",
            "average",
            "interpolated",
            "trend",
            "days",
        ],
    )
    df = df[df["average"] >= 0]
    annual = df.groupby("year", as_index=False)["average"].mean()
    annual = annual.rename(columns={"average": "co2_ppm"})
    return annual


def fit_and_forecast(series: pd.DataFrame, value_col: str, years_ahead: int = 10) -> pd.DataFrame:
    """Fit a linear regression and forecast future years."""
    model = LinearRegression()
    x = series[["year"]].values
    y = series[value_col].values
    model.fit(x, y)

    last_year = int(series["year"].max())
    future_years = np.arange(last_year + 1, last_year + years_ahead + 1)
    predictions = model.predict(future_years.reshape(-1, 1))

    forecast = pd.DataFrame({"year": future_years, value_col: predictions})
    forecast["model"] = "linear_regression"
    return forecast


def plot_temperature_trends(temp_data: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(10, 6))
    for key, df in temp_data.items():
        plt.plot(df["year"], df["anomaly"], label=TEMPERATURE_SERIES[key]["label"])
    plt.title("Global Temperature Anomalies (NOAA)")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "temperature_trends.png", dpi=150)
    plt.close()


def plot_co2_trend(co2: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(co2["year"], co2["co2_ppm"], color="tab:green")
    plt.title("Global CO₂ Concentration (NOAA)")
    plt.xlabel("Year")
    plt.ylabel("CO₂ (ppm)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "co2_trend.png", dpi=150)
    plt.close()


def plot_temperature_co2_relationship(combined: pd.DataFrame, co2: pd.DataFrame) -> None:
    merged = combined.merge(co2, on="year", how="inner")
    plt.figure(figsize=(8, 6))
    plt.scatter(merged["co2_ppm"], merged["anomaly"], alpha=0.7)
    plt.title("Temperature Anomaly vs CO₂ Concentration")
    plt.xlabel("CO₂ (ppm)")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "temp_co2_relationship.png", dpi=150)
    plt.close()


def plot_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
    output_name: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history["year"], history[value_col], label="Historical", color="tab:blue")
    plt.plot(
        forecast["year"],
        forecast[value_col],
        label="Forecast (Linear Regression)",
        color="tab:orange",
        linestyle="--",
    )
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=150)
    plt.close()


def main() -> None:
    temperature_data: dict[str, pd.DataFrame] = {}
    for key, config in TEMPERATURE_SERIES.items():
        dest = DATA_DIR / f"temperature_{key}.csv"
        download_file(config["url"], dest)
        temperature_data[key] = parse_temperature_csv(dest)

    co2_path = DATA_DIR / "co2_global_monthly.csv"
    download_file(CO2_URL, co2_path)
    co2_annual = parse_co2_csv(co2_path)

    plot_temperature_trends(temperature_data)
    plot_co2_trend(co2_annual)
    plot_temperature_co2_relationship(temperature_data["combined"], co2_annual)

    forecasts = []
    for key, df in temperature_data.items():
        forecast = fit_and_forecast(df, "anomaly")
        forecast["series"] = key
        forecasts.append(forecast)
        if key == "combined":
            plot_forecast(
                df,
                forecast,
                "anomaly",
                "Global Temperature Anomalies with Forecast",
                "Temperature Anomaly (°C)",
                "temperature_forecast.png",
            )
    co2_forecast = fit_and_forecast(co2_annual, "co2_ppm")
    co2_forecast["series"] = "co2"
    forecasts.append(co2_forecast)
    plot_forecast(
        co2_annual,
        co2_forecast,
        "co2_ppm",
        "Global CO₂ Concentration with Forecast",
        "CO₂ (ppm)",
        "co2_forecast.png",
    )

    forecast_table = pd.concat(forecasts, ignore_index=True)
    forecast_table.to_csv(OUTPUT_DIR / "forecast.csv", index=False)

    temperature_combined = temperature_data["combined"].copy()
    temperature_combined["series"] = "combined"
    temperature_combined.to_csv(OUTPUT_DIR / "temperature_combined_annual.csv", index=False)
    co2_annual.to_csv(OUTPUT_DIR / "co2_annual.csv", index=False)

    metadata = {
        "temperature_series": {k: v["url"] for k, v in TEMPERATURE_SERIES.items()},
        "co2_series": CO2_URL,
        "generated_files": [
            "temperature_trends.png",
            "co2_trend.png",
            "temp_co2_relationship.png",
            "temperature_forecast.png",
            "co2_forecast.png",
            "forecast.csv",
        ],
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Analysis complete. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
