"""Inspect the public CSV without fitting a prediction model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data/car_price_train.csv"


def inspect_data(path=TRAIN_PATH):
    data = pd.read_csv(path)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    summary = pd.DataFrame({"dtype": data.dtypes.astype(str),
                            "missing": data.isna().sum(),
                            "unique_observed": data.nunique()})
    summary.to_csv(output / "data_summary.csv", index_label="column")
    data.describe().T.to_csv(output / "numeric_ranges.csv", index_label="column")
    numeric = data.select_dtypes(include="number")
    assert np.isfinite(numeric.dropna().to_numpy()).all()
    correlations = numeric.corr()
    correlations.to_csv(output / "correlations.csv", index_label="column")
    categories = {c: {str(k): int(v) for k, v in data[c].value_counts(dropna=False).items()}
                  for c in data.select_dtypes(include="object") if c != "listing_id"}
    (output / "category_levels.json").write_text(json.dumps(categories, indent=2) + "\n")
    print(f"Shape: {data.shape}; duplicate IDs: {data.listing_id.duplicated().sum()}")
    print(summary.to_string())
    print("Loan/price correlation:", correlations.loc["loan_approval_value_usd", "sale_price_usd"])
    print("Year/age correlation:", correlations.loc["model_year", "vehicle_age_years"])
    return data


if __name__ == "__main__":
    inspect_data()
