"""Train-only median filling, feature choices, one-hot encoding, and scaling."""
import numpy as np
import pandas as pd

NUMERIC = ["model_year", "vehicle_age_years", "odometer_km", "annual_mileage_km",
           "engine_liters", "horsepower", "condition_score", "service_history_score",
           "prior_owners", "accident_count"]
CATEGORICAL = ["brand", "body_style", "fuel_type", "transmission", "region", "seller_type"]
VARIANTS = ["raw", "cap_odometer", "log_odometer", "without_odometer",
            "age_mileage", "age_curvature"]


class Preprocessor:
    def __init__(self, variant="cap_odometer"):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown preprocessing variant: {variant}")
        self.variant = variant

    def _numeric(self, frame):
        columns = NUMERIC.copy()
        if self.variant in ["without_odometer", "age_mileage"]:
            columns.remove("odometer_km")
        missing = set(columns + CATEGORICAL) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing predictor columns: {sorted(missing)}")
        return frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _category(series):
        return series.astype("string").str.strip().replace("", pd.NA).fillna("__MISSING__")

    def fit(self, frame):
        numeric = self._numeric(frame)
        self.numeric_columns_ = numeric.columns.tolist()
        self.medians_ = numeric.median().fillna(0.0)
        self.odometer_cap_ = None
        if self.variant in ["cap_odometer", "age_curvature"]:
            cap = numeric["odometer_km"].quantile(0.99)
            self.odometer_cap_ = float(cap) if pd.notna(cap) else 0.0
        self.levels_, self.frequencies_ = {}, {}
        for column in CATEGORICAL:
            values = self._category(frame[column])
            self.levels_[column] = sorted(values.unique().tolist())
            self.frequencies_[column] = np.array([(values == level).mean()
                                                  for level in self.levels_[column]])
        raw = self._build(frame)
        self.means_ = raw.mean(axis=0)
        self.scales_ = raw.std(axis=0)
        self.scales_[self.scales_ < 1e-12] = 1.0
        return self

    def _build(self, frame):
        numeric = self._numeric(frame)
        indicators = numeric.isna().to_numpy(dtype=float)
        values = numeric.fillna(self.medians_).copy()
        if self.odometer_cap_ is not None:
            values["odometer_km"] = values["odometer_km"].clip(upper=self.odometer_cap_)
        if self.variant == "log_odometer":
            values["odometer_km"] = np.log1p(values["odometer_km"].clip(lower=0))
        if self.variant == "age_mileage":
            values["age_times_annual_mileage"] = values.vehicle_age_years * values.annual_mileage_km
        if self.variant == "age_curvature":
            values["age_squared"] = values.vehicle_age_years ** 2
        blocks = [values.to_numpy(dtype=float), indicators]
        names = values.columns.tolist() + [f"{c}__missing" for c in numeric.columns]
        for column in CATEGORICAL:
            levels = self.levels_[column]
            observed = self._category(frame[column])
            encoded = np.column_stack([(observed == level).to_numpy(dtype=float) for level in levels])
            unknown = ~observed.isin(levels).to_numpy()
            # Unknown levels receive the training-average category contribution.
            encoded[unknown] = self.frequencies_[column]
            blocks.append(encoded)
            names.extend([f"{column}={level}" for level in levels])
        self.feature_names_ = names
        return np.column_stack(blocks)

    def transform(self, frame):
        if not hasattr(self, "means_"):
            raise ValueError("Fit preprocessing on training rows first.")
        transformed = (self._build(frame) - self.means_) / self.scales_
        if not np.isfinite(transformed).all():
            raise ValueError("Preprocessing produced nonfinite features.")
        return transformed

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)
