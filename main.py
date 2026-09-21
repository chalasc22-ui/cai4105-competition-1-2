"""Refit the selected configuration and predict the instructor's feature CSV."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import numpy as np
import pandas as pd
from preprocessing import Preprocessor
from regression import LinearRegressionGD

# Change these paths, or use the command-line options shown in README.md.
ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data/car_price_train.csv"
TEST_PATH = ROOT / "hidden_data/car_price_test.csv"
PREDICTIONS_PATH = ROOT / "predictions.csv"
CONFIG_PATH = ROOT / "results/selected_config.json"
MODEL_PATH = ROOT / "artifacts/model.pkl"


def read_listings(path, require_target=False):
    data = pd.read_csv(path, dtype={"listing_id": "string"})
    if "listing_id" not in data:
        raise ValueError("CSV must contain listing_id.")
    ids = data.listing_id
    if len(data) == 0 or ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("CSV must have rows and unique, nonempty listing IDs.")
    if require_target:
        if "sale_price_usd" not in data:
            raise ValueError("Training data must contain sale_price_usd.")
        data["sale_price_usd"] = pd.to_numeric(data.sale_price_usd, errors="coerce")
        if not np.isfinite(data.sale_price_usd).all():
            raise ValueError("All training targets must be numeric and finite.")
    return data


def train_selected(train_path=TRAIN_PATH, model_path=MODEL_PATH, config_path=CONFIG_PATH):
    data = read_listings(train_path, require_target=True)
    config = json.loads(Path(config_path).read_text())
    prep = Preprocessor(config["variant"])
    X = prep.fit_transform(data)
    model = LinearRegressionGD(**config["settings"], l2=config["l2"]).fit(X, data.sale_price_usd.to_numpy())
    if not model.converged_:
        raise RuntimeError("Final model did not converge. Inspect the training settings before predicting.")
    metadata = {"rows": len(data), "features": X.shape[1], "config": config,
                "iterations": model.n_iter_, "max_abs_gradient": model.max_abs_gradient_,
                "train_sha256": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
                "python_version": sys.version.split()[0], "numpy_version": np.__version__,
                "pandas_version": pd.__version__}
    bundle = {"preprocessor": prep, "model": model, "metadata": metadata}
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as output:
        pickle.dump(bundle, output)
    (model_path.parent / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pd.DataFrame({"feature": prep.feature_names_, "coefficient": model.coef_}).to_csv(
        model_path.parent / "coefficients.csv", index=False)
    pd.DataFrame(model.history_).to_csv(model_path.parent / "final_loss_trace.csv", index=False)
    print(f"Fitted selected pipeline on {len(data)} public rows; L2={config['l2']:g}.")
    return bundle


def write_predictions(bundle, test_path, output_path=PREDICTIONS_PATH, expected_rows=600):
    test = read_listings(test_path)
    if expected_rows is not None and len(test) != expected_rows:
        raise ValueError(f"Expected {expected_rows} test rows, found {len(test)}.")
    X = bundle["preprocessor"].transform(test)
    predictions = bundle["model"].predict(X)
    output = pd.DataFrame({"listing_id": test.listing_id,
                           "sale_price_usd_pred": predictions})
    if len(output) != len(test) or not np.isfinite(output.sale_price_usd_pred).all():
        raise ValueError("Every test ID must have one finite prediction.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(output_path)
    print(f"Wrote {len(output)} predictions to {output_path}.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "predict"], default="predict")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--expected-rows", type=int, default=600)
    args = parser.parse_args()
    if args.mode == "train":
        train_selected(args.train, args.model, args.config)
    else:
        if not args.test.exists():
            parser.error(f"Test CSV not found: {args.test}. Set --test to the instructor's CSV.")
        if args.model.exists():
            # Load only a model file produced by this project, not an unknown pickle.
            with args.model.open("rb") as saved:
                bundle = pickle.load(saved)
            selected = json.loads(args.config.read_text())
            if bundle["metadata"]["config"] != selected:
                parser.error("Selected configuration changed. Run --mode train to refit.")
            current_hash = hashlib.sha256(args.train.read_bytes()).hexdigest()
            if bundle["metadata"]["train_sha256"] != current_hash:
                parser.error("Training data changed. Run --mode train to refit.")
        else:
            bundle = train_selected(args.train, args.model, args.config)
        write_predictions(bundle, args.test, args.output, args.expected_rows)
