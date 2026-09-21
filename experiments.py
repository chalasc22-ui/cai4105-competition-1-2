"""Reproducible Part I experiments, followed later by a separate Part II."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocessing import Preprocessor, VARIANTS
from regression import LinearRegressionGD, rmse

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data/car_price_train.csv"
RESULTS = ROOT / "results"
SPLIT_SEED = 42


def split_rows(data, seed=42):
    order = np.random.default_rng(seed).permutation(len(data))
    validation_size = int(round(0.20 * len(data)))
    return data.iloc[order[validation_size:]].copy(), data.iloc[order[:validation_size]].copy()


def prepare_split(data, variant, seed=42):
    train, validation = split_rows(data, seed)
    prep = Preprocessor(variant)
    X_train = prep.fit_transform(train)
    X_valid = prep.transform(validation)
    return prep, X_train, X_valid, train.sale_price_usd.to_numpy(), validation.sale_price_usd.to_numpy()


def fit_record(X_train, X_valid, y_train, y_valid, settings):
    model = LinearRegressionGD(**settings).fit(X_train, y_train)
    record = {**settings, "train_rmse": rmse(y_train, model.predict(X_train)),
              "validation_rmse": rmse(y_valid, model.predict(X_valid)),
              "coefficient_norm": float(np.sqrt(np.sum(model.coef_ ** 2))),
              "iterations": model.n_iter_, "converged": model.converged_,
              "max_abs_gradient": model.max_abs_gradient_}
    return model, record


def save_json(name, content):
    (RESULTS / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def part1(data):
    RESULTS.mkdir(exist_ok=True)
    defaults = {"learning_rate": 0.05, "max_iter": 150000, "tolerance": 0.001}
    rows = []
    for variant in VARIANTS:
        prep, X, V, y, v = prepare_split(data, variant, SPLIT_SEED)
        model, record = fit_record(X, V, y, v, defaults)
        rows.append({"variant": variant, **record})
        print(variant, record, flush=True)
    features = pd.DataFrame(rows)
    features.to_csv(RESULTS / "part1_feature_comparison.csv", index=False)
    winner = features.sort_values("validation_rmse").iloc[0]
    variant = winner["variant"]
    prep, X, V, y, v = prepare_split(data, variant, SPLIT_SEED)
    settings_grid = [defaults]
    settings_grid += [{**defaults, "learning_rate": rate} for rate in [0.01, 0.10, 0.50]]
    settings_grid += [{**defaults, "max_iter": count} for count in [1000, 10000, 50000]]
    settings_grid += [{**defaults, "tolerance": tol} for tol in [0.1, 0.00001]]
    tuning = []
    for settings in settings_grid:
        try:
            model, record = fit_record(X, V, y, v, settings)
            tuning.append({**record, "status": "ok"})
        except FloatingPointError as error:
            tuning.append({**settings, "status": str(error)})
        print("tuning", tuning[-1], flush=True)
    tuning = pd.DataFrame(tuning)
    tuning.to_csv(RESULTS / "part1_optimizer_comparison.csv", index=False)
    # Require convergence so under-training does not stand in for regularization.
    eligible = tuning[(tuning.status == "ok") & (tuning.converged == True)]
    best = eligible.sort_values("validation_rmse").iloc[0]
    settings = {"learning_rate": float(best.learning_rate), "max_iter": int(best.max_iter),
                "tolerance": float(best.tolerance)}
    model, record = fit_record(X, V, y, v, settings)
    baseline = {"variant": variant, "split_seed": SPLIT_SEED, "validation_fraction": 0.2,
                "l2": 0.0, "settings": settings, "n_features": X.shape[1], **record}
    save_json("part1_baseline.json", baseline)
    np.savez(RESULTS / "part1_baseline_parameters.npz", coef=model.coef_, intercept=model.intercept_,
             validation_predictions=model.predict(V))
    pd.DataFrame(model.history_).to_csv(RESULTS / "part1_loss_trace.csv", index=False)
    train, validation = split_rows(data, SPLIT_SEED)
    pd.DataFrame({"listing_id": validation.listing_id, "actual": v,
                  "prediction": model.predict(V)}).to_csv(RESULTS / "part1_validation_predictions.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    trace = pd.DataFrame(model.history_)
    axes[0].plot(trace.iteration, trace.objective, color="#24577b")
    axes[0].set(xlabel="Gradient-descent updates", ylabel="Training MSE (USD squared)",
                title="Unregularized training loss", yscale="log")
    axes[1].scatter(v, model.predict(V), s=16, alpha=0.55, color="#24577b")
    lo, hi = min(v.min(), model.predict(V).min()), max(v.max(), model.predict(V).max())
    axes[1].plot([lo, hi], [lo, hi], "--", color="#ba552c")
    axes[1].set(xlabel="Actual price (USD)", ylabel="Predicted price (USD)", title="Seed 42 validation")
    figure.tight_layout()
    figure.savefig(RESULTS / "part1_diagnostics.png", dpi=180)
    plt.close(figure)
    print("FROZEN BASELINE", baseline, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["1"], default="1")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    args = parser.parse_args()
    part1(pd.read_csv(args.train))
