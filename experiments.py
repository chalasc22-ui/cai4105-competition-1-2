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


def part2(data):
    baseline_path = RESULTS / "part1_baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError("Complete Part I first: python experiments.py --part 1")
    baseline = json.loads(baseline_path.read_text())
    settings, variant = baseline["settings"], baseline["variant"]
    prep, X, V, y, v = prepare_split(data, variant, baseline["split_seed"])
    rows = []
    for strength in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        model, record = fit_record(X, V, y, v, {**settings, "l2": strength})
        rows.append(record)
        pd.DataFrame(model.history_).to_csv(RESULTS / f"l2_{strength:g}_loss_trace.csv", index=False)
        if strength == 0:
            frozen = np.load(RESULTS / "part1_baseline_parameters.npz")
            difference = float(np.max(np.abs(model.predict(V) - frozen["validation_predictions"])))
            assert difference < 1e-6, "L2=0 did not reproduce the Part I predictions."
            save_json("lambda_zero_check.json", {"max_prediction_difference_usd": difference,
                      "same_coefficients": bool(np.allclose(model.coef_, frozen["coef"], atol=1e-8, rtol=0))})
        print("lambda comparison", record, flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "part2_lambda_comparison.csv", index=False)
    positive = float(comparison[comparison.l2 > 0].sort_values("validation_rmse").iloc[0].l2)
    repeated = []
    # These five predeclared seeds are separate from the initial selection seed.
    for seed in [7, 21, 84, 123, 2026]:
        prep, X, V, y, v = prepare_split(data, variant, seed)
        for strength in [0.0, positive]:
            model, record = fit_record(X, V, y, v, {**settings, "l2": strength})
            repeated.append({"seed": seed, **record})
            print("repeated", repeated[-1], flush=True)
    repeated = pd.DataFrame(repeated)
    repeated.to_csv(RESULTS / "part2_repeated_splits.csv", index=False)
    summary = repeated.groupby("l2").validation_rmse.agg(["mean", "min", "max"])
    summary.to_csv(RESULTS / "part2_summary.csv")
    # Sub-dollar differences in RMSE are treated as a practical tie.
    improvement = float(summary.loc[0.0, "mean"] - summary.loc[positive, "mean"])
    selected = positive if improvement > 1.0 else 0.0
    decision = {"variant": variant, "settings": settings, "l2": selected,
                "selected_positive_l2": positive,
                "decision_rule": "Choose positive L2 only if five-split mean RMSE improves by more than $1; otherwise keep Part I.",
                "rmse_tie_tolerance_usd": 1.0, "positive_improvement_usd": improvement,
                "repeated_split_seeds": [7, 21, 84, 123, 2026],
                "unregularized_mean_rmse": float(summary.loc[0.0, "mean"]),
                "positive_mean_rmse": float(summary.loc[positive, "mean"]),
                "positive_wins": int(sum(repeated[repeated.l2 == positive].validation_rmse.to_numpy()
                                         < repeated[repeated.l2 == 0].validation_rmse.to_numpy()))}
    save_json("selected_config.json", decision)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [f"{value:g}" for value in comparison.l2]
    axes[0].plot(labels, comparison.validation_rmse, marker="o", color="#24577b")
    axes[0].set(xlabel="L2 strength", ylabel="Validation RMSE (USD)", title="Controlled comparison: seed 42")
    axes[1].plot(labels, comparison.coefficient_norm, marker="o", color="#ba552c")
    axes[1].set(xlabel="L2 strength", ylabel="Coefficient L2 norm", title="Shrinkage with standardized features")
    figure.tight_layout()
    figure.savefig(RESULTS / "part2_comparison.png", dpi=180)
    plt.close(figure)
    print("FINAL CHOICE", decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["1", "2"], default="1")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    args = parser.parse_args()
    data = pd.read_csv(args.train)
    if args.part == "1":
        part1(data)
    else:
        part2(data)
