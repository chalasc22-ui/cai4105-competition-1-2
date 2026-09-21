"""Focused checks for gradients, leakage prevention, and prediction-file safety."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from preprocessing import Preprocessor, NUMERIC, CATEGORICAL
from regression import LinearRegressionGD, rmse
from main import read_listings, write_predictions, TRAIN_PATH


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(TRAIN_PATH)

    def test_known_line_and_constant_target(self):
        X = np.arange(-3., 4.)[:, None]
        model = LinearRegressionGD(tolerance=1e-9).fit(X, 4 * X[:, 0] + 12)
        self.assertLess(rmse(4 * X[:, 0] + 12, model.predict(X)), 1e-7)
        constant = LinearRegressionGD(l2=1.0).fit(X, np.full(7, 100.0))
        np.testing.assert_allclose(constant.predict(X), 100.0, atol=1e-10)

    def test_finite_difference_gradients(self):
        X = np.array([[1., -2.], [0., 1.], [-1., 3.]])
        y, w, b = np.array([2., 0., 4.]), np.array([0.4, -0.3]), 0.2
        step = 1e-6
        for strength in [0.0, 0.2]:
            objective, gw, gb = LinearRegressionGD.objective_gradient(X, y, w, b, strength)
            for column in range(len(w)):
                delta = np.zeros_like(w)
                delta[column] = step
                plus = LinearRegressionGD.objective_gradient(X, y, w + delta, b, strength)[0]
                minus = LinearRegressionGD.objective_gradient(X, y, w - delta, b, strength)[0]
                self.assertAlmostEqual(gw[column], (plus - minus) / (2 * step), places=6)
            plus = LinearRegressionGD.objective_gradient(X, y, w, b + step, strength)[0]
            minus = LinearRegressionGD.objective_gradient(X, y, w, b - step, strength)[0]
            self.assertAlmostEqual(gb, (plus - minus) / (2 * step), places=6)

    def test_zero_l2_and_known_ridge_solution(self):
        X = np.array([[-1.], [0.], [1.]])
        y = 3 * X[:, 0] + 5
        plain = LinearRegressionGD(tolerance=1e-9).fit(X, y)
        zero = LinearRegressionGD(tolerance=1e-9, l2=0).fit(X, y)
        np.testing.assert_array_equal(plain.predict(X), zero.predict(X))
        ridge = LinearRegressionGD(tolerance=1e-9, l2=1/3).fit(X, y)
        self.assertAlmostEqual(ridge.coef_[0], 2.0, places=7)
        self.assertAlmostEqual(ridge.intercept_, 5.0, places=7)

    def test_cached_updates_equal_residual_updates(self):
        X = np.array([[-1., 2.], [0., 1.], [1., 3.]])
        y = np.array([1., 2., 4.])
        for strength in [0.0, 0.1]:
            weights, intercept = np.zeros(2), y.mean()
            for _ in range(12):
                _, gw, gb = LinearRegressionGD.objective_gradient(X, y, weights, intercept, strength)
                weights -= 0.01 * gw
                intercept -= 0.01 * gb
            model = LinearRegressionGD(learning_rate=.01, max_iter=12, tolerance=0, l2=strength).fit(X, y)
            np.testing.assert_allclose(model.coef_, weights, atol=1e-12)
            self.assertAlmostEqual(model.intercept_, intercept, places=12)

    def test_preprocessing_is_train_only_and_excludes_leakage(self):
        train = self.data.iloc[:100].copy()
        train["engine_liters"] = 2.0
        prep = Preprocessor("age_mileage").fit(train)
        validation = self.data.iloc[100:105].copy()
        validation["engine_liters"] = 999.0
        prep.transform(validation)
        self.assertEqual(prep.medians_["engine_liters"], 2.0)
        altered = validation.copy()
        altered["loan_approval_value_usd"] = -999999
        altered["sale_price_usd"] = -999999
        altered["listing_id"] = "different"
        np.testing.assert_array_equal(prep.transform(validation), prep.transform(altered))

    def test_missing_unknown_reordered_and_all_missing_training(self):
        train = self.data.iloc[:100].copy()
        train["engine_liters"] = np.nan
        prep = Preprocessor("age_mileage").fit(train)
        sample = self.data.iloc[100:110].copy()
        for column in NUMERIC:
            sample[column] = np.nan
        for column in CATEGORICAL:
            sample[column] = "unseen level"
        first = prep.transform(sample)
        second = prep.transform(sample[sample.columns[::-1]])
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(first.shape[1], len(prep.feature_names_))

    def test_six_hundred_predictions_preserve_ids_and_schema(self):
        prep = Preprocessor("age_mileage").fit(self.data.iloc[:200])
        X = prep.transform(self.data.iloc[:200])
        model = LinearRegressionGD(max_iter=2000, l2=.01).fit(X, self.data.sale_price_usd.iloc[:200].to_numpy())
        sample = self.data.iloc[:600].drop(columns="sale_price_usd").copy()
        sample["listing_id"] = [f"CHECK-{i:04d}" for i in range(600)]
        sample.loc[sample.index[::2], "engine_liters"] = np.nan
        sample.loc[sample.index[::3], "fuel_type"] = "new_fuel"
        with tempfile.TemporaryDirectory() as folder:
            path, output = Path(folder) / "features.csv", Path(folder) / "output.csv"
            sample.to_csv(path, index=False)
            result = write_predictions({"preprocessor": prep, "model": model}, path, output)
            reread = pd.read_csv(output)
            self.assertEqual(reread.columns.tolist(), ["listing_id", "sale_price_usd_pred"])
            self.assertEqual(reread.listing_id.tolist(), sample.listing_id.tolist())
            self.assertEqual(len(result), 600)
            self.assertTrue(np.isfinite(reread.sale_price_usd_pred).all())
            sample.loc[1, "listing_id"] = sample.loc[0, "listing_id"]
            sample.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_listings(path)

    def test_rejects_nonfinite_training_and_unstable_rate(self):
        with self.assertRaises(ValueError):
            LinearRegressionGD().fit(np.array([[np.inf]]), np.array([1.]))
        X = np.array([[-1.], [0.], [1.]])
        with self.assertRaises(FloatingPointError):
            LinearRegressionGD(learning_rate=5., max_iter=100).fit(X, np.array([-2., 0., 2.]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
