"""Full-batch gradient descent. No direct fitting or optimization routines."""
import numpy as np


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


class LinearRegressionGD:
    def __init__(self, learning_rate=0.05, max_iter=150000, tolerance=0.001):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

    @staticmethod
    def objective_gradient(X, y, weights, intercept):
        error = X @ weights + intercept - y
        objective = float(np.mean(error ** 2))
        gradient_w = 2 * X.T @ error / len(y)
        gradient_b = float(2 * error.mean())
        return objective, gradient_w, gradient_b

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("Expected a nonempty feature matrix and matching target vector.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Training inputs must be finite.")
        if self.learning_rate <= 0 or self.max_iter < 1 or self.tolerance < 0:
            raise ValueError("Invalid gradient-descent settings.")
        design = np.column_stack([X, np.ones(len(X))])
        # These products cache the exact batch gradient; no equation is solved.
        gram = design.T @ design / len(y)
        cross = design.T @ y / len(y)
        theta = np.zeros(design.shape[1])
        theta[-1] = y.mean()
        self.history_ = []
        self.converged_ = False
        for iteration in range(self.max_iter + 1):
            gradient = 2 * (gram @ theta - cross)
            largest_gradient = float(np.max(np.abs(gradient)))
            if not np.isfinite(theta).all() or not np.isfinite(gradient).all():
                raise FloatingPointError("Diverged; reduce the learning rate.")
            converged = largest_gradient <= self.tolerance
            if iteration % 100 == 0 or converged or iteration == self.max_iter:
                loss = float(np.mean((design @ theta - y) ** 2))
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; reduce the learning rate.")
                if self.history_ and loss > self.history_[-1]["objective"] + 1e-7 * max(1, self.history_[-1]["objective"]):
                    raise FloatingPointError("Training objective increased; reduce the learning rate.")
                self.history_.append({"iteration": iteration, "objective": loss,
                                      "max_abs_gradient": largest_gradient})
            if converged or iteration == self.max_iter:
                self.converged_ = converged
                break
            theta -= self.learning_rate * gradient
        self.coef_ = theta[:-1]
        self.intercept_ = float(theta[-1])
        self.n_iter_ = iteration
        self.max_abs_gradient_ = largest_gradient
        return self

    def predict(self, X_test):
        if not hasattr(self, "coef_"):
            raise ValueError("Fit the model before predicting.")
        X = np.asarray(X_test, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.coef_) or not np.isfinite(X).all():
            raise ValueError("Prediction features must be finite with the fitted number of columns.")
        prediction = X @ self.coef_ + self.intercept_
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite predictions.")
        return prediction
