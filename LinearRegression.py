import numpy as np
from numpy import subtract, mean, square


class SimpleLinearRegression:
    def __init__(self, alpha=.01, max_iter=10000):
        self._alpha = alpha
        self._max_iter = max_iter
        self._thetas = None

    def fit(self, xx: np.matrix, yy: np.array) -> None:
        n_columns = xx.shape[1]
        thetas_init = np.random.rand(n_columns + 1)
        self._thetas = self._gradient_descent(thetas_init, xx, yy, self._max_iter)

    def predict(self, xs: np.matrix) -> np.array:
        return self._thetas[0] + np.dot(xs, self._thetas[1:])

    def _gradient_descent(self, thetas, xs: np.matrix, ys: np.array, max_iter: int) -> np.array:
        thetas_progress = np.copy(thetas)
        for ii in range(max_iter):
            if ii % 500 == 0:
                print(thetas_progress, _loss(self.predict(thetas_progress, xs), ys))
            thetas_progress -= self._alpha * _compute_gradient(thetas_progress, xs, ys)
        return thetas_progress


def _compute_gradient(thetas: np.array, xs: np.matrix, ys: np.array) -> np.array:
    predictions = _predict_from_thetas(thetas, xs)
    error = subtract(predictions, ys)
    grad0 = mean(error)
    n_rows = xs.shape[0]
    other_grads = np.dot(error.T, xs) / n_rows
    return np.append([grad0], other_grads)


def _predict_from_thetas(thetas: np.array, xs: np.matrix) -> np.array:
    return thetas[0] + np.dot(xs, thetas[1:])


def _loss(predictions: np.array, actual: np.array) -> float:
    """ Mean Squared Error """
    return mean(square(subtract(predictions, actual)))
