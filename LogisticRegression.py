import numpy as np
from typing import Callable


class SimpleLogisticRegression:
    def __init__(self, alpha=0.01, max_iter=10000):
        self._alpha = alpha
        self._max_iter = max_iter
        self._thetas = None

    def fit(self, xs: np.matrix, ys: np.array) -> None:
        starter_thetas = np.random.rand(xs.shape[0] + 1)
        self._thetas = gradient_descent(starter_thetas, xs, ys, self._alpha, self._max_iter, _predict)

    def predict(self, xs: np.matrix) -> np.array:
        assert self._thetas is not None, "You forgot to run fit!"
        return _predict(self._thetas, xs)


def gradient_descent(starter_thetas: np.array, xs: np.matrix, ys: np.array, alpha: float, max_iter: int,
                     predict_func: Callable[[np.array, np.matrix], np.array]) -> np.array:
    """ Minimize a function """
    thetas_progress = np.copy(starter_thetas)
    for ii in range(max_iter):
        thetas_progress -= alpha * compute_gradient(thetas_progress, xs, ys, predict_func)
    return thetas_progress


def compute_gradient(thetas: np.array, xs: np.matrix, ys: np.array,
                     predict_func: Callable[[np.array, np.matrix], np.array]) -> np.array:
    predictions = predict_func(thetas, xs)
    diffs = predictions - ys
    grad0 = np.mean(diffs)
    n_rows = xs.shape[0]
    grads_1_plus = np.mean(np.dot(diffs.T, xs)) / n_rows
    return np.append([grad0], grads_1_plus)


def sigmoid(ins: np.array) -> np.array:
    return 1 / (1 + np.exp(ins))


def _predict(thetas: np.array, xs: np.matrix) -> np.array:
    return sigmoid(thetas[0] + np.dot(xs, thetas[1:]))
