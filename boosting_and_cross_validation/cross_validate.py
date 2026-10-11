from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from base_estimator import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    folds = np.array_split(np.arange(X.shape[0]), cv)

    train_scores, val_scores = [], []
    for fold in folds:
        # this fold is the validation set and the rest is the train set
        train_mask = np.ones(X.shape[0], dtype=bool)
        train_mask[fold] = False

        model = deepcopy(estimator).fit(X[train_mask], y[train_mask])
        train_scores.append(scoring(y[train_mask], model.predict(X[train_mask])))
        val_scores.append(scoring(y[fold], model.predict(X[fold])))

    return float(np.mean(train_scores)), float(np.mean(val_scores))
