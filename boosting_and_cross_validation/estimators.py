from __future__ import annotations
from typing import NoReturn
import numpy as np
from sklearn import linear_model
from base_estimator import BaseEstimator
from loss_functions import mean_square_error


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `RidgeRegression.fit` function.
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            X = np.column_stack((np.ones(X.shape[0]), X))

        # closed form: (X^T X + lam I) w = X^T y
        reg = self.lam_ * np.eye(X.shape[1])
        if self.include_intercept_:
            # don't regularize the intercept
            reg[0, 0] = 0
        self.coefs_ = np.linalg.solve(X.T @ X + reg, X.T @ y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            X = np.column_stack((np.ones(X.shape[0]), X))
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self._predict(X))


class Lasso(BaseEstimator):
    """
    Lasso Regression Estimator, based on scikit learn's implementation
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> Lasso:
        """
        Initialize a lasso regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        self.lam_ = lam
        self.include_intercept_ = include_intercept
        self.model_ = linear_model.Lasso(alpha=lam, fit_intercept=include_intercept, max_iter=10000)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self.model_.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self._predict(X))


class LinearRegression(BaseEstimator):
    """
    Least Squares Estimator, based on scikit learn's implementation
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Initialize a least squares model

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.model_ = linear_model.LinearRegression(fit_intercept=include_intercept)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self.model_.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self._predict(X))
