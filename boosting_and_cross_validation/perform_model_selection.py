from __future__ import annotations
import numpy as np
from sklearn import datasets
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression
from loss_functions import mean_square_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # load diabetes, 50 random samples for train and the rest for test
    X, y = datasets.load_diabetes(return_X_y=True)
    idx = np.random.permutation(X.shape[0])
    train_X, train_y = X[idx[:n_samples]], y[idx[:n_samples]]
    test_X, test_y = X[idx[n_samples:]], y[idx[n_samples:]]

    # 5 fold cross validation for ridge and lasso over a range of lambdas
    ranges = {"Ridge": np.linspace(1e-4, 0.5, n_evaluations),
              "Lasso": np.linspace(1e-3, 2, n_evaluations)}
    models = {"Ridge": RidgeRegression, "Lasso": Lasso}

    fig = make_subplots(rows=1, cols=2, subplot_titles=list(ranges), horizontal_spacing=.08)
    best = {}
    for i, name in enumerate(ranges):
        scores = np.array([cross_validate(models[name](lam), train_X, train_y, mean_square_error)
                           for lam in ranges[name]])
        fig.add_traces([go.Scatter(x=ranges[name], y=scores[:, 0], mode="lines", name=f"{name} train error"),
                        go.Scatter(x=ranges[name], y=scores[:, 1], mode="lines", name=f"{name} validation error")],
                       rows=1, cols=i + 1)
        best[name] = ranges[name][np.argmin(scores[:, 1])]
    fig.update_layout(title="Train and validation errors (5 fold CV) as a function of lambda",
                      width=1100, height=450)
    fig.update_xaxes(title_text="lambda")
    fig.update_yaxes(title_text="MSE", col=1)
    fig.write_image("cross_validation.png")

    # fit on the whole train set with the best lambdas and check on the test set
    print(f"best ridge lambda: {best['Ridge']:.4f}, best lasso lambda: {best['Lasso']:.4f}")
    final = {"Ridge": RidgeRegression(best["Ridge"]), "Lasso": Lasso(best["Lasso"]),
             "Least Squares": LinearRegression()}
    for name, model in final.items():
        model.fit(train_X, train_y)
        print(f"{name} test error: {model.loss(test_X, test_y):.2f}")


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
