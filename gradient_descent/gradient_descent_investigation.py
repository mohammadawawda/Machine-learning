import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR




from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from cross_validate import cross_validate
from loss_functions import misclassification_error


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def callback(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for module, name in [(L1, "L1"), (L2, "L2")]:
        conv = go.Figure()
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gd.fit(module(weights=init.copy()), X=None, y=None)

            # descent path, starting from the initial point
            path = np.array([init] + weights)
            plot_descent_path(module, path, title=f"{name}, eta={eta}") \
                .write_image(f"descent_path_{name}_eta_{eta}.png")

            # the norm in every iteration
            conv.add_trace(go.Scatter(x=np.arange(1, len(values) + 1), y=values, mode="lines", name=f"eta={eta}"))

            # lowest loss
            print(f"{name}, eta={eta}: lowest loss {np.min(values):.6f} after {len(values)} iterations")

        conv.update_layout(title=f"{name} norm as a function of the GD iteration",
                           xaxis_title="iteration", yaxis_title="norm")
        conv.write_image(f"convergence_{name}.png")


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values

    # logistic regression without regularization and its roc curve
    solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    model = LogisticRegression(solver=solver).fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_train, model.predict_proba(X_train))

    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash="dash"),
                                showlegend=False),
                     go.Scatter(x=fpr, y=tpr, mode="markers+lines", text=thresholds, showlegend=False,
                                hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
                    layout=go.Layout(title=f"ROC curve of the fitted model, AUC={auc(fpr, tpr):.3f}",
                                     xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"))
    fig.write_image("roc_curve.png")

    # best alpha is the one with the biggest tpr minus fpr
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    model.alpha_ = best_alpha
    print(f"best alpha: {best_alpha:.3f}, test error: {model.loss(X_test, y_test):.3f}")

    # l1 regularization, lambda chosen with 5 fold cross validation
    lams = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    val_errors = []
    for lam in lams:
        solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
        model = LogisticRegression(solver=solver, penalty="l1", lam=lam, alpha=.5)
        train_err, val_err = cross_validate(model, X_train, y_train, misclassification_error)
        val_errors.append(val_err)
        print(f"lambda={lam}: train error {train_err:.3f}, validation error {val_err:.3f}")

    best_lam = lams[int(np.argmin(val_errors))]
    solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    model = LogisticRegression(solver=solver, penalty="l1", lam=best_lam, alpha=.5).fit(X_train, y_train)
    print(f"chosen lambda: {best_lam}, test error: {model.loss(X_test, y_test):.3f}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
