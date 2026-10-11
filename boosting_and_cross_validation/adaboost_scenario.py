import numpy as np
from typing import Tuple
from utils import decision_surface, custom
from adaboost import AdaBoost
from decision_stump import DecisionStump
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def surface(predict, lims):
    # fixed color range so a constant prediction still gets the right color
    return decision_surface(predict, lims[0], lims[1], showscale=False).update(zmin=-1, zmax=1)


def points(X, y, size=6):
    # scatter of the samples, color and shape by label
    return go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                      marker=dict(color=(y == 1).astype(int), symbol=np.where(y == 1, "circle", "x"), size=size,
                                  colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # train and test errors as a function of the number of learners
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    ts = np.arange(1, n_learners + 1)
    train_err = [model.partial_loss(train_X, train_y, t) for t in ts]
    test_err = [model.partial_loss(test_X, test_y, t) for t in ts]

    fig = go.Figure([go.Scatter(x=ts, y=train_err, mode="lines", name="train error"),
                     go.Scatter(x=ts, y=test_err, mode="lines", name="test error")],
                    layout=go.Layout(title=f"AdaBoost error as a function of the number of learners, noise={noise}",
                                     xaxis_title="number of learners", yaxis_title="misclassification error"))
    fig.write_image(f"errors_noise_{noise}.png")

    # decision boundaries after 5, 50, 100 and 250 learners
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} learners" for t in T],
                        horizontal_spacing=.05, vertical_spacing=.08)
    for i, t in enumerate(T):
        fig.add_traces([surface(lambda X, t=t: model.partial_predict(X, t), lims),
                        points(test_X, test_y)],
                       rows=i // 2 + 1, cols=i % 2 + 1)
    fig.update_layout(title=f"Decision boundary by number of learners, noise={noise}", width=900, height=900,
                      margin=dict(t=100))
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"boundaries_noise_{noise}.png")

    # the ensemble size with the lowest test error
    best_t = int(np.argmin(test_err)) + 1
    acc = 1 - test_err[best_t - 1]
    fig = go.Figure([surface(lambda X: model.partial_predict(X, best_t), lims),
                     points(test_X, test_y)],
                    layout=go.Layout(title=f"Best ensemble size: {best_t}, accuracy: {acc:.3f}, noise={noise}",
                                     width=700, height=700))
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"best_ensemble_noise_{noise}.png")
    print(f"noise={noise}: best ensemble size {best_t}, test accuracy {acc:.3f}")

    # train set with point size by the last weights D
    D = model.D_ / np.max(model.D_) * 5
    fig = go.Figure([surface(model.predict, lims),
                     points(train_X, train_y, size=D)],
                    layout=go.Layout(title=f"Train set, point size by the final sample weights, noise={noise}",
                                     width=700, height=700))
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(f"weights_noise_{noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    # same with noise
    fit_and_evaluate_adaboost(0.4)
