# Gradient descent

## Gradient descent

* `gradient_descent.py` gradient descent that returns the last, the best or the average point
* `learning_rate.py` fixed and exponentially decaying learning rates
* `modules.py` the objectives: L1, L2, logistic regression and a regularized objective
* `gradient_descent_investigation.py` runs gradient descent on L1 and L2 with learning rates 1, 0.1, 0.01 and 0.001, and plots the descent paths and the convergence

## Logistic regression

* `logistic_regression.py` logistic regression trained with gradient descent, with no penalty, L1 or L2
* the same investigation script fits it on the South Africa heart disease data (`SAheart.data`), draws the ROC curve, picks the best threshold, and picks the L1 lambda with 5 fold cross validation

`base_estimator.py`, `base_module.py`, `base_learning_rate.py`, `utils.py`, `loss_functions.py` and `cross_validate.py` are helper files.

## Results

* best threshold alpha 0.325, test error 0.337
* L1 regularization: lambda 0.02 was chosen, test error 0.283

The plots are in `plots/`.

## How to run

```
python gradient_descent_investigation.py
```
