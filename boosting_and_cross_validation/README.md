# Boosting and cross validation

## AdaBoost

* `decision_stump.py` a decision stump (a tree of depth 1) that also works with weighted samples
* `adaboost.py` AdaBoost over any weak learner, and predictions that use only the first T learners
* `adaboost_scenario.py` trains 250 stumps on points labeled by a circle, with no noise and with 40% label noise, and plots the errors, the decision boundaries and the sample weights

## Choosing lambda with cross validation

* `cross_validate.py` k fold cross validation
* `estimators.py` ridge regression (closed form, the intercept is not regularized), and lasso and least squares from scikit learn
* `perform_model_selection.py` takes 50 samples of the diabetes dataset, picks lambda for ridge and lasso with 5 fold cross validation, and compares them on the test set

`base_estimator.py`, `utils.py` and `loss_functions.py` are helper files.

## Results

* no noise: the lowest test error was with 238 learners, accuracy 0.996
* 40% noise: the lowest test error was with 33 learners, accuracy 0.666
* ridge: lambda 0.053, test MSE 3353.5
* lasso: lambda 0.318, test MSE 3504.7
* least squares with no regularization: test MSE 7991.3

The plots are in `plots/`.

## How to run

```
python adaboost_scenario.py
python perform_model_selection.py
```
