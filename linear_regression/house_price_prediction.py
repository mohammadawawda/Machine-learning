
import os
import random

import pandas as pd
import numpy as np
from typing import NoReturn
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import plotly.graph_objects as go


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    # Remove samples with missing target values
    preprocess_data = pd.concat([X, y], axis=1)  # Temporarily merge to handle rows consistently
    preprocess_data.dropna(subset=[y.name], inplace=True)  # y.name gives the label column name

    # preprocess the columns
    preprocess_data = preprocess_columns(preprocess_data)

    # removing unrealistic entries
    if 'bedrooms' in X.columns:
        preprocess_data = preprocess_data[preprocess_data['bedrooms'] > 0]  # there cannot be apartment with 0 bedrooms

    if 'bathrooms' in X.columns:
        preprocess_data = preprocess_data[
            preprocess_data['bathrooms'] > 0]  # there cannot be apartment with 0 bathrooms

    # remove the entries where the price is negative
    preprocess_data = preprocess_data[preprocess_data[y.name] > 0]

    # Re-separate the features and response vector after preprocessing
    X_clean = preprocess_data.drop([y.name], axis=1)
    y_clean = preprocess_data[y.name]

    return X_clean, y_clean

def preprocess_columns(preprocess_data : pd.DataFrame):
    """

    :param preprocess_data: the data which is its columns are to be preprocessed
    :return: preprocessed data such that the
    """
    # drop duplicate entries
    preprocess_data = preprocess_data.dropna().drop_duplicates()
    # drop the column date which is not relevant to the house sales
    preprocess_data = preprocess_data.drop('date', axis=1)  # axis=1 specifies that we are dropping a column
    # drop the column lat which is not relevant to the house sales
    preprocess_data = preprocess_data.drop('lat', axis=1)  # axis=1 specifies that we are dropping a column
    # drop the column long which is not relevant to the house sales
    preprocess_data = preprocess_data.drop('long', axis=1)  # axis=1 specifies that we are dropping a column
    # drop the column id which is not relevant to the house sales
    preprocess_data = preprocess_data.drop('id', axis=1)  # axis=1 specifies that we are dropping a column
    preprocess_data = preprocess_data.drop(["sqft_lot15", "sqft_living15"], axis=1)
    # Feature: Handling years and renovations , the value 0 in column YearRenovated is replaced with
    # the value in the column YearBuilt
    if 'YearBuilt' in preprocess_data.columns and 'YearRenovated' in preprocess_data.columns:
        # year 0 means not renovated
        preprocess_data['YearRenovated'] = preprocess_data['YearRenovated'].replace(0, np.nan)
        # Fill missing renovation years with the year built
        preprocess_data['YearRenovated'].fillna(preprocess_data['YearBuilt'], inplace=True)
    # there cannot be number of bathrooms of type double
    preprocess_data["bathrooms"] = preprocess_data["bathrooms"].astype(int)
    preprocess_data["waterfront"] = preprocess_data["waterfront"].astype(int)
    preprocess_data["view"] = preprocess_data["view"].astype(int)
    preprocess_data["condition"] = preprocess_data["condition"].astype(int)
    preprocess_data["grade"] = preprocess_data["grade"].astype(int)
    return preprocess_data


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    preprocess_test = preprocess_columns(X)

    # fill the missing value with the median ,since we cannot delete an entry
    for column in preprocess_test.select_dtypes(include=[np.number]).columns:
        if preprocess_test[column].isnull().any():
            median_val = preprocess_test[column].median()
            preprocess_test[column].fillna(median_val, inplace=True)

    return preprocess_test


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)  # Ensure the output directory exists

    for feature in X.columns:
        if feature == 'id':
            continue
        # Extract the feature data
        feature_data = X[feature]

        # we stack the vectors so now we have matrix where the first column is the feature vectore and
        # the second column is the resonse vector y
        data = np.column_stack((feature_data, y))
        # Compute the covariance matrix using rowvar=False to treat columns as separate variables
        covariance_matrix = np.cov(data, rowvar=False)
        # extract covariance between X and y which is in entries (0,1) and (1,0)
        covariance = covariance_matrix[0, 1]

        # std_x = np.std(feature_data, ddof=1)
        # std_y = np.std(y, ddof=1)
        # Compute the standard deviations of X and y
        std_x = np.std(feature_data)
        std_y = np.std(y)

        # Calculate Pearson correlation coefficient
        pearson_correlation = covariance / (std_x * std_y)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(feature_data, y, alpha=0.5)
        plt.title(f'{feature} vs. Price\nPearson Correlation: {pearson_correlation:.2f}')
        plt.xlabel(feature)
        plt.ylabel('Price')

        # Save the plot
        plot_filename = output_path + str(feature) + "_vs_Price.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the figure after saving to free up memory

def perform_experiment(X_clean_train, y_clean_train, X_processed_test, y_processed_test, percentages):
    results = np.zeros((len(percentages), 10))
    for i, percentage in enumerate(percentages):
        for j in range(10):
            # Set seed for reproducibility
            np.random.seed(39)

            # Shuffle indices
            indices = np.arange(len(X_clean_train))
            np.random.shuffle(indices)

            # Calculate the split point for the sample size
            sample_size = int((percentage / 100.0) * len(indices))

            # Create training sample
            train_indices = indices[:sample_size]
            X_sample = X_clean_train.iloc[train_indices]
            y_sample = y_clean_train.iloc[train_indices]
            # Fit model and calculate loss
            model = LinearRegression(True)
            model.fit(X_sample, y_sample)
            results[i, j] = model.loss(X_processed_test, y_processed_test)

    return results

if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # split train test

    # Set the random seed (a random number was chosen as seed)
    np.random.seed(39)

    # Shuffle indices
    indices = np.arange(df.shape[0])
    np.random.shuffle(indices)

    # Calculate the split point for 75% training data
    split_idx = int(0.75 * len(indices))

    # Create training and test sets
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # we produce the test and train sets while maintining the order of the rows in the shuffled version using iloc
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]


    # preprocessing of the train set
    X_clean_train, y_clean_train = preprocess_train(X_train, y_train)


    # how every feature relates to the price
    feature_evaluation(X_clean_train, y_clean_train, output_path="feature_plots\\")

    # preprocess the test data
    test_set_to_process = pd.concat([X_test, y_test], axis=1)
    test_set_processed = preprocess_test(test_set_to_process)
    X_processed = test_set_processed.drop(columns=['price'])
    y_processed = test_set_processed['price']

    # fit the model over increasing percentages of the training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percentages = list(range(10, 101))
    results = perform_experiment(X_clean_train, y_clean_train, X_processed, y_test, percentages)
    results = np.array(results)
    m = results.mean(axis=1)
    s = results.std(axis=1)
    fig = go.Figure([go.Scatter(x=percentages, y=m - 2 * s, fill=None, mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=percentages, y=m + 2 * s, fill='tonexty', mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=percentages, y=m, mode="markers+lines", marker=dict(color="black"))],
                    layout=go.Layout(title="Test MSE as Function Of Training Size",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set"),
                                     showlegend=False))
    fig.write_image("mse.over.training.percentage.png")

# ----------------------------------------------------------------------------------------------------
