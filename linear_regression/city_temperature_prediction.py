
import pandas as pd
import numpy as np
import plotly.express as px
from polynomial_fitting import PolynomialFitting
from sklearn.model_selection import train_test_split


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename,
                     parse_dates=["Date"]).dropna().drop_duplicates()
    # drop impossible temps
    df = df[df.Temp > 0]
    # add day of year
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df["Year"] = df["Year"].astype(str)
    df["DM"] = df["Date"].dt.strftime('%d-%m')
    return df


if __name__ == '__main__':
    # load and preprocess the city temperature data
    df = load_data("city_temperature.csv")

    # Israel only
    israel = df[df.Country == 'Israel']
    # scatter DayOfYear vs Temp, color by Year (discrete)
    fig = px.scatter(
        israel,
        x='DayOfYear', y='Temp',
        color='Year', color_discrete_sequence=px.colors.qualitative.Bold,
        title="Israel Daily Temp by DayOfYear, colored by Year"
    )
    fig.write_image("israel_scatter.png")

    # monthly std dev bar
    month_std = israel.groupby('Month')['Temp'].std().reset_index()
    fig = px.bar(
        month_std,
        x='Month', y='Temp',
        title="Israel: Monthly Temperature Standard Deviation"
    )
    fig.write_image("israel_monthly_std.png")

    # all countries, monthly mean±std
    grp = df.groupby(['Country', 'Month'])['Temp']
    agg = grp.agg(['mean', 'std']).reset_index().rename(columns={'mean': 'Avg', 'std': 'Std'})
    fig = px.line(
        agg, x='Month', y='Avg', color='Country',
        error_y='Std',
        title="Average Monthly Temperature by Country ±1σ"
    )
    fig.write_image("countries_monthly.png")

    # polynomial fit for every k on Israel
    train, test = train_test_split(israel, test_size=0.25, random_state=0)
    X_train = train['DayOfYear'].values
    y_train = train['Temp'].values
    X_test = test['DayOfYear'].values
    y_test = test['Temp'].values

    errors = {}
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        err = model.loss(X_test, y_test)
        errors[k] = round(err, 2)
    # print and plot
    print("Test MSE per degree k:", errors)
    fig = px.bar(
        x=list(errors.keys()), y=list(errors.values()),
        labels={'x': 'Degree k', 'y': 'Test MSE'},
        title="Israel: Test MSE vs Polynomial Degree"
    )
    fig.write_image("israel_mse.png")

    best_k = min(errors, key=errors.get)
    print("Best k:", best_k)

    # train on all of Israel with best_k, then test on the other countries
    final_model = PolynomialFitting(best_k)
    final_model.fit(israel['DayOfYear'].values, israel['Temp'].values)

    other_err = {}
    for country, sub in df.groupby('Country'):
        if country == 'Israel': continue
        Xc = sub['DayOfYear'].values
        yc = sub['Temp'].values
        other_err[country] = round(final_model.loss(Xc, yc), 2)

    print("Errors on other countries:", other_err)
    fig = px.bar(
        x=list(other_err.keys()), y=list(other_err.values()),
        labels={'x': 'Country', 'y': 'Test MSE'},
        title=f"Model trained on Israel (k={best_k}): MSE on other countries"
    )
    fig.write_image("crosscountry_mse.png")

