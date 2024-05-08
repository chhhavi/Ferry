import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

def load_csv_into_dataframes():    

    # Load CSV into DataFrame
    file_path = '/Users/chhavi/Documents/Python/Ferry/data.csv'  
    df_ferry = pd.read_csv(file_path)

    # Load Weather data CSV into DataFrame
    file_path = '/Users/chhavi/Documents/Python/Ferry/weather.csv'  
    df_weather = pd.read_csv(file_path)

    return df_ferry, df_weather

def clean_data(df_ferry, df_weather):

    # Convert first column to python datetime
    df_ferry['Date'] = pd.to_datetime(df_ferry['Date'])

    # Extracting year from date and make new column
    df_ferry['Year'] = df_ferry['Date'].dt.year

    # Remove entries in the year 2017, 2020, 2021, 2022
    df_ferry = df_ferry[(df_ferry['Year'] == 2018) | (df_ferry['Year'] == 2019)]

    # Convert first column to python datetime
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

    # Extracting year from date and make new column
    df_weather['Year'] = df_weather['datetime'].dt.year

    # Remove entries in the year 2017, 2020, 2021, 2022
    df_weather = df_weather[(df_weather['Year'] == 2018) | (df_weather['Year'] == 2019)]

    # Add hour column to weather data
    df_weather['Hour'] = df_weather['datetime'].dt.hour

    # Add Date column to weather data
    df_weather["Date"] = pd.to_datetime(df_weather["datetime"].dt.date)

    # Remove columns from ferry data
    df_ferry = df_ferry.drop(columns=['Route', 'Direction', 'Stop', 'TypeDay'])

    # Remove columns from weather data
    df_weather = df_weather.drop(columns=[
    'name', 
    'datetime',
    'dew',
    'icon',
    'stations',
    'humidity',
    'precipprob',
    'preciptype',
    'snow',
    'solarenergy',
    'uvindex',
    'snowdepth',
    'windgust',
    'windspeed',
    'winddir',
    'sealevelpressure',
    'cloudcover',
    'visibility',
    'solarradiation',
    'severerisk',
    'conditions'])

    # Sum boardings at the same hour and date
    df_ferry = df_ferry.groupby(['Date', 'Hour'])['Boardings'].sum().reset_index()

    # Remove boardings that have the value 0
    df_ferry = df_ferry[df_ferry['Boardings'] != 0]

    # Merge dataframes
    merged_df = pd.merge(df_ferry, df_weather, on=['Date', 'Hour'])

    return merged_df

def regression_plot(df):
    X = df['temp'].values.reshape(-1, 1)
    y = df['Boardings'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    slope, intercept, r_value, p_value, std_err = stats.linregress(df['temp'], df['Boardings'])


    # Plot scatter plot
    plt.scatter(df['temp'], df['Boardings'], color='blue', label='Data', s=1)

    # Plot the regression line
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

    # Add labels and title
    plt.xlabel('Temperature')
    plt.ylabel('Boardings')
    plt.title('Temperature vs Boardings')

    summary_text = (
        f"Intercept: {model.intercept_}\n"
        f"Slope: {model.coef_[0]}\n"
        f"R-squared: {model.score(X, y)}\n"
        f"P-value: {p_value}\n"
    )
    plt.text(25, 130, summary_text, fontsize=10, verticalalignment='top')

    # Show plot
    plt.show()


def main():
    df_ferry, df_weather = load_csv_into_dataframes()
    df = clean_data(df_ferry, df_weather)

    regression_plot(df)

if __name__ == "__main__":
    main()