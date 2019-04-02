import click
import os
import requests
import numpy as np
import pandas
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
import time
import math
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


@click.command()
@click.option('--pattern-row', default=0,
              help="Select the row of input data to use to generate load (train_1.csv")
@click.option('--input-len', default=100,
              help="Length of input data")
@click.option('--predict-num', default=1,
              help="Number of predicted values")
def cli(pattern_row, input_len, predict_num):
    data_frame = read_csv('data/train_1_row_1.csv', header=0, index_col=0)
    series = data_frame.iloc[pattern_row]
    # print(series)

    actual_values = list()
    for i in range(200):
        actual_values.append(series[i])

    learning_values = list()
    for j in range(180):
        learning_values.append(series[j])

    # model = ARIMA(np.asarray(learning_values), order=(10, 0, 5))
    # model_fit = model.fit(disp=0)
    #
    # # round the predicted values to integers
    # predicted_values = [round(pv) for pv in model_fit.forecast(20)[0]]
    # pred = pandas.Series(data=learning_values+predicted_values)
    # pred.plot()

    plt.plot(np.convolve(actual_values, np.ones((10,))/10, mode='valid'))

    actual_series = pandas.Series(data=actual_values[10:])
    actual_series.plot()

    # autocorrelation_plot(series)
    plt.show()

# User.Google from line 10763 in train.csv





if __name__ == '__main__':
    cli()

