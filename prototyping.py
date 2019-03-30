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
    data_frame = read_csv('data/train_1.csv', header=0, index_col=0)
    series = data_frame.iloc[pattern_row+1]
    print(series)

    # autocorrelation_plot(series)
    series.plot()
    plt.show()

# User.Google from line 10763 in train.csv


if __name__ == '__main__':
    cli()
