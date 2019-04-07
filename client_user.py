import click
import os
import requests
import numpy as np
import pandas
from pandas import read_csv
import time
import math
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

###############################
#  This predicts the users
###############################

webapp_url = os.getenv("ONLAB_WEBAPP_URL", "http://localhost:8080/time")


def load_input_data_series(pattern_row):
    data_frame = read_csv('data/train_1_row_1.csv', header=0, index_col=0)
    return data_frame.iloc[pattern_row]


def smape(predicted_list, actual_list):
    assert len(predicted_list) == len(actual_list)
    sum_val = 0
    for i in range(len(predicted_list)):
        nominator = math.fabs(predicted_list[i] - actual_list[i])
        denominator = (math.fabs(actual_list[i]) + math.fabs(predicted_list[i])) / 2
        sum_val += nominator / denominator

    return sum_val / len(predicted_list)


class ControllerUserPredict(object):

    prediction_type = 'arima'
    ar = 8
    ir = 0
    ma = 0

    def __init__(self, prediction_type):
        self.prediction_type = prediction_type

    def start(self, learning_window):
        pattern_row = 0
        series = load_input_data_series(pattern_row)

        interval = 10  # 10 sec

        for i in range(learning_window, series.size):
            iteration_start = time.time()
            request_num = int(series.iloc[i])
            print("REQUEST NUM: {}".format(request_num))

            for j in range(request_num):
                requests.get(webapp_url)

            # get predicted value
            print(series[i - learning_window: i])
            predicted_val = self.predict(self.prediction_type, series[i - learning_window: i], learning_window)

            if predicted_val > 20:
                print("####### SCALE #######")

            if i != series.size - 1:
                actual_val = int(series.iloc[i])
                print("PREDICTED: {}, ACTUAL: {}".format(
                    predicted_val,
                    actual_val
                ))

            iteration_end = time.time()
            time_to_sleep = interval - (iteration_end - iteration_start)
            print("TIME TO SLEEP: " + str(time_to_sleep))
            time.sleep(time_to_sleep)

    def ema(self, series, learning_window, start_from):
        pass

    def arima(self, series, learning_window):

        try:
            model = ARIMA(np.asarray(series), order=(self.ar, self.ir, self.ma))
            model_fit = model.fit(disp=0)
            predicted_value = round(model_fit.forecast(1)[0][0])
        except Exception as e:
            print(e)
            predicted_value = math.nan

        # round the predicted values to integers

        return predicted_value

    def predict(self, method, series, learning_window):
        prediction_methods = {
            'arima': self.arima,
            'ema': self.ema
        }

        return prediction_methods.get(method)(series, learning_window)


if __name__ == '__main__':

    ControllerUserPredict('arima').start(
        100
    )
