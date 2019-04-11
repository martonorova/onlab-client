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


webapp_url = os.getenv("ONLAB_WEBAPP_URL", "http://localhost:8080/time")


class Controller(object):

    predicted_values = list()  # holds the predicted values for the next iteration to calculate error
    predictions_num = 0  # how many times did the prediction has run so far
    average_smape = 0
    smape_samples_num = 0

    predicted_values_to_plot = list()
    actual_values_to_plot = list()
    should_draw_plot = False

    def __init__(self, predict_model, should_draw_plot):
        self.predict_model = predict_model
        self.should_draw_plot = should_draw_plot


    def calculate_avg_smape(self, new_smape):
        if not math.isnan(new_smape):
            new_avg_smape = (self.average_smape * self.smape_samples_num + new_smape) / (self.smape_samples_num + 1)
            self.smape_samples_num += 1
            self.average_smape = new_avg_smape

    def save_values_to_plot(self, predicted_list, actual_list):
        assert len(predicted_list) == len(actual_list)
        length = len(predicted_list)

        for i in range(length):
            # next_pred_value = predicted_list[i]
            # if math.isnan(next_pred_value):
            #     self.predicted_values_to_plot.append()
            self.predicted_values_to_plot.append(predicted_list[i])
            self.actual_values_to_plot.append(actual_list[i])

    def draw_plot(self):
        plt.plot(self.predicted_values_to_plot, 'r')
        plt.plot(self.actual_values_to_plot, 'b')
        plt.draw()
        plt.pause(1)
        plt.clf()

    def start(self, predict_num, learning_interval):
        pattern_row = 0
        series = load_input_data_series(pattern_row)

        # generate load
        interval = 10  # 10 seconds
        for i in range(series.size):

            request_num = int(series.iloc[i])

            for j in range(request_num):
                requests.get(webapp_url)

            if i >= int(learning_interval / interval):
                predict_start = time.time()
                print("PREDICT START: {}".format(time.ctime()))

                metrics = requests.get(
                    "http://localhost:9090/api/v1/query?query=free_worker_threads[{}s]".format(learning_interval))
                metric_values = [int(record[1]) for record in metrics.json().get('data').get('result')[0].get('values')]
                print("Metric values: {} at {}".format(metric_values, time.ctime()))

                # check error of last prediction
                if len(self.predicted_values) != 0:
                    previous_predicted_values = self.predicted_values
                    actual_values = metric_values[-len(previous_predicted_values):]

                    assert len(previous_predicted_values) == len(actual_values)

                    smape_val = smape(previous_predicted_values, actual_values)

                    self.save_values_to_plot(previous_predicted_values, actual_values)

                    if self.should_draw_plot:
                        self.draw_plot()

                    self.calculate_avg_smape(smape_val)

                    print("Prev predicted: {}, Actual: {} || SMAPE: {}, AVG_SMAPE: {}".format(
                        previous_predicted_values,
                        actual_values,
                        smape_val,
                        self.average_smape
                    ))

                try:
                    # model = ARIMA(np.asarray(metric_values), order=(ar, ir, ma))
                    # model_fit = model.fit(disp=0)
                    # # round the predicted values to integers
                    # self.predicted_values = [round(pv) for pv in model_fit.forecast(predict_num)[0]]
                    self.predicted_values = self.predict_model.forecast(metric_values, predict_num)
                except Exception as e:
                    print(e)
                    predict_end = time.time()

                    time_to_wait = interval - (predict_end - predict_start)
                    print("TIME TO WAIT EXC: {}s".format(time_to_wait))
                    time.sleep(time_to_wait)
                    continue


                print("Predicted: {} at {}".format(self.predicted_values, time.ctime()))
                predict_end = time.time()

                time_to_wait = interval - (predict_end - predict_start)
                print("TIME TO WAIT: {}s".format(time_to_wait))
                time.sleep(time_to_wait)
            else:
                time.sleep(interval)


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


class ARIMAPredict(object):
    input_values = list()

    def __init__(self, ar, ir, ma):
        self.ar = ar
        self.ir = ir
        self.ma = ma

    def forecast(self, input_values, predict_num):
        model = ARIMA(np.asarray(input_values), order=(self.ar, self.ir, self.ma))
        model_fit = model.fit(disp=0)

        # round the predicted values to integers
        predicted_values = [round(pv) for pv in model_fit.forecast(predict_num)[0]]

        return predicted_values


class MovingAveragePredict(object):
    input_values = list()
    # window

    def forecast(self, input_values, predict_num):
        input_series = pandas.Series(data=input_values)

        moving_avg = input_series.rolling(window=10).mean()

        predicted_values = [moving_avg for i in range(predict_num)]


# @click.group()
# def cli():
#     pass
#
#
# @cli.command()
# @click.option('--pattern-row', default=0,
#               help="Select the row of input data to use to generate load")
# @click.option('--input-len', default=100,
#               help="Length of input data")
# @click.option('--predict-num', default=1,
#               help="Number of predicted values")
# @click.option('--ar', default=5,
#               help="Autoregressive component of the ARIMA model")
# @click.option('--i', default=1,
#               help="Integrated component of the ARIMA model")
# @click.option('--ma', default=0,
#               help="Moving average component of the ARIMA model")
# def run_predict(pattern_row, input_len, predict_num, ar, i, ma):
#     series = load_input_data_series(pattern_row)
#     predict(series, pattern_row, input_len, predict_num, ar, i, ma)
#
#
# def predict(series, pattern_row, input_len, predict_num, ar, ir, ma):
#
#     if input_len > series.size:
#         input_len = series.size
#         print("Input length changed to {}".format(input_len))
#
#     start_time = time.time()
#     generate_load(series, input_len)
#
#     elapsed_time = math.ceil(time.time() - start_time)
#
#     values = get_metric_values(elapsed_time)
#
#     learning_values = values[:-predict_num]
#
#     learning_series = pandas.Series(learning_values)
#     # fit model
#     model = ARIMA(np.asarray(learning_series), order=(ar, ir, ma))
#     model_fit = model.fit(disp=0)
#
#     print(model_fit.forecast(predict_num)[0])
#     # round the predicted values to integers
#     predicted_values = [round(pv) for pv in model_fit.forecast(predict_num)[0]]
#
#     actual_values = values[-predict_num:]
#
#     print("Predicted: {}, Actual: {}, SMAPE: {}"
#           .format(predicted_values, actual_values, smape(predicted_values, actual_values)))
#
#     return predicted_values, actual_values, smape(predicted_values, actual_values)
#
#
# def generate_load(series, input_len):
#     interval = 10  # 10 sec
#     for i in range(input_len):
#         try:
#             request_num = int(series.iloc[i])
#         except IndexError:
#             print("{} is not a valid index.".format(i))
#             break
#
#         for k in range(request_num):
#             requests.get(webapp_url)
#
#         time.sleep(interval)
#
#
# def get_metric_values(elapsed_time):
#
#     average_query = "sum(sum_over_time(free_worker_threads[10s])) / sum(count_over_time(free_worker_threads[10s]))"
#
#     metrics = requests.get("http://localhost:9090/api/v1/query?query=free_worker_threads[{}s]".format(elapsed_time))
#     # metrics = requests.get("http://localhost:9090/api/v1/query?query=" + average_query)
#     values = [int(record[1]) for record in metrics.json().get('data').get('result')[0].get('values')]
#
#     return values
#
#
# def get_predicted_values(series, input_len, predict_num, ar, ir, ma):
#
#     if input_len > series.size:
#         input_len = series.size
#         print("Input length changed to {}".format(input_len))
#
#     start_time = time.time()
#     generate_load(series, input_len)
#     elapsed_time = math.ceil(time.time() - start_time)
#
#     metric_values = get_metric_values(elapsed_time)
#
#     model = ARIMA(np.asarray(metric_values), order=(ar, ir, ma))  # this uses all the values to learn
#
#     model_fit = model.fit(disp=0)
#
#     # round the predicted values to integers
#     predicted_values = [round(pv) for pv in model_fit.forecast(predict_num)[0]]
#
#     print(predicted_values)
#     return predicted_values


if __name__ == '__main__':
    #cli()
    print(time.ctime())
    Controller(predict_model=ARIMAPredict(ar=4, ir=0, ma=2),
               should_draw_plot=True).start(10, 100)
