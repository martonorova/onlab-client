import click
import os
import requests
import numpy as np
import pandas
from pandas import read_csv
import time
import math
from statsmodels.tsa.arima_model import ARIMA


webapp_url = os.getenv("ONLAB_WEBAPP_URL", "http://localhost:8080/time")


def smape(predicted_list, actual_list):
    assert len(predicted_list) == len(actual_list)
    sum_val = 0
    for i in range(len(predicted_list)):
        nominator = math.fabs(predicted_list[i] - actual_list[i])
        denominator = (math.fabs(actual_list[i]) + math.fabs(predicted_list[i])) / 2
        sum_val += nominator / denominator

    return sum_val / len(predicted_list)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--pattern-row', default=0,
              help="Select the row of input data to use to generate load (train_1.csv")
@click.option('--input-len', default=100,
              help="Length of input data")
@click.option('--predict-num', default=1,
              help="Number of predicted values")
@click.option('--AR', default=5,
              help="Autoregressive component of the ARIMA model")
@click.option('--I', default=1,
              help="Integrated component of the ARIMA model")
@click.option('--MA', default=0,
              help="Moving average component of the ARIMA model")
def predict(pattern_row, input_len, predict_num, AR, I, MA):
    data_frame = read_csv('data/train_1.csv', header=0, index_col=0)
    series = data_frame.iloc[pattern_row]

    if input_len > series.size:
        input_len = series.size
        print("Input length changed to {}".format(input_len))

    interval = 10  # 10 sec
    start_time = time.time()
    for i in range(input_len):
        try:
            request_num = int(series.iloc[i])
        except IndexError:
            print("{} is not a valid index.".format(i))
            break

        time_to_wait = interval / request_num

        for k in range(request_num):
            requests.get(webapp_url)
            time.sleep(time_to_wait)

    elapsed_time = math.ceil(time.time() - start_time)

    metrics = requests.get("http://localhost:9090/api/v1/query?query=free_worker_threads[{}s]".format(elapsed_time))

    values = [int(record[1]) for record in metrics.json().get('data').get('result')[0].get('values')]

    learning_values = values[:-predict_num]

    learning_series = pandas.Series(learning_values)
    # fit model
    model = ARIMA(np.asarray(learning_series), order=(AR, I, MA))
    model_fit = model.fit(disp=0)

    print(model_fit.forecast(predict_num)[0])
    # round the predicted values to integers
    predicted_values = [round(pv) for pv in model_fit.forecast(predict_num)[0]]

    actual_values = values[-predict_num:]

    print("Predicted: {}, Actual: {}, SMAPE: {}"
          .format(predicted_values, actual_values, smape(predicted_values, actual_values)))

    return predicted_values, actual_values, smape(predicted_values, actual_values)




#############################################
# TEST
#############################################


class ARIMATestResult(object):
    def __init__(self, predicted_values, actual_values, smape):
        self.actual_values = actual_values
        self.predicted_values = predicted_values
        self.smape = smape

    def __str__(self):
        return "SMAPE: {}".format(self.smape)


class ARIMATestCase(object):
    def __init__(self, pattern_row, input_len, predict_num, AR, I, MA):
        self.AR = AR
        self.I = I
        self.MA = MA
        self.pattern_row = pattern_row
        self.input_len = input_len
        self.predict_num = predict_num

    def test(self):
        res = predict(self.pattern_row,
                      self.input_len,
                      self.predict_num,
                      self.AR,
                      self.I,
                      self.MA)

        return ARIMATestResult(predicted_values=res[0],
                               actual_values=res[1],
                               smape=res[2])

    def __str__(self):
        return "patt_row: {}, input_len: {}, predict_num: {}, AR: {}, I: {}, MA: {}".format(
            self.pattern_row,
            self.input_len,
            self.predict_num,
            self.AR,
            self.I,
            self.MA
        )


@cli.command()
def run_tests():
    pattern_row = 0
    data_frame = read_csv('data/train_1.csv', header=0, index_col=0)

   # series = data_frame.iloc[pattern_row]

    test_results = list()

    for ar in range(2):
        for i in range(2):
            for ma in range(2):
                input_len = 2
                predict_num = 2
                test_case = ARIMATestCase(pattern_row=pattern_row,
                                                  input_len=input_len,
                                                  predict_num=predict_num,
                                                  AR=ar,
                                                  I=i,
                                                  MA=ma)
                res = test_case.test()

                test_results.append((res, test_case))

    test_results.sort(key=lambda record: record[0].smape)
    file = open("test_results.txt", "w")
    for result in test_results:
        file.write(str(result))

    file.close()


if __name__ == '__main__':
    cli()