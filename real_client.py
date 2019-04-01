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


@click.group()
def cli():
    pass


@cli.command()
@click.option('--pattern-row', default=0,
              help="Select the row of input data to use to generate load")
@click.option('--input-len', default=100,
              help="Length of input data")
@click.option('--predict-num', default=1,
              help="Number of predicted values")
@click.option('--ar', default=5,
              help="Autoregressive component of the ARIMA model")
@click.option('--i', default=1,
              help="Integrated component of the ARIMA model")
@click.option('--ma', default=0,
              help="Moving average component of the ARIMA model")
def run_predict(pattern_row, input_len, predict_num, ar, i, ma):
    series = load_input_data_series(pattern_row)
    predict(series, pattern_row, input_len, predict_num, ar, i, ma)


def predict(series, pattern_row, input_len, predict_num, ar, ir, ma):

    if input_len > series.size:
        input_len = series.size
        print("Input length changed to {}".format(input_len))

    start_time = time.time()
    generate_load(series, input_len)

    elapsed_time = math.ceil(time.time() - start_time)

    values = get_metric_values(elapsed_time)

    learning_values = values[:-predict_num]

    learning_series = pandas.Series(learning_values)
    # fit model
    model = ARIMA(np.asarray(learning_series), order=(ar, ir, ma))
    model_fit = model.fit(disp=0)

    print(model_fit.forecast(predict_num)[0])
    # round the predicted values to integers
    predicted_values = [round(pv) for pv in model_fit.forecast(predict_num)[0]]

    actual_values = values[-predict_num:]

    print("Predicted: {}, Actual: {}, SMAPE: {}"
          .format(predicted_values, actual_values, smape(predicted_values, actual_values)))

    return predicted_values, actual_values, smape(predicted_values, actual_values)


def generate_load(series, input_len):
    interval = 10  # 10 sec
    for i in range(input_len):
        try:
            request_num = int(series.iloc[i])
        except IndexError:
            print("{} is not a valid index.".format(i))
            break

        for k in range(request_num):
            requests.get(webapp_url)

        time.sleep(interval)


def get_metric_values(elapsed_time):

    average_query = "sum(sum_over_time(free_worker_threads[10s])) / sum(count_over_time(free_worker_threads[10s]))"

    metrics = requests.get("http://localhost:9090/api/v1/query?query=free_worker_threads[{}s]".format(elapsed_time))
    # metrics = requests.get("http://localhost:9090/api/v1/query?query=" + average_query)
    values = [int(record[1]) for record in metrics.json().get('data').get('result')[0].get('values')]

    return values


def get_predicted_values(series, input_len, predict_num, ar, ir, ma):

    if input_len > series.size:
        input_len = series.size
        print("Input length changed to {}".format(input_len))

    start_time = time.time()
    generate_load(series, input_len)
    elapsed_time = math.ceil(time.time() - start_time)

    metric_values = get_metric_values(elapsed_time)

    model = ARIMA(np.asarray(metric_values), order=(ar, ir, ma))  # this uses all the values to learn

    model_fit = model.fit(disp=0)

    # round the predicted values to integers
    predicted_values = [round(pv) for pv in model_fit.forecast(predict_num)[0]]

    print(predicted_values)
    return predicted_values

@cli.command()
@click.option('--pattern-row', default=0)
@click.option('--input-len', default=10)
@click.option('--predict-num', default=1)
@click.option('--ar', default=4)
@click.option('--ir', default=0)
@click.option('--ma', default=2)
# loop, that reads input_len data from wiki_data then predicts the next 'predict_num' values and stores it for the next iteration
def run_loop():
    series = load_input_data_series()




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

    def __str__(self):
        return "patt_row: {}, input_len: {}, predict_num: {}, || AR: {}, I: {}, MA: {}".format(
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
    series = load_input_data_series(pattern_row)

    test_results = list()

    input_len = 5
    predict_num = 5

    for ar in range(1, 6):
        for i in range(2):
            for ma in range(4):
                print("Test case started.")


                test_case = ARIMATestCase(pattern_row=pattern_row,
                                                  input_len=input_len,
                                                  predict_num=predict_num,
                                                  AR=ar,
                                                  I=i,
                                                  MA=ma)
                try:
                    print("pr: {}, il: {}, pn: {}, AR: {}, I: {}, MA: {}".format(
                                 pattern_row,
                                 input_len,
                                 predict_num,
                                 ar,
                                 i,
                                 ma
                             ))
                    res = predict(series,
                                  test_case.pattern_row,
                                  test_case.input_len,
                                  test_case.predict_num,
                                  test_case.AR,
                                  test_case.I,
                                  test_case.MA)

                    test_result = ARIMATestResult(
                        predicted_values=res[0],
                        actual_values=res[1],
                        smape=res[2]
                    )
                except Exception as e:
                    print(e)
                    print("Test case ended, waiting the webapp to recover.")
                    time.sleep(20)
                    continue

                test_results.append((test_result, test_case))
                # wait before starting the next round
                print("Test case ended, waiting the webapp to recover.")
                time.sleep(20)

    test_results.sort(key=lambda record: record[0].smape)
    print("CREATE/OPEN FILE")
    file = open("test_results" + "_" + str(input_len) + "_" + str(predict_num) + ".txt", "w")
    for result in test_results:
        file.write(str(result[0]) + ' || ' + str(result[1]) + '\n')

    file.close()
    print("FILE CLOSED")


if __name__ == '__main__':
    cli()


