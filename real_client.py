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


@click.command()
@click.option('--pattern-row', default=0,
              help="Select the row of input data to use to generate load (train_1.csv")
@click.option('--input-len', default=100,
              help="Length of input data")
@click.option('--predict-num', default=1,
              help="Number of predicted values")
def cli(pattern_row, input_len, predict_num):
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
    model = ARIMA(np.asarray(learning_series), order=(5, 0, 0))
    model_fit = model.fit(disp=0)
    predicted_values = model_fit.forecast(predict_num)[0]

    print("Predicted: {}, Real: {}".format(predicted_values, values[-predict_num:]))


if __name__ == '__main__':
    cli()
