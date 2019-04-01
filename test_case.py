import click
from pandas import read_csv
from . real_client import cli


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

    def test(self, series):
        res = cli(pattern_row=self.pattern_row,
                  input_len=self.input_len,
                  predict_num=self.predict_num,
                  AR=self.AR,
                  I=self.I,
                  MA=self.MA,
                  series=series)

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


@click.command()
def run_tests():
    pattern_row = 0
    data_frame = read_csv('data/train_1.csv', header=0, index_col=0)

    series = data_frame.iloc[pattern_row]

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
                res = test_case.test(series=series)

                test_results.append((res, test_case))

    test_results.sort(key=lambda record: record[0].smape)
    file = open("test_results_5_5.txt", "w")
    for result in test_results:
        file.write(str(result))

    file.close()


if __name__ == '__main__':
    run_tests()
