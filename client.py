import click
import os
import requests
import math
import time


webapp_url = os.getenv("ONLAB_WEBAPP_URL", "http://localhost:8080/time")


class Config(object):
    pass
    # def __init__(self):
    #     self.period_num = 3


pass_config = click.make_pass_decorator(Config, ensure=True)


class SineLoadGenerator(object):
    def __init__(self, interval_num=12, period_num=3):
        self.interval_num = interval_num
        self.period_num = period_num

    def generate_load(self):
        for period in range(self.period_num):
            for i in range(0, self.interval_num + 1):
                seconds_to_wait = 3 * (math.cos(2 * i * (math.pi / self.interval_num)) + 1)
                print(seconds_to_wait)
                requests.get(webapp_url)
                time.sleep(seconds_to_wait)


class SpikeLoadGenerator(object):
    def __init__(self, interval_num=20, period_num=3):
        self.interval_num = interval_num
        self.period_num = period_num

    def generate_load(self):
        for period in range(self.period_num):
            for i in range(self.interval_num):
                if i > 2 * self.interval_num / 4:
                    seconds_to_wait = 0.5
                    requests.get(webapp_url)
                else:
                    seconds_to_wait = 5
                # requests.get(webapp_url)
                # print(seconds_to_wait)
                time.sleep(seconds_to_wait)


@click.group()
@pass_config
def cli(config):
    pass


@cli.command()
@click.option('--period-num', default=3,
              help="How many times the pattern is repeated")
@pass_config
def sine(config, period_num):
    sine_generator = SineLoadGenerator(period_num=period_num)
    sine_generator.generate_load()


@cli.command()
@click.option('--period-num', default=3,
              help="How many times the pattern is repeated")
@pass_config
def spike(config, period_num):
    spike_generator = SpikeLoadGenerator(period_num=period_num)
    spike_generator.generate_load()


if __name__ == '__main__':
    cli()
