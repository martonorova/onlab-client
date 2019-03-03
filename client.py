import click
import os
import requests
import math
import time


webapp_url = os.getenv("ONLAB_WEBAPP_URL", "http://localhost:8080/time")


class Config(object):
    pass



class SineLoadGenerator(object):
    def __init__(self, interval_num=12, period_num=3):
        self.interval_num = interval_num
        self.period_num = period_num

    def generate_load(self):
        for period in range(self.period_num):
            for i in range(0, self.interval_num + 1):
                print('{}'.format(3 * (math.cos(2 * i * (math.pi / self.interval_num)) + 1)))
                seconds_to_wait = 3 * (math.cos(2 * i * (math.pi / self.interval_num)) + 1)
                requests.get(webapp_url)
                time.sleep(seconds_to_wait)


@click.group()
def cli():
    pass


@cli.command()
def sine():
    sine_generator = SineLoadGenerator()
    sine_generator.generate_load()


if __name__ == '__main__':
    cli()
