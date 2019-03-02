import click


class LoadGenerator(object):
    pass


@click.command()
def cli():
    click.echo("Hello")


if __name__ == '__main__':
    cli()
