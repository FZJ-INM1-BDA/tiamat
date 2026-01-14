"""
Command line interface for tiamat.
"""

import click


@click.group()
def cli():
    """Command line interface for tiamat."""
    pass


if __name__ == "__main__":
    cli()
