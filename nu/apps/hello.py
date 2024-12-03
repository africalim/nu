#! /usr/bin/python

import click
from scabha.schema_utils import clickify_parameters
from omegaconf import OmegaConf

schemas = OmegaConf.load("nu/recipes/hello.yml")

@click.command() 
@clickify_parameters(schemas.cabs.get("hello"))
def hello(**kw):
    # say hello
    opts = OmegaConf.create(kw)
    click.echo(f"Hello {opts.name}!")

def main():
    hello()
