#! /usr/bin/python

import click
from scabha.schema_utils import clickify_parameters
from omegaconf import OmegaConf

schemas = OmegaConf.load("nu/recipes/hello.yml")

@click.command("hello") 
@clickify_parameters(schemas.cabs.get("hello"))
def main(**kw):
    hello(**kw)

def hello(name):
    greeting = f"Hello {name}!"
    click.echo(greeting)
    return greeting
