#! /usr/bin/python

import click
from pathlib import Path
from scabha.schema_utils import clickify_parameters
from omegaconf import OmegaConf

recipe = Path(__file__).parent.parent / "recipes/hello.yml"

schemas = OmegaConf.load(recipe)

@click.command("hello") 
@clickify_parameters(schemas.cabs.get("hello"))
def main(**kw):
    print(recipe)
    hello(**kw)

def hello(name):
    greeting = f"Hello {name}!"
    click.echo(greeting)
    return greeting
