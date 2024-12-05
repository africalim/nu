#! /usr/bin/python

import os
import click
from pathlib import Path
from scabha.schema_utils import clickify_parameters
from omegaconf import OmegaConf

recipe = os.path.join(os.path.dirname(__file__), "../cabs/hello.yml")

schemas = OmegaConf.load(recipe)

@click.command("hello") 
@clickify_parameters(schemas.cabs.get("hello"))
def main(**kw):
    hello(**kw)

def hello(name):
    greeting = f"Hello {name}!"
    click.echo(greeting)
    return greeting
