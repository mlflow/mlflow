import os

from jinja2 import Environment, FileSystemLoader


def get_default_embedding_model() -> str:
    return "openai/text-embedding-3-small"


def load_distillation_template():
    template_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(template_dir))
    return env.get_template("distillation_guidelines.txt")
