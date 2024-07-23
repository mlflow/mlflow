from jinja2 import Template
from promptflow import tool


@tool
def render_template(template: str, **kwargs) -> str:
    return Template(template).render(**kwargs)
