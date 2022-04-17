from jinja2 import Template

DOCKERFILE_TEMPLATE = Template(
    """
FROM {{ base_image_name }}

ENV VENV_DIR=/mlflow/.venv
RUN virtualenv --python=$(which python{{ python_version }}) $VENV_DIR

COPY requirements /tmp/requirements
RUN source $VENV_DIR/bin/activate && pip install -r /tmp/requirements/{{ small_requirements_file }}

COPY {{ mlflow_requirements_file }} /tmp/requirements/{{ mlflow_requirements_file }}
RUN source $VENV_DIR/bin/activate && pip install -r /tmp/requirements/{{ mlflow_requirements_file }}

COPY {{ install_script }} /tmp/{{ install_script }}
RUN source $VENV_DIR/bin/activate && /tmp/{{ install_script }}

COPY {{ run_script }} /tmp/{{ run_script }}
COPY {{ entrypoint_script }} /tmp/{{ entrypoint_script }}
""".lstrip()
)

DOCKER_COMPOSE_TEMPLATE = Template(
    """
services:
{% for job in jobs %}
  {{ job }}:
    image: {{ job }}
    build:
      context: {{ job }}
    volumes:
      - ${PWD}:/mlflow/home
    working_dir: /mlflow/home
    entrypoint: /tmp/entrypoint.sh
    command: /tmp/run.sh
{% endfor %}
""".lstrip()
)

SHELL_SCRIPT_TEMPLATE = Template(
    """
#!/bin/bash
set -exo pipefail

{{ cmd }}
""".lstrip()
)
