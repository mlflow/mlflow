import os

from flask import Flask, send_from_directory

from mlflow.server import handlers
from mlflow.utils.process import exec_cmd

FILE_STORE_ENV_VAR = "MLFLOW_SERVER_FILE_STORE"
ARTIFACT_ROOT_ENV_VAR = "MLFLOW_SERVER_ARTIFACT_ROOT"

REL_STATIC_DIR = "js/build"
app = Flask(__name__, static_folder=REL_STATIC_DIR)
STATIC_DIR = os.path.join(app.root_path, REL_STATIC_DIR)

for http_path, handler, methods in handlers.get_endpoints():
    app.add_url_rule(http_path, handler.__name__, handler, methods=methods)


# Serve the font awesome fonts for the React app
@app.route('/webfonts/<path:path>')
def serve_webfonts(path):
    return send_from_directory(STATIC_DIR, os.path.join('webfonts', path))


# We expect the react app to be built assuming it is hosted at /static-files, so that requests for
# CSS/JS resources will be made to e.g. /static-files/main.css and we can handle them here.
@app.route('/static-files/<path:path>')
def serve_static_file(path):
    return send_from_directory(STATIC_DIR, path)


# Serve the index.html for the React App for all other routes.
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):  # pylint: disable=unused-argument
    return send_from_directory(STATIC_DIR, 'index.html')


def _run_server(file_store_path, artifact_root, host, port, workers):
    """Run the MLflow server, wrapping it in gunicorn"""
    env_map = {}
    if file_store_path:
        env_map[FILE_STORE_ENV_VAR] = file_store_path
    if artifact_root:
        env_map[ARTIFACT_ROOT_ENV_VAR] = artifact_root
    bind_address = "%s:%s" % (host, port)
    exec_cmd(["gunicorn", "-b", bind_address, "-w", "%s" % workers, "mlflow.server:app"],
             env=env_map, stream_output=True)
