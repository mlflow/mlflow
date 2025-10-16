# Custom Authentication

MLflow's authentication system is designed to be extensible. You can use custom authentication methods through plugins or pluggable functions.

### Using a Plugin

If your organization desires more advanced authentication logic
(e.g., token-based authentication), it is possible to install a third party plugin or to create your own plugin. For example, you can install the [mlflow-oidc-auth](https://github.com/mlflow-oidc/mlflow-oidc-auth) plugin to enable OIDC-based SSO.

Your plugin should be an installable Python package. It should include an app factory that extends the MLflow app and, optionally, implement a client to manage permissions.
The app factory function name will be passed to the `--app` argument in Flask CLI.
See https://flask.palletsprojects.com/en/latest/cli/#application-discovery for more information.

```python title="Example: my_auth/__init__.py"
from flask import Flask, Response, request
from werkzeug.datastructures import Authorization

from mlflow.server import app
from mlflow.server.handlers import catch_mlflow_exception


def authenticate_request_custom() -> Authorization | Response:
    """Custom auth logic for your organization."""
    ...


@catch_mlflow_exception
def _before_request():
    if request.path.startswith("/public"):
        return

    authorization = authenticate_request_custom()
    if isinstance(authorization, Response):
        return authorization

    # Perform additional authorization checks with the Authorization object as needed.


def create_app(app: Flask = app):
    app.add_url_rule("/api/custom-auth/login", view_func=..., methods=["POST"])
    app.before_request(_before_request)
    return app


class MyAuthClient:
    ...
```

Then, the plugin should be installed in your Python environment:

```bash
pip install my_auth
```

Then, register your plugin in `mlflow/setup.py`:

```python
setup(
    ...,
    entry_points="""
        ...

        [mlflow.app]
        my-auth=my_auth:create_app

        [mlflow.app.client]
        my-auth=my_auth:MyAuthClient
    """,
)
```

Then, you can start the MLflow server:

```bash
mlflow server --app-name my-auth
```

### Using a Function

You can configure the server to use a custom authentication function extending MLflow's authentication system.

First, install the auth extension:

```bash
pip install mlflow[auth]
```

Create a custom authentication function. The function should return a `werkzeug.datastructures.Authorization` object if
the request is authenticated, or a `Response` object (typically
`401: Unauthorized`) if the request is not authenticated. See [this example](https://github.com/mlflow/mlflow/blob/master/examples/jwt_auth/jwt_auth.py) for the reference implementation.

```python
# custom_auth.py
from werkzeug.datastructures import Authorization
from flask import Response


def custom_authenticate() -> Union[Authorization, Response]:
    # Your custom authentication logic
    # Return Authorization object if authenticated
    # Return Response object (401) if not authenticated
    pass
```

Then, update the auth configuration to use your custom function. The config file is located at `mlflow/server/auth/basic_auth.ini` by default. Alternatively, assign the environment variable `MLFLOW_AUTH_CONFIG_PATH` to point to your custom configuration file. Set the `authorization_function` setting with the value specifies `module_name:function_name`. The function has the following signature:

```ini
# /path/to/auth_config.ini
[mlflow]
authorization_function = mlflow.server.auth:custom_authenticate
```

Finally, start the MLflow server with the `--app-name` flag to enable authentication.

```bash
mlflow server --app-name basic-auth
```
