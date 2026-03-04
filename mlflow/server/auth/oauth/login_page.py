from flask import render_template_string

from mlflow.server.auth.logo import MLFLOW_LOGO

LOGIN_PAGE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign in to MLflow</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .login-container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            padding: 48px 40px;
            width: 100%;
            max-width: 400px;
            text-align: center;
        }
        .logo-container {
            margin-bottom: 24px;
        }
        .logo-container svg {
            width: 120px;
            height: auto;
        }
        h1 {
            font-size: 20px;
            font-weight: 600;
            color: #333;
            margin-bottom: 32px;
        }
        .provider-btn {
            display: block;
            width: 100%;
            padding: 12px 16px;
            margin-bottom: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            color: #333;
            font-size: 14px;
            font-weight: 500;
            text-decoration: none;
            cursor: pointer;
            transition: background 0.15s, border-color 0.15s;
            text-align: center;
        }
        .provider-btn:hover {
            background: #f8f9fa;
            border-color: #bbb;
        }
        .provider-btn:active {
            background: #e9ecef;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo-container">
            {{ mlflow_logo|safe }}
        </div>
        <h1>Sign in to MLflow</h1>
        {% for provider in providers %}
        <a class="provider-btn"
           href="/auth/start/{{ provider.name }}?next={{ next_url|urlencode }}">
            {{ provider.display_name }}
        </a>
        {% endfor %}
    </div>
</body>
</html>
"""


def render_login_page(providers: list[dict[str, str]], next_url: str = "/") -> str:
    return render_template_string(
        LOGIN_PAGE_TEMPLATE,
        mlflow_logo=MLFLOW_LOGO,
        providers=providers,
        next_url=next_url,
    )
