from flask import Flask

from mlflow.server.auth.oauth.login_page import render_login_page


def test_login_page_renders_provider_buttons():
    app = Flask(__name__)
    providers = [
        {"name": "google", "display_name": "Google", "type": "oidc"},
        {"name": "corporate", "display_name": "Corporate SSO", "type": "saml"},
    ]
    with app.test_request_context():
        html = render_login_page(providers, "/experiments")

        response_text = html if isinstance(html, str) else html.get_data(as_text=True)
        assert "Google" in response_text
        assert "Corporate SSO" in response_text
        assert "/auth/start/google" in response_text
        assert "/auth/start/corporate" in response_text


def test_login_page_includes_next_url():
    app = Flask(__name__)
    providers = [{"name": "test", "display_name": "Test", "type": "oidc"}]
    with app.test_request_context():
        html = render_login_page(providers, "/my-experiment")
        response_text = html if isinstance(html, str) else html.get_data(as_text=True)
        assert "/my-experiment" in response_text


def test_login_page_single_provider():
    app = Flask(__name__)
    providers = [{"name": "only", "display_name": "Only Provider", "type": "oidc"}]
    with app.test_request_context():
        html = render_login_page(providers, "/")
        response_text = html if isinstance(html, str) else html.get_data(as_text=True)
        assert "Only Provider" in response_text
        assert "/auth/start/only" in response_text


def test_login_page_mlflow_branding():
    app = Flask(__name__)
    providers = [{"name": "test", "display_name": "Test", "type": "oidc"}]
    with app.test_request_context():
        html = render_login_page(providers, "/")
        response_text = html if isinstance(html, str) else html.get_data(as_text=True)
        assert "MLflow" in response_text or "mlflow" in response_text.lower()
