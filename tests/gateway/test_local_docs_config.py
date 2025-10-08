
import pytest
from fastapi.testclient import TestClient

from mlflow.gateway.app import create_app_from_config
from mlflow.gateway.config import GatewayConfig, SwaggerConfig

custom_cdn_config = {
    "swagger_js_url": "https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
    "swagger_css_url": "https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"
}

@pytest.fixture
def client(request) -> TestClient:
    config = GatewayConfig(
        **{
             "endpoints": [
                {
                    "name": "completions-gpt4",
                    "endpoint_type": "llm/v1/completions",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "mykey",
                            "openai_api_base": "https://api.openai.com/v1",
                            "openai_api_version": "2023-05-10",
                            "openai_api_type": "openai",
                        },
                    },
                },
                {
                    "name": "chat-gpt4",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "MY_API_KEY",
                        },
                    },
                },
            ],
            "swagger_config": request.param
        }
    )
    app = create_app_from_config(config)
    return TestClient(app)

@pytest.mark.parametrize("client", [custom_cdn_config], indirect=True)
def test_docs_custom_cdn(client: TestClient):
    response = client.get("/docs")
    html = response.content.decode()
    swagger_css_url = custom_cdn_config['swagger_css_url']
    swagger_js_url = custom_cdn_config['swagger_js_url']
    assert swagger_css_url in html 
    assert swagger_js_url in html
    assert f'<script src="{swagger_js_url}">' in html 
    assert f'<link type="text/css" rel="stylesheet" href="{swagger_css_url}">' in html
    assert response.status_code == 200

@pytest.mark.parametrize("client", [{"use_static_assets": True}], indirect=True)
def test_docs_static_assets(client: TestClient):
    response = client.get("/docs")
    html = response.content.decode()
    assert '<script src="/static/swagger-ui-bundle.js">' in html 
    assert '<link type="text/css" rel="stylesheet" href="/static/swagger-ui.css">' in html
    assert response.status_code == 200

@pytest.mark.parametrize("client", [None], indirect=True)
def test_docs_no_config(client: TestClient):
    response = client.get("/docs")
    assert response.status_code == 200

