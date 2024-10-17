import json
import tempfile
from pathlib import Path

from mlflow.gateway.app import create_app_from_path

# This HTML was obtained by sending a request to the `/docs` route and saving the response.
# To hide the "try it out" button, we set `supportedSubmitMethods` to an empty list.
# The url was changed to "./openapi.json" from "/openapi.json" because `api.html` and `openapi.json`
# are served from the same directory.
API_HTML = """
<!DOCTYPE html>
<html>
  <head>
    <link
      type="text/css"
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
    />
    <link
      rel="shortcut icon"
      href="../_static/favicon.ico"
    />
    <title>MLflow AI Gateway - Swagger UI</title>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <!-- `SwaggerUIBundle` is now available on the page -->
    <script>
      const ui = SwaggerUIBundle({
        supportedSubmitMethods: [],
        url: "./openapi.json",
        dom_id: "#swagger-ui",
        layout: "BaseLayout",
        deepLinking: true,
        showExtensions: true,
        showCommonExtensions: true,
        oauth2RedirectUrl: window.location.origin + "/docs/oauth2-redirect",
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIBundle.SwaggerUIStandalonePreset,
        ],
      });
    </script>
  </body>
</html>
"""


def main():
    config = """
routes:
  - name: chat
    route_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-4o-mini
      config:
        openai_api_key: key

  - name: completions
    route_type: llm/v1/completions
    model:
      provider: openai
      name: gpt-4o-mini
      config:
        openai_api_key: key

  - name: embeddings
    route_type: llm/v1/embeddings
    model:
      provider: openai
      name: text-embedding-ada-002
      config:
        openai_api_key: key
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir).joinpath("config.yaml")
        config_path.write_text(config)

        app = create_app_from_path(config_path)
        docs_build = Path("build/html/llms/deployments")
        docs_build.mkdir(parents=True, exist_ok=True)
        with docs_build.joinpath("openapi.json").open("w") as f:
            json.dump(app.openapi(), f)

        with docs_build.joinpath("api.html").open("w") as f:
            f.write(API_HTML)


if __name__ == "__main__":
    main()
