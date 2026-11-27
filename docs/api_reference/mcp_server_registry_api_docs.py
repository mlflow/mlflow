import json
from pathlib import Path

from fastapi import FastAPI

from mlflow.server.mcp_server_api import mcp_server_router

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
      href="../../_static/favicon.ico"
    />
    <title>MLflow MCP Server Registry - Swagger UI</title>
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
    app = FastAPI(title="MLflow MCP Server Registry")
    app.include_router(mcp_server_router, prefix="/api/3.0/mlflow/mcp-servers")

    docs_build = Path("build/html/genai/mcp-server-registry")
    docs_build.mkdir(parents=True, exist_ok=True)
    with docs_build.joinpath("openapi.json").open("w") as f:
        json.dump(app.openapi(), f)

    with docs_build.joinpath("api.html").open("w") as f:
        f.write(API_HTML)


if __name__ == "__main__":
    main()
