import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from mlflow.gateway.app import create_app_from_path


def main():
    config = """
routes:
  - name: chat
    route_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-3.5-turbo
      config:
        openai_api_base: https://api.openai.com/v1
        openai_api_key: key

  - name: completions
    route_type: llm/v1/completions
    model:
      provider: openai
      name: gpt-3.5-turbo
      config:
        openai_api_base: https://api.openai.com/v1
        openai_api_key: key

  - name: embeddings
    route_type: llm/v1/embeddings
    model:
      provider: openai
      name: text-embedding-ada-002
      config:
        openai_api_base: https://api.openai.com/v1
        openai_api_key: key
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir).joinpath("config.yaml")
        config_path.write_text(config)

        app = create_app_from_path(config_path)
        with TestClient(app) as client:
            docs_build = Path("build/html/gateway")
            docs_build.mkdir(parents=True, exist_ok=True)
            with docs_build.joinpath("openapi.json").open("w") as f:
                json.dump(app.openapi(), f)

            resp = client.get("/docs")
            resp.raise_for_status()
            with docs_build.joinpath("api.html").open("w") as f:
                f.write(resp.text.replace("'/openapi.json'", "'./openapi.json'"))


if __name__ == "__main__":
    main()
