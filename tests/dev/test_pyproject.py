import toml


def test_mcp_extra_includes_click_pin():
    """
    Test that the mcp extra dependencies include a pin to exclude click==8.3.0.
    This is necessary because click 8.3.0 causes the MLflow MCP server to fail.
    See https://github.com/mlflow/mlflow/issues/18747
    """
    with open("pyproject.toml") as f:
        pyproject = toml.load(f)

    mcp_deps = pyproject["project"]["optional-dependencies"]["mcp"]
    assert "click!=8.3.0" in mcp_deps, (
        "MCP extra dependencies should include 'click!=8.3.0' to avoid "
        "compatibility issues with click 8.3.0"
    )
