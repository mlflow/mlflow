from click.testing import CliRunner


def invoke_cli_runner(*args, **kwargs):
    """
    Helper method to invoke the CliRunner while asserting that the exit code is actually 0.
    """

    res = CliRunner().invoke(*args, **kwargs)
    assert res.exit_code == 0, f"Got non-zero exit code {res.exit_code}. Output is: {res.output}"
    return res
