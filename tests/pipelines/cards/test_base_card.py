import uuid

import pytest

from mlflow.exceptions import MlflowException
from mlflow.pipelines.cards import BaseCard, CardTab


class ProfileReport:
    def to_html(self):
        return "pandas-profiling"


def test_verify_card_information():
    card = BaseCard(
        pipeline_name="fake pipeline",
        step_name="fake step",
    )
    (
        card.add_tab(
            "First tab",
            """
        <h3 class="section-title">Markdown:</h3>
        {{ MARKDOWN_1 }}<hr>
        <h3 class="section-title">Other:</h3>
        {{ HTML_1 }}""",
        )
        .add_markdown("MARKDOWN_1", "#### Hello, world!")
        .add_html("HTML_1", "<span style='color:blue'>blue</span>")
    )
    card.add_tab("Profile 1", "{{PROFILE}}").add_pandas_profile("PROFILE", ProfileReport())
    card.add_tab("Profile 2", "{{PROFILE}}").add_pandas_profile("PROFILE", ProfileReport())
    card.add_text("1,2,3.")

    expected_html = """
        <h3 class="section-title">Markdown:</h3>
        <h4>Hello, world!</h4><hr>
        <h3 class="section-title">Other:</h3>
        <span style='color:blue'>blue</span></div>
    """
    assert expected_html in card.to_html()
    assert card.to_text() == "1,2,3."
    assert all(card.get_tab(name) is not None for name in ["Profile 1", "Profile 2", "First tab"])
    assert all(card.get_tab(name) is None for name in ["", "!x", "fake tab 3"])


def test_card_tab_works():
    tab = (
        CardTab("tab", "{{MARKDOWN_1}}{{HTML_1}}{{PROFILE_1}}")
        .add_html("HTML_1", "<span style='color:blue'>blue</span>")
        .add_markdown("MARKDOWN_1", "#### Hello, world!")
        .add_pandas_profile("PROFILE_1", ProfileReport())
    )
    assert (
        tab.to_html()
        == "<h4>Hello, world!</h4><span style='color:blue'>blue</span>"
        + "<iframe srcdoc='pandas-profiling' width='100%' height='500' frameborder='0'></iframe>"
    )


def test_card_tab_fails_with_invalid_variable():
    with pytest.raises(MlflowException, match=r"(not a valid template variable)"):
        CardTab("tab", "{{MARKDOWN_1}}").add_html("HTML_1", "<span style='color:blue'>blue</span>")


def test_render_table():
    import pandas as pd
    import numpy as np

    assert "</table>" in BaseCard.render_table(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert "</table>" in BaseCard.render_table(pd.DataFrame({"a": [1, 2], "b": [3, 4]}).style)
    assert "</table>" in BaseCard.render_table([(1, 2), (3, 4)], columns=["a", "b"])
    assert "</table>" in BaseCard.render_table([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert "</table>" in BaseCard.render_table({"a": [1, 2], "b": [3, 4]})
    assert "</table>" in BaseCard.render_table(np.array([[1, 2], [3, 4]]), columns=["a", "b"])

    col_name = uuid.uuid4().hex
    assert col_name in BaseCard.render_table({"a": [1, 2], "b": [3, 4], col_name: [5, 6]})
    assert col_name not in BaseCard.render_table(
        {"a": [1, 2], "b": [3, 4], col_name: [5, 6]}, columns=["a", "b"]
    )
