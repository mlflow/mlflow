from __future__ import annotations

import base64
import html
import logging
import os
import pathlib
import pickle
import random
import re
import string
from io import StringIO
from typing import Optional, Union

from packaging.version import Version

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException

CARD_PICKLE_NAME = "card.pkl"
CARD_HTML_NAME = "card.html"

_PP_VARIABLE_LINK_REGEX = re.compile(r'<a\s+href="?(?P<href>#pp_var_[-0-9]+)"?\s*>')


_logger = logging.getLogger(__name__)


class CardTab:
    def __init__(self, name: str, template: str) -> None:
        """
        Construct a step card tab with supported HTML template.

        Args:
            name: A string representing the name of the tab.
            template: A string representing the HTML template for the card content.
        """
        import jinja2
        from jinja2 import meta as jinja2_meta

        self.name = name
        self.template = template

        j2_env = jinja2.Environment()
        self._variables = jinja2_meta.find_undeclared_variables(j2_env.parse(template))
        self._context = {}

    def add_html(self, name: str, html_content: str) -> CardTab:
        """
        Adds html to the CardTab.

        Args:
            name: Name of the variable in the Jinja2 template.
            html_content: The html to replace the named template variable.

        Returns:
            The updated card instance.
        """
        if name not in self._variables:
            raise MlflowException(
                f"{name} is not a valid template variable defined in template: '{self.template}'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self._context[name] = html_content
        return self

    def add_markdown(self, name: str, markdown: str) -> CardTab:
        """
        Adds markdown to the card replacing the variable name in the CardTab template.

        Args:
            name: Name of the variable in the CardTab Jinja2 template.
            markdown: The markdown content.

        Returns:
            The updated card tab instance.
        """
        from markdown import markdown as md_to_html

        self.add_html(name, md_to_html(markdown))
        return self

    def add_image(
        self,
        name: str,
        image_file_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        if not os.path.exists(image_file_path):
            self.add_html(name, "Image Unavailable")
            _logger.warning(f"Unable to locate image file {image_file_path} to render {name}.")
            return

        with open(image_file_path, "rb") as f:
            base64_str = base64.b64encode(f.read()).decode("utf-8")

        image_type = pathlib.Path(image_file_path).suffix[1:]

        width_style = f'width="{width}"' if width else ""
        height_style = f'height="{width}"' if height else ""
        img_html = (
            f'<img src="data:image/{image_type};base64, {base64_str}" '
            f"{width_style} {height_style} />"
        )
        self.add_html(name, img_html)

    def add_pandas_profile(self, name: str, profile: str) -> CardTab:
        """
        Add a new tab representing the provided pandas profile to the card.

        Args:
            name: Name of the variable in the Jinja2 template.
            profile: HTML string to render profile in the step card.

        Returns:
            The updated card instance.
        """
        try:
            profile_iframe = (
                "<iframe srcdoc='{src}' width='100%' height='500' frameborder='0'></iframe>"
            ).format(src=html.escape(profile))
        except Exception as e:
            profile_iframe = f"Unable to create data profile. Error found:\n{e}"
        self.add_html(name, profile_iframe)
        return self

    def to_html(self) -> str:
        """
        Returns a rendered HTML representing the content of the tab.

        Returns:
            a HTML string
        """
        from jinja2 import BaseLoader
        from jinja2.sandbox import SandboxedEnvironment

        j2_env = SandboxedEnvironment(loader=BaseLoader()).from_string(self.template)
        return j2_env.render({**self._context})


class BaseCard:
    def __init__(self, recipe_name: str, step_name: str) -> None:
        """
        BaseCard Constructor

        Args:
            recipe_name: A string representing name of the recipe.
            step_name: A string representing the name of the step.
        """
        self._recipe_name = recipe_name
        self._step_name = step_name
        self._template_name = "base.html"

        self._string_builder = StringIO()
        self._tabs = []

    def add_tab(self, name, html_template) -> CardTab:
        """
        Add a new tab with arbitrary content.

        Args:
            name: A string representing the name of the tab.
            html_template: A string representing the HTML template for the card content.
        """
        tab = CardTab(name, html_template)
        self._tabs.append((name, tab))
        return tab

    def get_tab(self, name) -> Union[CardTab, None]:
        """
        Returns an existing tab with the specified name. Returns None if not found.

        Args:
            name: A string representing the name of the tab.

        Returns:
            An existing tab with the specified name. If not found, returns None.
        """
        for key, tab in self._tabs:
            if key == name:
                return tab
        return None

    def add_text(self, text: str) -> BaseCard:
        """
        Add text to the textual representation of this card.

        Args:
            text: A string text.

        Returns:
            The updated card instance.
        """
        self._string_builder.write(text)
        return self

    def to_html(self) -> str:
        """
        This function renders the Jinja2 template based on the provided context so far.

        Returns:
            A HTML string.
        """
        import jinja2

        def get_random_id(length=6):
            return "".join(
                random.choice(string.ascii_lowercase + string.digits) for _ in range(length)
            )

        base_template_path = os.path.join(os.path.dirname(__file__), "templates")
        j2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(base_template_path))
        tab_list = [(name, tab.to_html()) for name, tab in self._tabs]
        page_id = get_random_id()
        return j2_env.get_template(self._template_name).render(
            {
                "HEADER_TITLE": f"{self._step_name.capitalize()}@{self._recipe_name}",
                "TABLINK": f"tablink-{page_id}",
                "CONTENT": f"content-{page_id}",
                "BUTTON_CONTAINER": f"button-container-{page_id}",
                "tab_list": tab_list,
            }
        )

    def to_text(self) -> str:
        """
        Returns:
            The textual representation of the card.
        """
        return self._string_builder.getvalue()

    def save_as_html(self, path) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, CARD_HTML_NAME)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, CARD_PICKLE_NAME)
        with open(path, "wb") as out:
            pickle.dump(self, out)

    @staticmethod
    def load(path):
        if os.path.isdir(path):
            path = os.path.join(path, CARD_PICKLE_NAME)
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def render_table(table, columns=None, hide_index=True):
        """
        Renders a table-like object as an HTML table.

        Args:
            table: Table-like object (e.g. pandas DataFrame, 2D numpy array, list of tuples).
            columns: Column names to use. If `table` doesn't have column names, this argument
                provides names for the columns. Otherwise, only the specified columns will be
                included in the output HTML table.
            hide_index: Hide index column when rendering.
        """
        import pandas as pd
        from pandas.io.formats.style import Styler

        pandas_version = Version(pd.__version__)

        if not isinstance(table, Styler):
            table = pd.DataFrame(table, columns=columns)
            # Escape specific characters in HTML to prevent
            # javascript code injection
            # Note that `pandas_df.style.to_html(escape=True) does not work
            # So that we have to manually escape values in dataframe cells.

            def escape_value(x):
                return html.escape(str(x))

            if hasattr(table, "map"):
                table = table.map(escape_value)
            else:
                if pandas_version >= Version("2.1.0"):
                    table = table.map(escape_value)
                else:
                    table = table.applymap(escape_value)
            table = table.style

        styler = table.set_table_attributes('style="border-collapse:collapse"').set_table_styles(
            [
                {
                    "selector": "table, th, td",
                    "props": [
                        ("border", "1px solid grey"),
                        ("text-align", "left"),
                        ("padding", "5px"),
                    ],
                },
            ]
        )
        if hide_index:
            rendered_table = (
                styler.hide(axis="index").to_html()
                if pandas_version >= Version("1.4.0")
                else styler.hide_index().render()
            )
        else:
            rendered_table = (
                styler.to_html() if pandas_version >= Version("1.4.0") else styler.render()
            )
        return f'<div style="max-height: 500px; overflow: scroll;">{rendered_table}</div>'


class FailureCard(BaseCard):
    """
    Step card providing information about a failed step execution, including a stacktrace.

    TODO: Migrate the failure card to a tab-based card, removing this class and its associated
          HTML template in the process.
    """

    def __init__(
        self, recipe_name: str, step_name: str, failure_traceback: str, output_directory: str
    ):
        super().__init__(
            recipe_name=recipe_name,
            step_name=step_name,
        )
        self.add_tab("Step Status", "{{ STEP_STATUS }}").add_html(
            "STEP_STATUS",
            '<p><strong>Step status: <span style="color:red">Failed</span></strong></p>',
        )
        self.add_tab(
            "Stacktrace",
            (
                "<div class='stacktrace-container'><p style='margin-top:0px'><code>"
                "{{ STACKTRACE|e }}</code></p></div>"
            ),
        ).add_html("STACKTRACE", str(failure_traceback))
        warning_output_path = os.path.join(output_directory, "warning_logs.txt")
        if os.path.exists(warning_output_path):
            with open(warning_output_path) as f:
                self.add_tab("Warning Logs", "{{ STEP_WARNINGS }}").add_html(
                    "STEP_WARNINGS", f"<pre>{f.read()}</pre>"
                )
