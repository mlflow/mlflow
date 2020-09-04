""" Tabbed views for Sphinx, with HTML builder """

import base64
import json
import os
import posixpath

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from pkg_resources import resource_filename
from pygments.lexers import get_all_lexers
from sphinx.application import Sphinx
from sphinx.util.osutil import copyfile
from sphinx.util import logging
from sphinx.builders.html import StandaloneHTMLBuilder


FILES = [
    "semantic-ui-2.4.1/segment.min.css",
    "semantic-ui-2.4.1/menu.min.css",
    "semantic-ui-2.4.1/tab.min.css",
    "semantic-ui-2.4.1/tab.min.js",
    "tabs.js",
    "tabs.css",
]


LEXER_MAP = {}
for lexer in get_all_lexers():
    for short_name in lexer[1]:
        LEXER_MAP[short_name] = lexer[0]


def get_compatible_builders(app):
    builders = [
        "html",
        "singlehtml",
        "dirhtml",
        "readthedocs",
        "readthedocsdirhtml",
        "readthedocssinglehtml",
        "readthedocssinglehtmllocalmedia",
        "spelling",
    ]
    builders.extend(app.config["sphinx_tabs_valid_builders"])
    return builders


class TabsDirective(Directive):
    """ Top-level tabs directive """

    has_content = True

    def run(self):
        """ Parse a tabs directive """
        self.assert_has_content()
        env = self.state.document.settings.env

        node = nodes.container()
        node["classes"] = ["sphinx-tabs"]

        if "next_tabs_id" not in env.temp_data:
            env.temp_data["next_tabs_id"] = 0
        if "tabs_stack" not in env.temp_data:
            env.temp_data["tabs_stack"] = []

        tabs_id = env.temp_data["next_tabs_id"]
        tabs_key = "tabs_%d" % tabs_id
        env.temp_data["next_tabs_id"] += 1
        env.temp_data["tabs_stack"].append(tabs_id)

        env.temp_data[tabs_key] = {}
        env.temp_data[tabs_key]["tab_ids"] = []
        env.temp_data[tabs_key]["tab_titles"] = []
        env.temp_data[tabs_key]["is_first_tab"] = True

        self.state.nested_parse(self.content, self.content_offset, node)

        if env.app.builder.name in get_compatible_builders(env.app):
            tabs_node = nodes.container()
            tabs_node.tagname = "div"

            classes = "ui top attached tabular menu sphinx-menu"
            tabs_node["classes"] = classes.split(" ")

            tab_titles = env.temp_data[tabs_key]["tab_titles"]
            for idx, [data_tab, tab_name] in enumerate(tab_titles):
                tab = nodes.container()
                tab.tagname = "a"
                tab["classes"] = ["item"] if idx > 0 else ["active", "item"]
                tab["classes"].append(data_tab)
                tab += tab_name
                tabs_node += tab

            node.children.insert(0, tabs_node)

        env.temp_data["tabs_stack"].pop()
        return [node]


class TabDirective(Directive):
    """ Tab directive, for adding a tab to a collection of tabs """

    has_content = True

    def run(self):
        """ Parse a tab directive """
        self.assert_has_content()
        env = self.state.document.settings.env

        tabs_id = env.temp_data["tabs_stack"][-1]
        tabs_key = "tabs_%d" % tabs_id

        args = self.content[0].strip()
        if args.startswith("{"):
            try:
                args = json.loads(args)
                self.content.trim_start(1)
            except ValueError:
                args = {}
        else:
            args = {}

        tab_name = nodes.container()
        self.state.nested_parse(self.content[:1], self.content_offset, tab_name)
        args["tab_name"] = tab_name

        include_tabs_id_in_data_tab = False
        if "tab_id" not in args:
            args["tab_id"] = env.new_serialno(tabs_key)
            include_tabs_id_in_data_tab = True
        i = 1
        while args["tab_id"] in env.temp_data[tabs_key]["tab_ids"]:
            args["tab_id"] = "%s-%d" % (args["tab_id"], i)
            i += 1
        env.temp_data[tabs_key]["tab_ids"].append(args["tab_id"])

        data_tab = str(args["tab_id"])
        if include_tabs_id_in_data_tab:
            data_tab = "%d-%s" % (tabs_id, data_tab)
        data_tab = "sphinx-data-tab-{}".format(data_tab)

        env.temp_data[tabs_key]["tab_titles"].append((data_tab, args["tab_name"]))

        text = "\n".join(self.content)
        node = nodes.container(text)

        classes = "ui bottom attached sphinx-tab tab segment"
        node["classes"] = classes.split(" ")
        node["classes"].extend(args.get("classes", []))
        node["classes"].append(data_tab)

        if env.temp_data[tabs_key]["is_first_tab"]:
            node["classes"].append("active")
            env.temp_data[tabs_key]["is_first_tab"] = False

        self.state.nested_parse(self.content[2:], self.content_offset, node)

        if env.app.builder.name not in get_compatible_builders(env.app):
            outer_node = nodes.container()
            tab = nodes.container()
            tab.tagname = "a"
            tab["classes"] = ["item"]
            tab += tab_name

            outer_node.append(tab)
            outer_node.append(node)
            return [outer_node]

        return [node]


class GroupTabDirective(Directive):
    """ Tab directive that toggles with same tab names across page"""

    has_content = True

    def run(self):
        """ Parse a tab directive """
        self.assert_has_content()

        group_name = self.content[0]
        self.content.trim_start(2)

        for idx, line in enumerate(self.content.data):
            self.content.data[idx] = "   " + line

        tab_args = {
            "tab_id": base64.b64encode(group_name.encode("utf-8")).decode("utf-8"),
            "group_tab": True,
        }

        new_content = [
            ".. tab:: {}".format(json.dumps(tab_args)),
            "   {}".format(group_name),
            "",
        ]

        for idx, line in enumerate(new_content):
            self.content.data.insert(idx, line)
            self.content.items.insert(idx, (None, idx))

        node = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, node)
        return node.children


class CodeTabDirective(Directive):
    """ Tab directive with a codeblock as its content"""

    has_content = True
    option_spec = {"linenos": directives.flag}

    def run(self):
        """ Parse a tab directive """
        self.assert_has_content()

        args = self.content[0].strip().split()
        self.content.trim_start(2)

        lang = args[0]
        tab_name = " ".join(args[1:]) if len(args) > 1 else LEXER_MAP[lang]

        for idx, line in enumerate(self.content.data):
            self.content.data[idx] = "      " + line

        tab_args = {
            "tab_id": base64.b64encode(tab_name.encode("utf-8")).decode("utf-8"),
            "classes": ["code-tab"],
        }

        new_content = [
            ".. tab:: {}".format(json.dumps(tab_args)),
            "   {}".format(tab_name),
            "",
            "   .. code-block:: {}".format(lang),
        ]

        if "linenos" in self.options:
            new_content.append("      :linenos:")

        new_content.append("")

        for idx, line in enumerate(new_content):
            self.content.data.insert(idx, line)
            self.content.items.insert(idx, (None, idx))

        node = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, node)
        return node.children


class _FindTabsDirectiveVisitor(nodes.NodeVisitor):
    """ Visitor pattern than looks for a sphinx tabs
        directive in a document """

    def __init__(self, document):
        nodes.NodeVisitor.__init__(self, document)
        self._found = False

    def unknown_visit(self, node):
        if (
            not self._found
            and isinstance(node, nodes.container)
            and "classes" in node
            and isinstance(node["classes"], list)
        ):
            self._found = "sphinx-tabs" in node["classes"]

    @property
    def found_tabs_directive(self):
        """ Return whether a sphinx tabs directive was found """
        return self._found


# pylint: disable=unused-argument
def update_context(app, pagename, templatename, context, doctree):
    """ Remove sphinx-tabs CSS and JS asset files if not used in a page """
    if doctree is None:
        return
    visitor = _FindTabsDirectiveVisitor(doctree)
    doctree.walk(visitor)
    if not visitor.found_tabs_directive:
        paths = [posixpath.join("_static", "sphinx_tabs/" + f) for f in FILES]
        if "css_files" in context:
            context["css_files"] = context["css_files"][:]
            for path in paths:
                if path.endswith(".css") and path in context["css_files"]:
                    context["css_files"].remove(path)
        if "script_files" in context:
            context["script_files"] = context["script_files"][:]
            for path in paths:
                if path.endswith(".js") and path in context["script_files"]:
                    context["script_files"].remove(path)


# pylint: enable=unused-argument


def copy_assets(app, exception):
    """ Copy asset files to the output """
    if "getLogger" in dir(logging):
        log = logging.getLogger(__name__).info  # pylint: disable=no-member
        warn = logging.getLogger(__name__).warning  # pylint: disable=no-member
    else:
        log = app.info
        warn = app.warning
    builders = get_compatible_builders(app)
    if exception:
        return
    if app.builder.name not in builders:
        if not app.config["sphinx_tabs_nowarn"]:
            warn(
                "Not copying tabs assets! Not compatible with %s builder"
                % app.builder.name
            )
        return

    log("Copying tabs assets")

    installdir = os.path.join(app.builder.outdir, "_static", "sphinx_tabs")

    for path in FILES:
        source = resource_filename("sphinx_tabs", path)
        dest = os.path.join(installdir, path)

        destdir = os.path.dirname(dest)
        if not os.path.exists(destdir):
            os.makedirs(destdir)

        copyfile(source, dest)


def add_stylesheets(app: Sphinx):
    for path in ["sphinx_tabs/" + f for f in FILES]:
        if path.endswith(".css"):
            if "add_css_file" in dir(app):
                app.add_css_file(path)
            else:
                app.add_stylesheet(path)
        if path.endswith(".js"):
            if "add_script_file" in dir(app):
                app.add_script_file(path)
            else:
                app.add_js_file(path)


def setup(app: Sphinx):
    """ Set up the plugin """
    app.add_config_value("sphinx_tabs_nowarn", False, "")
    app.add_config_value("sphinx_tabs_valid_builders", [], "")
    app.add_directive("tabs", TabsDirective)
    app.add_directive("tab", TabDirective)
    app.add_directive("group-tab", GroupTabDirective)
    app.add_directive("code-tab", CodeTabDirective)
    app.connect('builder-inited', add_stylesheets)
    app.connect("html-page-context", update_context)
    app.connect("build-finished", copy_assets)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
