import os
from sphinx.util import logging

from docutils.parsers.rst import Directive
from docutils import nodes
from sphinx.util.osutil import copyfile

logger = logging.getLogger(__name__)

JS_FILE = 'languagesections.js'

class CodeSectionDirective(Directive):
    has_content = True

    def run(self):
        self.assert_has_content()
        text = '\n'.join(self.content)
        node = nodes.container(text)
        node['classes'].append('code-section')
        self.add_name(node)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

class PlainSectionDirective(Directive):
    has_content = True

    def run(self):
        self.assert_has_content()
        text = '\n'.join(self.content)
        node = nodes.container(text)
        node['classes'].append('plain-section')
        self.add_name(node)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

def add_assets(app):
    app.add_javascript(JS_FILE)

def copy_assets(app, exception):
    if app.builder.name != 'html' or exception:
        return
    logger.info('Copying examplecode stylesheet/javascript... ', nonl=True)
    dest = os.path.join(app.builder.outdir, '_static', JS_FILE)
    source = os.path.join(os.path.abspath(os.path.dirname(__file__)), JS_FILE)
    copyfile(source, dest)
    logger.info('done')

def setup(app):
    app.add_directive('code-section', CodeSectionDirective)
    app.add_directive('plain-section', PlainSectionDirective)
    app.connect('builder-inited', add_assets)
    app.connect('build-finished', copy_assets)
