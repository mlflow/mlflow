import re
from typing import List

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.directives import TocTree
from docutils.nodes import container
from sphinx.directives.other import glob_re
from sphinx.util import url_re, docname_join
from sphinx.util.matching import Matcher, patfilter

explicit_title_re = re.compile(r'^(.+?)\s*(?<!\x00)<([^<>]*?)>$', re.DOTALL)


class EvalTocTree(TocTree):
    has_content = TocTree.has_content
    option_spec = TocTree.option_spec
    required_arguments = TocTree.required_arguments
    optional_arguments = TocTree.optional_arguments
    final_argument_whitespace = TocTree.final_argument_whitespace

    def parse_content(self, toctree):
        suffixes = self.config.source_suffix

        # glob target documents
        all_docnames = self.env.found_docs.copy()
        all_docnames.remove(self.env.docname)  # remove current document

        ret = []
        excluded = Matcher(self.config.exclude_patterns)
        for entry in self.content:
            if not entry:
                continue
            # look for explicit titles ("Some Title <document>")
            explicit = explicit_title_re.match(entry)  # this regexp is the only difference from the original
            if (toctree['glob'] and glob_re.match(entry) and
                    not explicit and not url_re.match(entry)):
                patname = docname_join(self.env.docname, entry)
                docnames = sorted(patfilter(all_docnames, patname))
                for docname in docnames:
                    all_docnames.remove(docname)  # don't include it again
                    toctree['entries'].append((None, docname))
                    toctree['includefiles'].append(docname)
                if not docnames:
                    ret.append(self.state.document.reporter.warning(
                        'toctree glob pattern %r didn\'t match any documents'
                        % entry, line=self.lineno))
            else:
                if explicit:
                    ref = explicit.group(2)
                    title = container('')

                    self.state.nested_parse(StringList([explicit.group(1)]), self.content_offset, title)
                    toctree.append(title)
                    docname = ref
                else:
                    ref = docname = entry
                    title = None
                # remove suffixes (backwards compatibility)
                for suffix in suffixes:
                    if docname.endswith(suffix):
                        docname = docname[:-len(suffix)]
                        break
                # absolutize filenames
                docname = docname_join(self.env.docname, docname)
                if url_re.match(ref) or ref == 'self':
                    toctree['entries'].append((title, ref))
                elif docname not in self.env.found_docs:
                    if excluded(self.env.doc2path(docname, None)):
                        message = 'toctree contains reference to excluded document %r'
                    else:
                        message = 'toctree contains reference to nonexisting document %r'

                    ret.append(self.state.document.reporter.warning(message % docname,
                                                                    line=self.lineno))
                    self.env.note_reread()
                else:
                    all_docnames.discard(docname)
                    toctree['entries'].append((title, docname))
                    toctree['includefiles'].append(docname)

        # entries contains all entries (self references, external links etc.)
        if 'reversed' in self.options:
            toctree['entries'] = list(reversed(toctree['entries']))

        return ret


class TileTocTree(EvalTocTree):
    option_spec = {
        'name': directives.unchanged,
        'caption': directives.unchanged_required,
        'glob': directives.flag,
        'hidden': directives.flag,
        'includehidden': directives.flag,
        'newtab': directives.flag
    }

    def run(self) -> List[nodes.Node]:
        self.options['maxdepth'] = 1
        ret = super().run()
        ret[0]['classes'].append('tile-toctree')
        if 'newtab' in self.options:
            ret[0]['newtab'] = True
        return ret
