from typing import Dict

from docutils import nodes
from docutils.nodes import Element, paragraph
from sphinx import addnodes
from sphinx.environment import BuildEnvironment, TocTree
from docutils.utils import relative_path
from os import path


def _transform_toc_item(env: BuildEnvironment, docname: str, meta: Dict[str, Dict[str, str]], item: Element,
                        new_tab: bool) -> None:
    item['classes'] = ['tile-toctree__item']

    title = item[0]
    title[0]['classes'] = ['tile-toctree__title']
    item.remove(title)

    right = nodes.compound('', classes=['title-toctree__right'])
    item.append(right)
    right.append(title)

    if title[0].attributes.get('internal', False):
        if 'toc-description' in meta:
            description = paragraph(meta['toc-description'], meta['toc-description'],
                                    classes=['tile-toctree__description'])
            right.append(description)
        if 'toc-icon' in meta:
            image = nodes.image(meta['toc-icon'], alt='Toc tree icon', classes=['tile-toctree__icon'])
            uri = meta['toc-icon']
            if (uri.startswith('/')):
                (uri, abs) = env.relfn2path(uri, docname)
                uri = relative_path(path.join(env.srcdir, 'dummy'), uri)
            image['uri'] = uri
            image['candidates'] = ['*']
            item.insert(0, image)

    if new_tab:
        refs = item.traverse(nodes.reference)
        for ref in refs:
            ref['target'] = '_blank'


def _try_transform_toctree(env: BuildEnvironment, original: Element, docname: str, resolved: Element) -> None:
    new_tab = 'newtab' in original.parent
    if 'tile-toctree' in original.parent['classes']:
        for i in range(len(original['entries'])):
            item = resolved[0][i]
            ref = original['entries'][i][1]
            meta = env.metadata[ref]
            _transform_toc_item(env, docname, meta, item, new_tab)


def apply_overrides():
    def get_and_resolve_doctree(self, docname, builder, doctree=None,
                                prune_toctrees=True, includehidden=False):
        # type: (unicode, Builder, nodes.Node, bool, bool) -> nodes.Node
        """Read the doctree from the pickle, resolve cross-references and
        toctrees and return it.
        """
        if doctree is None:
            doctree = self.get_doctree(docname)

        # resolve all pending cross-references
        self.apply_post_transforms(doctree, docname)

        # now, resolve all toctree nodes
        for toctreenode in doctree.traverse(addnodes.toctree):
            result = TocTree(self).resolve(docname, builder, toctreenode,
                                           prune=prune_toctrees,
                                           includehidden=includehidden)
            if result is None:
                toctreenode.replace_self([])
            else:
                if self.config['tile-toctree-enable']:
                    _try_transform_toctree(self, toctreenode, docname, result)
                toctreenode.replace_self(result)

        return doctree

    BuildEnvironment.get_and_resolve_doctree = get_and_resolve_doctree
