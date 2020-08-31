from docutils import nodes
from docutils.transforms.references import Substitutions
from sphinx import addnodes
from sphinx.transforms import SphinxTransform, DoctreeReadEvent


class DocinfoTransform(SphinxTransform):
    default_priority = DoctreeReadEvent.default_priority - 1

    def apply(self, **kwargs):
        docinfo_meta = {}

        use_toc_description = False
        enable = self.config['tile-toctree-enable']

        for meta in self.document.traverse(addnodes.meta):
            if meta['name'] == 'toc-description':
                if enable:
                    docinfo_meta['toc-description'] = meta.children
                    use_toc_description = True
            elif meta['name'] == 'description':
                if enable and not use_toc_description:
                    docinfo_meta['toc-description'] = meta.children
            elif meta['name'] == 'toc-icon':
                if enable:
                    docinfo_meta['toc-icon'] = meta.children
                meta.parent.remove(meta)

        if bool(docinfo_meta):
            docinfo_index = self.document.first_child_matching_class(nodes.docinfo)
            if docinfo_index is None:
                docinfo = nodes.docinfo()
                self.document.insert(0, docinfo)
            else:
                docinfo = self.document[docinfo_index]
            for key in docinfo_meta:
                field = nodes.field()
                field.append(nodes.field_name(key, key))
                field.append(nodes.field_body('', *docinfo_meta[key]))
                docinfo.append(field)


class EvalTocTreeTransform(SphinxTransform):
    default_priority = Substitutions.default_priority + 1

    def apply(self, **kwargs):
        for toctree in self.document.traverse(addnodes.toctree):
            entries = toctree['entries']
            toctree.clear()
            for i in range(len(entries)):
                (title, docname) = entries[i]
                if title:
                    # <container><paragraph>text</paragraph></container> - a kind of hack
                    entries[i] = (str(title)[22:-24], docname)
