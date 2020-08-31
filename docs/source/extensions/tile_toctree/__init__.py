from sphinx.application import Sphinx

from .directives import TileTocTree, EvalTocTree
from .toctree import apply_overrides
from .transforms import DocinfoTransform, EvalTocTreeTransform


def setup(app: Sphinx):
    app.add_directive('tile-toctree', TileTocTree)
    app.add_directive('toctree', EvalTocTree, override=True)
    app.add_transform(DocinfoTransform)
    app.add_transform(EvalTocTreeTransform)
    app.add_config_value('tile-toctree-enable', True, True)
    apply_overrides()
