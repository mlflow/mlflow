# Tile TocTree 

The module 

* adds a new directive, `tile-toctree` to [Sphinx](http://www.sphinx-doc.org/).
* makes the existing `toctree` directive to accept interactive markup (e.g. <Substitutions>) into titles.

Tested with Sphinx 1.8.5.

## Usage

### conf.py
```python
extensions = ['tile_toctree']
```

### tile-toctree

index.md

``` markdown
.. tile-toctree::
    :glob:

    administration/index
    cloud/index
    bi/index
```

administration/index.md

```markdown
description: `description` will be used when no `toc-description` meta is present; optional
toc-description: has priority over the regular `description` meta; also optional
toc-icon: /_static/icons/page-icon.png

# File title

file contents
```

### toctree

```markdown
:toctree:
    
    Custom title with <Definition> inside <docs/index.md>
```

## Directive settings

The directive code is based on [Sphinx TocTree directive](https://www.sphinx-doc.org/en/1.5/markup/toctree.html), so use the sphinx doc for parameters reference.

**Note**, some `toctree` parameters have been disabled for `tile-toctree`. The list of working parameters is:

* name
* caption
* glob
* hidden
* includehidden

### Additional settings

* `newtab` - makes the resulting links to open in new tabs

Example:

```markdown
.. tile-toctree::
    :newtab:

    runtime/index
```

## Directive metadata

The directve collects the following document metadata:

* `description` - the field value is used as tile description
* `toc-icon` - the field value is used as tile icon

## Enable/Disable the extension

The extension might not be needed while building some kinds of outputs, e.g. `markdown`. So there is a config setting for temporal enabling/disabling of the extension.

### tile-toctree-enable

Description: `True` to enable `tile-toctree`. `False` to disable and make it behave as a regular Sphinx `toctree`, (with the parameter-set still limited).

Default: `True`

To pass a config value through the command line, see the [command-line options](https://www.sphinx-doc.org/en/1.5/man/sphinx-build.html):

```shell script
sphinx-build -D tile-toctree-enable=0 <other_options>
```