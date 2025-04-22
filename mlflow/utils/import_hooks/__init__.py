"""
NOTE: The contents of this file have been inlined from the wrapt package's source code
https://github.com/GrahamDumpleton/wrapt/blob/1.12.1/src/wrapt/importer.py.
Some modifications, have been made in order to:
    - avoid duplicate registration of import hooks
    - inline functions from dependent wrapt submodules rather than importing them.

This module implements a post import hook mechanism styled after what is described in PEP-369.
Note that it doesn't cope with modules being reloaded.
It also extends the functionality to support custom hooks for import errors
(as opposed to only successful imports).
"""

import importlib  # noqa: F401
import sys
import threading

string_types = (str,)


# from .decorators import synchronized
# NOTE: Instead of using this import (from wrapt's decorator module, see
# https://github.com/GrahamDumpleton/wrapt/blob/68316bea668fd905a4acb21f37f12596d8c30d80/src/wrapt/decorators.py#L430-L456),
# we define a decorator with similar behavior that acquires a lock while calling the decorated
# function
def synchronized(lock):
    def decorator(f):
        # See e.g. https://www.python.org/dev/peps/pep-0318/#examples
        def new_fn(*args, **kwargs):
            with lock:
                return f(*args, **kwargs)

        return new_fn

    return decorator


# The dictionary registering any post import hooks to be triggered once
# the target module has been imported. Once a module has been imported
# and the hooks fired, the list of hooks recorded against the target
# module will be truncated but the list left in the dictionary. This
# acts as a flag to indicate that the module had already been imported.

_post_import_hooks = {}
_post_import_hooks_lock = threading.RLock()

# A dictionary for any import hook error handlers to be triggered when the
# target module import fails.

_import_error_hooks = {}
_import_error_hooks_lock = threading.RLock()

_import_hook_finder_init = False

# Register a new post import hook for the target module name. This
# differs from the PEP-369 implementation in that it also allows the
# hook function to be specified as a string consisting of the name of
# the callback in the form 'module:function'. This will result in a
# proxy callback being registered which will defer loading of the
# specified module containing the callback function until required.


def _create_import_hook_from_string(name):
    def import_hook(module):
        module_name, function = name.split(":")
        attrs = function.split(".")
        __import__(module_name)
        callback = sys.modules[module_name]
        for attr in attrs:
            callback = getattr(callback, attr)
        return callback(module)

    return import_hook


def register_generic_import_hook(hook, name, hook_dict, overwrite):
    # Create a deferred import hook if hook is a string name rather than
    # a callable function.

    if isinstance(hook, string_types):
        hook = _create_import_hook_from_string(hook)

    # Automatically install the import hook finder if it has not already
    # been installed.

    global _import_hook_finder_init
    if not _import_hook_finder_init:
        _import_hook_finder_init = True
        sys.meta_path.insert(0, ImportHookFinder())

    # Determine if any prior registration of an import hook for
    # the target modules has occurred and act appropriately.

    hooks = hook_dict.get(name, None)

    if hooks is None:
        # No prior registration of import hooks for the target
        # module. We need to check whether the module has already been
        # imported. If it has we fire the hook immediately and add an
        # empty list to the registry to indicate that the module has
        # already been imported and hooks have fired. Otherwise add
        # the post import hook to the registry.

        module = sys.modules.get(name, None)

        if module is not None:
            hook_dict[name] = []
            hook(module)

        else:
            hook_dict[name] = [hook]

    elif hooks == []:
        # A prior registration of import hooks for the target
        # module was done and the hooks already fired. Fire the hook
        # immediately.

        module = sys.modules[name]
        hook(module)

    else:
        # A prior registration of import hooks for the target
        # module was done but the module has not yet been imported.

        def hooks_equal(existing_hook, hook):
            if hasattr(existing_hook, "__name__") and hasattr(hook, "__name__"):
                return existing_hook.__name__ == hook.__name__
            else:
                return False

        if overwrite:
            hook_dict[name] = [
                existing_hook
                for existing_hook in hook_dict[name]
                if not hooks_equal(existing_hook, hook)
            ]

        hook_dict[name].append(hook)


@synchronized(_import_error_hooks_lock)
def register_import_error_hook(hook, name, overwrite=True):
    """
    Args:
        hook: A function or string entrypoint to invoke when the specified module is imported
            and an error occurs.
        name: The name of the module for which to fire the hook at import error detection time.
        overwrite: Specifies the desired behavior when a preexisting hook for the same
            function / entrypoint already exists for the specified module. If `True`,
            all preexisting hooks matching the specified function / entrypoint will be
            removed and replaced with a single instance of the specified `hook`.
    """
    register_generic_import_hook(hook, name, _import_error_hooks, overwrite)


@synchronized(_post_import_hooks_lock)
def register_post_import_hook(hook, name, overwrite=True):
    """
    Args:
        hook: A function or string entrypoint to invoke when the specified module is imported.
        name: The name of the module for which to fire the hook at import time.
        overwrite: Specifies the desired behavior when a preexisting hook for the same
            function / entrypoint already exists for the specified module. If `True`,
            all preexisting hooks matching the specified function / entrypoint will be
            removed and replaced with a single instance of the specified `hook`.
    """
    register_generic_import_hook(hook, name, _post_import_hooks, overwrite)


@synchronized(_post_import_hooks_lock)
def get_post_import_hooks(name):
    return _post_import_hooks.get(name)


# Register post import hooks defined as package entry points.


def _create_import_hook_from_entrypoint(entrypoint):
    def import_hook(module):
        __import__(entrypoint.module_name)
        callback = sys.modules[entrypoint.module_name]
        for attr in entrypoint.attrs:
            callback = getattr(callback, attr)
        return callback(module)

    return import_hook


def discover_post_import_hooks(group):
    # New in 3.9: https://docs.python.org/3/library/importlib.resources.html#importlib.resources.files
    if sys.version_info.major > 2 and sys.version_info.minor > 8:
        from importlib.resources import files  # clint: disable=lazy-builtin-import

        for entrypoint in (
            resource.name for resource in files(group).iterdir() if resource.is_file()
        ):
            callback = _create_import_hook_from_entrypoint(entrypoint)
            register_post_import_hook(callback, entrypoint.name)
    else:
        from importlib.resources import contents  # clint: disable=lazy-builtin-import

        for entrypoint in contents(group):
            callback = _create_import_hook_from_entrypoint(entrypoint)
            register_post_import_hook(callback, entrypoint.name)


# Indicate that a module has been loaded. Any post import hooks which
# were registered against the target module will be invoked. If an
# exception is raised in any of the post import hooks, that will cause
# the import of the target module to fail.


@synchronized(_post_import_hooks_lock)
def notify_module_loaded(module):
    name = getattr(module, "__name__", None)
    hooks = _post_import_hooks.get(name)

    if hooks:
        _post_import_hooks[name] = []

        for hook in hooks:
            hook(module)


@synchronized(_import_error_hooks_lock)
def notify_module_import_error(module_name):
    hooks = _import_error_hooks.get(module_name)

    if hooks:
        # Error hooks differ from post import hooks, in that we don't clear the
        # hook as soon as it fires.
        for hook in hooks:
            hook(module_name)


# A custom module import finder. This intercepts attempts to import
# modules and watches out for attempts to import target modules of
# interest. When a module of interest is imported, then any post import
# hooks which are registered will be invoked.


class _ImportHookChainedLoader:
    def __init__(self, loader):
        self.loader = loader

    def load_module(self, fullname):
        try:
            module = self.loader.load_module(fullname)
            notify_module_loaded(module)
        except (ImportError, AttributeError):
            notify_module_import_error(fullname)
            raise

        return module


class ImportHookFinder:
    def __init__(self):
        self.in_progress = {}

    @synchronized(_post_import_hooks_lock)
    @synchronized(_import_error_hooks_lock)
    def find_module(self, fullname, path=None):
        # If the module being imported is not one we have registered
        # import hooks for, we can return immediately. We will
        # take no further part in the importing of this module.

        if fullname not in _post_import_hooks and fullname not in _import_error_hooks:
            return None

        # When we are interested in a specific module, we will call back
        # into the import system a second time to defer to the import
        # finder that is supposed to handle the importing of the module.
        # We set an in progress flag for the target module so that on
        # the second time through we don't trigger another call back
        # into the import system and cause a infinite loop.

        if fullname in self.in_progress:
            return None

        self.in_progress[fullname] = True

        # Now call back into the import system again.

        try:
            # For Python 3 we need to use find_spec().loader
            # from the importlib.util module. It doesn't actually
            # import the target module and only finds the
            # loader. If a loader is found, we need to return
            # our own loader which will then in turn call the
            # real loader to import the module and invoke the
            # post import hooks.
            try:
                import importlib.util  # clint: disable=lazy-builtin-import

                loader = importlib.util.find_spec(fullname).loader
            # If an ImportError (or AttributeError) is encountered while finding the module,
            # notify the hooks for import errors
            except (ImportError, AttributeError):
                notify_module_import_error(fullname)
                loader = importlib.find_loader(fullname, path)
            if loader:
                return _ImportHookChainedLoader(loader)
        finally:
            del self.in_progress[fullname]

    @synchronized(_post_import_hooks_lock)
    @synchronized(_import_error_hooks_lock)
    def find_spec(self, fullname, path, target=None):
        # If the module being imported is not one we have registered
        # import hooks for, we can return immediately. We will
        # take no further part in the importing of this module.

        if fullname not in _post_import_hooks and fullname not in _import_error_hooks:
            return None

        # When we are interested in a specific module, we will call back
        # into the import system a second time to defer to the import
        # finder that is supposed to handle the importing of the module.
        # We set an in progress flag for the target module so that on
        # the second time through we don't trigger another call back
        # into the import system and cause a infinite loop.

        if fullname in self.in_progress:
            return None

        self.in_progress[fullname] = True

        # Now call back into the import system again.

        try:
            import importlib.util  # clint: disable=lazy-builtin-import

            spec = importlib.util.find_spec(fullname)
            # Replace the module spec's loader with a wrapped version that executes import
            # hooks when the module is loaded
            spec.loader = _ImportHookChainedLoader(spec.loader)
            return spec
        except (ImportError, AttributeError):
            notify_module_import_error(fullname)
        finally:
            del self.in_progress[fullname]


# Decorator for marking that a function should be called as a post
# import hook when the target module is imported.
# If error_handler is True, then apply the marked function as an import hook
# for import errors (instead of successful imports).
# It is assumed that all error hooks are added during driver start-up,
# and thus added prior to any import calls. If an error hook is added
# after a module has already failed the import, there's no guarantee
# that the hook will fire.


def when_imported(name, error_handler=False):
    def register(hook):
        if error_handler:
            register_import_error_hook(hook, name)
        else:
            register_post_import_hook(hook, name)
        return hook

    return register
