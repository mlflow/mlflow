#                     __ __ __
#   .-----.-----.----|__|  |  .---.-.
#   |  _  |  _  |   _|  |  |  |  _  |
#   |___  |_____|__| |__|__|__|___._|
#   |_____|
#

"""
NOTE: The contents of this file have been inlined from the gorilla package's source code
https://github.com/christophercrouzet/gorilla/blob/v0.3.0/gorilla.py

This module has fixes / adaptations for MLflow use cases that make it different from the original
gorilla library

The following modifications have been made:
    - Modify `get_original_attribute` logic, search from children classes to parent classes,
      and for each class check "_gorilla_original_{attr_name}" attribute first.
      first. This will ensure get the correct original attribute in any cases, e.g.,
      the case some classes in the hierarchy haven't been patched, but some others are
      patched, this case the previous code is risky to get wrong original attribute.
    - Make `get_original_attribute` support bypassing descriptor protocol.
    - remove `get_attribute` method, use `get_original_attribute` with
      `bypass_descriptor_protocol=True` instead of calling it.
    - After reverting patch, there will be no side-effect, restore object to be exactly the
      original status.
    - Remove `create_patches` and `patches` methods.

gorilla
~~~~~~~

Convenient approach to monkey patching.

:copyright: Copyright 2014-2017 by Christopher Crouzet.
:license: MIT, see LICENSE for details.
"""

import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types

__version__ = "0.3.0"
_logger = logging.getLogger(__name__)


def _iteritems(d, **kwargs):
    return iter(d.items(**kwargs))


def _load_module(finder, name):
    loader, _ = finder.find_loader(name)
    return loader.load_module()


# Pattern for each internal attribute name.
_PATTERN = "_gorilla_%s"

# Pattern for the name of the overidden attributes to be stored.
_ORIGINAL_NAME = _PATTERN % ("original_%s",)

# Pattern for the name of the patch attributes to be stored.
_ACTIVE_PATCH = "_gorilla_active_patch_%s"

# Attribute for the decorator data.
_DECORATOR_DATA = _PATTERN % ("decorator_data",)


def default_filter(name, obj):
    """Attribute filter.

    It filters out module attributes, and also methods starting with an
    underscore ``_``.

    This is used as the default filter for the :func:`create_patches` function
    and the :func:`patches` decorator.

    Parameters
    ----------
    name : str
        Attribute name.
    obj : object
        Attribute value.

    Returns
    -------
    bool
        Whether the attribute should be returned.
    """
    return not (isinstance(obj, types.ModuleType) or name.startswith("_"))


class DecoratorData:

    """Decorator data.

    Attributes
    ----------
    patches : list of gorilla.Patch
        Patches created through the decorators.
    override : dict
        Any overriding value defined by the :func:`destination`, :func:`name`,
        and :func:`settings` decorators.
    filter : bool or None
        Value defined by the :func:`filter` decorator, if any, or ``None``
        otherwise.
    """

    def __init__(self):
        """Constructor."""
        self.patches = []
        self.override = {}
        self.filter = None


class Settings:

    """Define the patching behaviour.

    Attributes
    ----------
    allow_hit : bool
        A hit occurs when an attribute at the destination already exists with
        the name given by the patch. If ``False``, the patch process won't
        allow setting a new value for the attribute by raising an exception.
        Defaults to ``False``.
    store_hit : bool
        If ``True`` and :attr:`allow_hit` is also set to ``True``, then any
        attribute at the destination that is hit is stored under a different
        name before being overwritten by the patch. Defaults to ``True``.
    """

    def __init__(self, **kwargs):
        """Constructor.

        Parameters
        ----------
        kwargs
            Keyword arguments, see the attributes.
        """
        self.allow_hit = False
        self.store_hit = True
        self._update(**kwargs)

    def __repr__(self):
        values = ", ".join([f"{key}={value!r}" for key, value in sorted(_iteritems(self.__dict__))])
        return f"{type(self).__name__}({values})"

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__

        return NotImplemented

    def __ne__(self, other):
        is_equal = self.__eq__(other)
        return is_equal if is_equal is NotImplemented else not is_equal

    def _update(self, **kwargs):
        """Update some settings.

        Parameters
        ----------
        kwargs
            Settings to update.
        """
        self.__dict__.update(**kwargs)


class Patch:

    """Describe all the information required to apply a patch.

    Attributes
    ----------
    destination : obj
        Patch destination.
    name : str
        Name of the attribute at the destination.
    obj : obj
        Attribute value.
    settings : gorilla.Settings or None
        Settings. If ``None``, the default settings are used.

    Warning
    -------
    It is highly recommended to use the output of the function
    :func:`get_attribute` for setting the attribute :attr:`obj`. This will
    ensure that the descriptor protocol is bypassed instead of possibly
    retrieving attributes invalid for patching, such as bound methods.
    """

    def __init__(self, destination, name, obj, settings=None):
        """Constructor.

        Parameters
        ----------
        destination : object
            See the :attr:`~Patch.destination` attribute.
        name : str
            See the :attr:`~Patch.name` attribute.
        obj : object
            See the :attr:`~Patch.obj` attribute.
        settings : gorilla.Settings
            See the :attr:`~Patch.settings` attribute.
        """
        self.destination = destination
        self.name = name
        self.obj = obj
        self.settings = settings
        self.is_inplace_patch = None

    def __repr__(self):
        return "{}(destination={!r}, name={!r}, obj={!r}, settings={!r})".format(
            type(self).__name__,
            self.destination,
            self.name,
            self.obj,
            self.settings,
        )

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__

        return NotImplemented

    def __ne__(self, other):
        is_equal = self.__eq__(other)
        return is_equal if is_equal is NotImplemented else not is_equal

    def __hash__(self):  # pylint: disable=useless-super-delegation
        return super().__hash__()

    def _update(self, **kwargs):
        """Update some attributes.

        If a 'settings' attribute is passed as a dict, then it will update the
        content of the settings, if any, instead of completely overwriting it.

        Parameters
        ----------
        kwargs
            Attributes to update.

        Raises
        ------
        ValueError
            The setting doesn't exist.
        """
        for key, value in _iteritems(kwargs):
            if key == "settings":
                if isinstance(value, dict):
                    if self.settings is None:
                        self.settings = Settings(**value)
                    else:
                        self.settings._update(**value)
                else:
                    self.settings = copy.deepcopy(value)
            else:
                setattr(self, key, value)


def apply(patch):
    """Apply a patch.

    The patch's :attr:`~Patch.obj` attribute is injected into the patch's
    :attr:`~Patch.destination` under the patch's :attr:`~Patch.name`.

    This is a wrapper around calling
    ``setattr(patch.destination, patch.name, patch.obj)``.

    Parameters
    ----------
    patch : gorilla.Patch
        Patch.

    Raises
    ------
    RuntimeError
        Overwriting an existing attribute is not allowed when the setting
        :attr:`Settings.allow_hit` is set to ``True``.

    Note
    ----
    If both the attributes :attr:`Settings.allow_hit` and
    :attr:`Settings.store_hit` are ``True`` but that the target attribute seems
    to have already been stored, then it won't be stored again to avoid losing
    the original attribute that was stored the first time around.
    """
    # is_inplace_patch = True represents the patch object will overwrite the original
    # attribute
    patch.is_inplace_patch = patch.name in patch.destination.__dict__
    settings = Settings() if patch.settings is None else patch.settings

    curr_active_patch = _ACTIVE_PATCH % (patch.name,)
    if curr_active_patch in patch.destination.__dict__:
        _logger.debug(
            f"Patch {patch.name} on {destination.__name__} already existed. Overwrite old patch."
        )

    # When a hit occurs due to an attribute at the destination already existing
    # with the patch's name, the existing attribute is referred to as 'target'.
    try:
        target = get_original_attribute(
            patch.destination, patch.name, bypass_descriptor_protocol=True
        )
    except AttributeError:
        pass
    else:
        if not settings.allow_hit:
            raise RuntimeError(
                "An attribute named '%s' already exists at the destination "  # noqa: UP031
                "'%s'. Set a different name through the patch object to avoid "
                "a name clash or set the setting 'allow_hit' to True to "
                "overwrite the attribute. In the latter case, it is "
                "recommended to also set the 'store_hit' setting to True in "
                "order to store the original attribute under a different "
                "name so it can still be accessed." % (patch.name, patch.destination.__name__)
            )

        if settings.store_hit:
            original_name = _ORIGINAL_NAME % (patch.name,)
            setattr(patch.destination, original_name, target)

    setattr(patch.destination, patch.name, patch.obj)
    setattr(patch.destination, curr_active_patch, patch)


def revert(patch):
    """Revert a patch.
    Parameters
    ----------
    patch : gorilla.Patch
        Patch.
    Note
    ----
    This is only possible if the attribute :attr:`Settings.store_hit` was set
    to ``True`` when applying the patch and overriding an existing attribute.

    Notice:
    This method is taken from
    https://github.com/christophercrouzet/gorilla/blob/v0.4.0/gorilla.py#L318-L351
    with modifictions for autologging disablement purposes.
    """
    # If an curr_active_patch has not been set on destination class for the current patch,
    # then the patch has not been applied and we do not need to revert anything.
    curr_active_patch = _ACTIVE_PATCH % (patch.name,)
    if curr_active_patch not in patch.destination.__dict__:
        # already reverted.
        return

    original_name = _ORIGINAL_NAME % (patch.name,)

    if patch.is_inplace_patch:
        # check whether original_name is in destination. We cannot use hasattr because it will
        # try to get attribute from parent classes if attribute not found in destination class.
        if original_name not in patch.destination.__dict__:
            raise RuntimeError(
                "Cannot revert the attribute named '%s' since the setting "  # noqa: UP031
                "'store_hit' was not set to True when applying the patch."
                % (patch.destination.__name__,)
            )
        # restore original method
        # during reverting patch, we need restore the raw attribute to the patch point
        # so get original attribute bypassing descriptor protocal
        original = object.__getattribute__(patch.destination, original_name)
        setattr(patch.destination, patch.name, original)
    else:
        # delete patched method
        delattr(patch.destination, patch.name)

    if original_name in patch.destination.__dict__:
        delattr(patch.destination, original_name)
    delattr(patch.destination, curr_active_patch)


def patch(destination, name=None, settings=None):
    """Decorator to create a patch.

    The object being decorated becomes the :attr:`~Patch.obj` attribute of the
    patch.

    Parameters
    ----------
    destination : object
        Patch destination.
    name : str
        Name of the attribute at the destination.
    settings : gorilla.Settings
        Settings.

    Returns
    -------
    object
        The decorated object.

    See Also
    --------
    :class:`Patch`.
    """

    def decorator(wrapped):
        base = _get_base(wrapped)
        name_ = base.__name__ if name is None else name
        settings_ = copy.deepcopy(settings)
        patch = Patch(destination, name_, wrapped, settings=settings_)
        data = get_decorator_data(base, set_default=True)
        data.patches.append(patch)
        return wrapped

    return decorator


def destination(value):
    """Modifier decorator to update a patch's destination.

    This only modifies the behaviour of the :func:`create_patches` function
    and the :func:`patches` decorator, given that their parameter
    ``use_decorators`` is set to ``True``.

    Parameters
    ----------
    value : object
        Patch destination.

    Returns
    -------
    object
        The decorated object.
    """

    def decorator(wrapped):
        data = get_decorator_data(_get_base(wrapped), set_default=True)
        data.override["destination"] = value
        return wrapped

    return decorator


def name(value):
    """Modifier decorator to update a patch's name.

    This only modifies the behaviour of the :func:`create_patches` function
    and the :func:`patches` decorator, given that their parameter
    ``use_decorators`` is set to ``True``.

    Parameters
    ----------
    value : object
        Patch name.

    Returns
    -------
    object
        The decorated object.
    """

    def decorator(wrapped):
        data = get_decorator_data(_get_base(wrapped), set_default=True)
        data.override["name"] = value
        return wrapped

    return decorator


def settings(**kwargs):
    """Modifier decorator to update a patch's settings.

    This only modifies the behaviour of the :func:`create_patches` function
    and the :func:`patches` decorator, given that their parameter
    ``use_decorators`` is set to ``True``.

    Parameters
    ----------
    kwargs
        Settings to update. See :class:`Settings` for the list.

    Returns
    -------
    object
        The decorated object.
    """

    def decorator(wrapped):
        data = get_decorator_data(_get_base(wrapped), set_default=True)
        data.override.setdefault("settings", {}).update(kwargs)
        return wrapped

    return decorator


def filter(value):  # pylint: disable=redefined-builtin
    """Modifier decorator to force the inclusion or exclusion of an attribute.

    This only modifies the behaviour of the :func:`create_patches` function
    and the :func:`patches` decorator, given that their parameter
    ``use_decorators`` is set to ``True``.

    Parameters
    ----------
    value : bool
        ``True`` to force inclusion, ``False`` to force exclusion, and ``None``
        to inherit from the behaviour defined by :func:`create_patches` or
        :func:`patches`.

    Returns
    -------
    object
        The decorated object.
    """

    def decorator(wrapped):
        data = get_decorator_data(_get_base(wrapped), set_default=True)
        data.filter = value
        return wrapped

    return decorator


def find_patches(modules, recursive=True):
    """Find all the patches created through decorators.

    Parameters
    ----------
    modules : list of module
        Modules and/or packages to search the patches in.
    recursive : bool
        ``True`` to search recursively in subpackages.

    Returns
    -------
    list of gorilla.Patch
        Patches found.

    Raises
    ------
    TypeError
        The input is not a valid package or module.

    See Also
    --------
    :func:`patch`, :func:`patches`.
    """
    out = []
    modules = (
        module for package in modules for module in _module_iterator(package, recursive=recursive)
    )
    for module in modules:
        members = _get_members(module, filter=None)
        for _, value in members:
            base = _get_base(value)
            decorator_data = get_decorator_data(base)
            if decorator_data is None:
                continue

            out.extend(decorator_data.patches)

    return out


def get_original_attribute(obj, name, bypass_descriptor_protocol=False):
    """Retrieve an overridden attribute that has been stored.

    Parameters
    ----------
    obj : object
        Object to search the attribute in.
    name : str
        Name of the attribute.
    bypass_descriptor_protocol: boolean
        bypassing descriptor protocol if true. When storing original method during patching or
        restoring original method during reverting patch, we need set bypass_descriptor_protocol
        to be True to ensure get the raw attribute object.

    Returns
    -------
    object
        The attribute found.

    Raises
    ------
    AttributeError
        The attribute couldn't be found.

    Note
    ----
    if setting store_hit=False, then after patch applied, this methods may return patched
    attribute instead of original attribute in specific cases.

    See Also
    --------
    :attr:`Settings.allow_hit`.
    """

    original_name = _ORIGINAL_NAME % (name,)
    curr_active_patch = _ACTIVE_PATCH % (name,)

    def _get_attr(obj_, name_):
        if bypass_descriptor_protocol:
            return object.__getattribute__(obj_, name_)
        else:
            return getattr(obj_, name_)

    no_original_stored_err = (
        "Original attribute %s was not stored when patching, set "
        "store_hit=True will address this."
    )

    if inspect.isclass(obj):
        # Search from children classes to parent classes, and check "original_name" attribute
        # first. This will ensure get the correct original attribute in any cases, e.g.,
        # the case some classes in the hierarchy haven't been patched, but some others are
        # patched, this case the previous code is risky to get wrong original attribute.
        for obj_ in inspect.getmro(obj):
            if original_name in obj_.__dict__:
                return _get_attr(obj_, original_name)
            elif name in obj_.__dict__:
                if curr_active_patch in obj_.__dict__:
                    patch = getattr(obj, curr_active_patch)
                    if patch.is_inplace_patch:
                        raise RuntimeError(no_original_stored_err % (f"{obj_.__name__}.{name}",))
                    else:
                        # non inplace patch, we can get original methods in parent classes.
                        # so go on checking parent classes
                        continue
                return _get_attr(obj_, name)
            else:
                # go on checking parent classes
                continue
        raise AttributeError(f"'{type(obj)}' object has no attribute '{name}'")
    else:
        try:
            return _get_attr(obj, original_name)
        except AttributeError:
            if curr_active_patch in obj.__dict__:
                raise RuntimeError(no_original_stored_err % (f"{type(obj).__name__}.{name}",))
            return _get_attr(obj, name)


def get_decorator_data(obj, set_default=False):
    """Retrieve any decorator data from an object.

    Parameters
    ----------
    obj : object
        Object.
    set_default : bool
        If no data is found, a default one is set on the object and returned,
        otherwise ``None`` is returned.

    Returns
    -------
    gorilla.DecoratorData
        The decorator data or ``None``.
    """
    if inspect.isclass(obj):
        datas = getattr(obj, _DECORATOR_DATA, {})
        data = datas.setdefault(obj, None)
        if data is None and set_default:
            data = DecoratorData()
            datas[obj] = data
            setattr(obj, _DECORATOR_DATA, datas)
    else:
        data = getattr(obj, _DECORATOR_DATA, None)
        if data is None and set_default:
            data = DecoratorData()
            setattr(obj, _DECORATOR_DATA, data)

    return data


def _get_base(obj):
    """Unwrap decorators to retrieve the base object.

    Parameters
    ----------
    obj : object
        Object.

    Returns
    -------
    object
        The base object found or the input object otherwise.
    """
    if hasattr(obj, "__func__"):
        obj = obj.__func__
    elif isinstance(obj, property):
        obj = obj.fget
    elif isinstance(obj, (classmethod, staticmethod)):
        # Fallback for Python < 2.7 back when no `__func__` attribute
        # was defined for those descriptors.
        obj = obj.__get__(None, object)
    else:
        return obj

    return _get_base(obj)


def _get_members(obj, traverse_bases=True, filter=default_filter, recursive=True):
    """Retrieve the member attributes of a module or a class.

    The descriptor protocol is bypassed.

    Parameters
    ----------
    obj : module or class
        Object.
    traverse_bases : bool
        If the object is a class, the base classes are also traversed.
    filter : function
        Attributes for which the function returns ``False`` are skipped. The
        function needs to define two parameters: ``name``, the attribute name,
        and ``obj``, the attribute value. If ``None``, no attribute is skipped.
    recursive : bool
        ``True`` to search recursively through subclasses.

    Returns
    ------
    list of (name, value)
        A list of tuples each containing the name and the value of the
        attribute.
    """
    if filter is None:
        filter = _true

    out = []
    stack = collections.deque((obj,))
    while stack:
        obj = stack.popleft()
        if traverse_bases and inspect.isclass(obj):
            roots = [base for base in inspect.getmro(obj) if base not in (type, object)]
        else:
            roots = [obj]

        members = []
        seen = set()
        for root in roots:
            for name, value in _iteritems(getattr(root, "__dict__", {})):
                if name not in seen and filter(name, value):
                    members.append((name, value))

                seen.add(name)

        members = sorted(members)
        for _, value in members:
            if recursive and inspect.isclass(value):
                stack.append(value)

        out.extend(members)

    return out


def _module_iterator(root, recursive=True):
    """Iterate over modules.

    Parameters
    ----------
    root : module
        Root module or package to iterate from.
    recursive : bool
        ``True`` to iterate within subpackages.

    Yields
    ------
    module
        The modules found.
    """
    yield root

    stack = collections.deque((root,))
    while stack:
        package = stack.popleft()
        # The '__path__' attribute of a package might return a list of paths if
        # the package is referenced as a namespace.
        paths = getattr(package, "__path__", [])
        for path in paths:
            modules = pkgutil.iter_modules([path])
            for finder, name, is_package in modules:
                module_name = f"{package.__name__}.{name}"
                module = sys.modules.get(module_name, None)
                if module is None:
                    # Import the module through the finder to support package
                    # namespaces.
                    module = _load_module(finder, module_name)

                if is_package:
                    if recursive:
                        stack.append(module)
                        yield module
                else:
                    yield module


def _true(*args, **kwargs):  # pylint: disable=unused-argument
    """Return ``True``."""
    return True
