#                     __ __ __
#   .-----.-----.----|__|  |  .---.-.
#   |  _  |  _  |   _|  |  |  |  _  |
#   |___  |_____|__| |__|__|__|___._|
#   |_____|
#

"""
NOTE: The contents of this file have been inlined from the gorilla package's source code
https://github.com/christophercrouzet/gorilla/blob/v0.3.0/gorilla.py
The following modifications have been made:
    - When a patch is applied to a method `foo` on parent class `A` and a subsequent
      patch is applied to `foo` on one of its children `B`, `get_original_attribute(B, "foo")`
      will now refer to `B.foo`, rather than `A.foo`.

gorilla
~~~~~~~

Convenient approach to monkey patching.

:copyright: Copyright 2014-2017 by Christopher Crouzet.
:license: MIT, see LICENSE for details.
"""

import collections
import copy
import inspect
import pkgutil
import sys
import types


__version__ = "0.3.0"


if sys.version_info[0] == 2:
    _CLASS_TYPES = (type, types.ClassType)

    def _iteritems(d, **kwargs):
        return d.iteritems(**kwargs)

    def _load_module(finder, name):
        loader = finder.find_module(name)
        return loader.load_module(name)


else:
    _CLASS_TYPES = (type,)

    def _iteritems(d, **kwargs):
        return iter(d.items(**kwargs))

    def _load_module(finder, name):
        loader, _ = finder.find_loader(name)
        return loader.load_module()


# Pattern for each internal attribute name.
_PATTERN = "_gorilla_%s"

# Pattern for the name of the overidden attributes to be stored.
_ORIGINAL_NAME = _PATTERN % ("original_%s",)

_ACTIVE_PATCH = "_gorilla_active_patch"

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


class DecoratorData(object):

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


class Settings(object):

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
        values = ", ".join(
            ["%s=%r" % (key, value) for key, value in sorted(_iteritems(self.__dict__))]
        )
        return "%s(%s)" % (type(self).__name__, values)

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


class Patch(object):

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

    def __repr__(self):
        return "%s(destination=%r, name=%r, obj=%r, settings=%r)" % (
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
    settings = Settings() if patch.settings is None else patch.settings

    # When a hit occurs due to an attribute at the destination already existing
    # with the patch's name, the existing attribute is referred to as 'target'.
    try:
        target = get_attribute(patch.destination, patch.name)
    except AttributeError:
        pass
    else:
        if not settings.allow_hit:
            raise RuntimeError(
                "An attribute named '%s' already exists at the destination "
                "'%s'. Set a different name through the patch object to avoid "
                "a name clash or set the setting 'allow_hit' to True to "
                "overwrite the attribute. In the latter case, it is "
                "recommended to also set the 'store_hit' setting to True in "
                "order to store the original attribute under a different "
                "name so it can still be accessed." % (patch.name, patch.destination.__name__)
            )

        if settings.store_hit:
            original_name = _ORIGINAL_NAME % (patch.name,)
            # For certain MLflow Models use cases, such as scikit-learn autologging, we patch
            # a method on a parent class
            # (e.g., `sklearn.feature_extraction.text.CountVectorizer.fit_transform()`) and
            # subsequently patch a corresponding overridden method on one of its children
            # (e.g., `feature_extraction.text.TfidfVectorizer.fit_transform()`)
            # to provide a different implementation. In these cases, we wish to swap out the
            # "original function" attribute stored on the child in order to refer to the child's
            # overriden method (e.g., `feature_extraction.text.TfidfVectorizer.fit_transform()`),
            # rather than the parent method
            # (e.g., `sklearn.feature_extraction.text.CountVectorizer.fit_transform()`)
            prev_patch = getattr(patch.destination, _ACTIVE_PATCH, None)
            if not hasattr(patch.destination, original_name) or (
                prev_patch
                and prev_patch.destination != patch.destination
                and issubclass(patch.destination, prev_patch.destination)
            ):
                setattr(patch.destination, original_name, target)

    setattr(patch.destination, patch.name, patch.obj)
    setattr(patch.destination, _ACTIVE_PATCH, patch)


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


def patches(
    destination,
    settings=None,
    traverse_bases=True,
    filter=default_filter,
    recursive=True,
    use_decorators=True,
):
    """Decorator to create a patch for each member of a module or a class.

    Parameters
    ----------
    destination : object
        Patch destination.
    settings : gorilla.Settings
        Settings.
    traverse_bases : bool
        If the object is a class, the base classes are also traversed.
    filter : function
        Attributes for which the function returns ``False`` are skipped. The
        function needs to define two parameters: ``name``, the attribute name,
        and ``obj``, the attribute value. If ``None``, no attribute is skipped.
    recursive : bool
        If ``True``, and a hit occurs due to an attribute at the destination
        already existing with the given name, and both the member and the
        target attributes are classes, then instead of creating a patch
        directly with the member attribute value as is, a patch for each of its
        own members is created with the target as new destination.
    use_decorators : bool
        Allows to take any modifier decorator into consideration to allow for
        more granular customizations.

    Returns
    -------
    object
        The decorated object.

    Note
    ----
    A 'target' differs from a 'destination' in that a target represents an
    existing attribute at the destination about to be hit by a patch.

    See Also
    --------
    :class:`Patch`, :func:`create_patches`.
    """

    def decorator(wrapped):
        settings_ = copy.deepcopy(settings)
        patches = create_patches(
            destination,
            wrapped,
            settings=settings_,
            traverse_bases=traverse_bases,
            filter=filter,
            recursive=recursive,
            use_decorators=use_decorators,
        )
        data = get_decorator_data(_get_base(wrapped), set_default=True)
        data.patches.extend(patches)
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


def filter(value):  # pylint: disable=W0622
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


def create_patches(
    destination,
    root,
    settings=None,
    traverse_bases=True,
    filter=default_filter,
    recursive=True,
    use_decorators=True,
):
    """Create a patch for each member of a module or a class.

    Parameters
    ----------
    destination : object
        Patch destination.
    root : object
        Root object, either a module or a class.
    settings : gorilla.Settings
        Settings.
    traverse_bases : bool
        If the object is a class, the base classes are also traversed.
    filter : function
        Attributes for which the function returns ``False`` are skipped. The
        function needs to define two parameters: ``name``, the attribute name,
        and ``obj``, the attribute value. If ``None``, no attribute is skipped.
    recursive : bool
        If ``True``, and a hit occurs due to an attribute at the destination
        already existing with the given name, and both the member and the
        target attributes are classes, then instead of creating a patch
        directly with the member attribute value as is, a patch for each of its
        own members is created with the target as new destination.
    use_decorators : bool
        ``True`` to take any modifier decorator into consideration to allow for
        more granular customizations.

    Returns
    -------
    list of gorilla.Patch
        The patches.

    Note
    ----
    A 'target' differs from a 'destination' in that a target represents an
    existing attribute at the destination about to be hit by a patch.

    See Also
    --------
    :func:`patches`.
    """
    if filter is None:
        filter = _true

    out = []
    root_patch = Patch(destination, "", root, settings=settings)
    stack = collections.deque((root_patch,))
    while stack:
        parent_patch = stack.popleft()
        members = _get_members(
            parent_patch.obj, traverse_bases=traverse_bases, filter=None, recursive=False
        )
        for name, value in members:
            patch = Patch(
                parent_patch.destination, name, value, settings=copy.deepcopy(parent_patch.settings)
            )
            if use_decorators:
                base = _get_base(value)
                decorator_data = get_decorator_data(base)
                filter_override = None if decorator_data is None else decorator_data.filter
                if (
                    filter_override is None and not filter(name, value)
                ) or filter_override is False:
                    continue

                if decorator_data is not None:
                    patch._update(**decorator_data.override)
            elif not filter(name, value):
                continue

            if recursive and isinstance(value, _CLASS_TYPES):
                try:
                    target = get_attribute(patch.destination, patch.name)
                except AttributeError:
                    pass
                else:
                    if isinstance(target, _CLASS_TYPES):
                        patch.destination = target
                        stack.append(patch)
                        continue

            out.append(patch)

    return out


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


def get_attribute(obj, name):
    """Retrieve an attribute while bypassing the descriptor protocol.

    As per the built-in |getattr()|_ function, if the input object is a class
    then its base classes might also be searched until the attribute is found.

    Parameters
    ----------
    obj : object
        Object to search the attribute in.
    name : str
        Name of the attribute.

    Returns
    -------
    object
        The attribute found.

    Raises
    ------
    AttributeError
        The attribute couldn't be found.


    .. |getattr()| replace:: ``getattr()``
    .. _getattr(): https://docs.python.org/library/functions.html#getattr
    """
    objs = inspect.getmro(obj) if isinstance(obj, _CLASS_TYPES) else [obj]
    for obj_ in objs:
        try:
            return object.__getattribute__(obj_, name)
        except AttributeError:
            pass

    raise AttributeError("'%s' object has no attribute '%s'" % (type(obj), name))


def get_original_attribute(obj, name):
    """Retrieve an overriden attribute that has been stored.

    Parameters
    ----------
    obj : object
        Object to search the attribute in.
    name : str
        Name of the attribute.

    Returns
    -------
    object
        The attribute found.

    Raises
    ------
    AttributeError
        The attribute couldn't be found.

    See Also
    --------
    :attr:`Settings.allow_hit`.
    """
    return getattr(obj, _ORIGINAL_NAME % (name,))


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
    if isinstance(obj, _CLASS_TYPES):
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
        if traverse_bases and isinstance(obj, _CLASS_TYPES):
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
            if recursive and isinstance(value, _CLASS_TYPES):
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
                module_name = "%s.%s" % (package.__name__, name)
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
