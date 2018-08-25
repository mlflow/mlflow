from abc import abstractmethod
import pprint


class _MLflowObject(object):
    def __iter__(self):
        # Iterate through list of properties and yield as key -> value
        for prop in self._properties():
            yield prop, self.__getattribute__(prop)

    @classmethod
    @abstractmethod
    def _properties(cls):
        pass

    @classmethod
    @abstractmethod
    def from_proto(cls, proto):
        pass

    @classmethod
    def from_dictionary(cls, the_dict):
        return cls(**the_dict)

    def __repr__(self):
        return serialize(self)


def serialize(obj):
    return _MLflowObjectPrinter().serialize(obj)


def get_classname(obj):
    return obj.__module__ + "." + obj.__class__.__qualname__


class _MLflowObjectPrinter(object):
    _MAX_LIST_LEN = 2

    def __init__(self):
        super(_MLflowObjectPrinter, self).__init__()
        self.printer = pprint.PrettyPrinter()

    def serialize(self, obj):
        if isinstance(obj, _MLflowObject):
            return "<%s: %s>" % (get_classname(obj), self.serialize_entity(obj))
        # Handle nested lists inside MLflow entities (e.g. lists of metrics/params)
        if isinstance(obj, list):
            res = [serialize(elem) for elem in obj[:self._MAX_LIST_LEN]]
            if len(obj) > self._MAX_LIST_LEN:
                res.append("...")
            return "[%s]" % ", ".join(res)
        return self.printer.pformat(obj)

    def serialize_entity(self, entity):
        return ", ".join(["%s=%s" % (self.serialize(key), self.serialize(value))
                          for key, value in sorted(dict(entity).items())])
