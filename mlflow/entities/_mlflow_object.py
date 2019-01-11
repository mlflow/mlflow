from abc import abstractmethod
import pprint


class _MLflowObject(object):
    def __iter__(self):
        # Iterate through list of properties and yield as key -> value
        for prop in self._properties():
            yield prop, self.__getattribute__(prop)

    @classmethod
    def _properties(cls):
        return sorted([p for p in cls.__dict__ if isinstance(getattr(cls, p), property)])

    @classmethod
    @abstractmethod
    def from_proto(cls, proto):
        pass

    @classmethod
    def from_dictionary(cls, the_dict):
        filtered_dict = {key: value for key, value in the_dict.items() if key in cls._properties()}
        return cls(**filtered_dict)

    def __repr__(self):
        return to_string(self)


def to_string(obj):
    return _MLflowObjectPrinter().to_string(obj)


def get_classname(obj):
    return type(obj).__name__


class _MLflowObjectPrinter(object):
    _MAX_LIST_LEN = 2

    def __init__(self):
        super(_MLflowObjectPrinter, self).__init__()
        self.printer = pprint.PrettyPrinter()

    def to_string(self, obj):
        if isinstance(obj, _MLflowObject):
            return "<%s: %s>" % (get_classname(obj), self._entity_to_string(obj))
        # Handle nested lists inside MLflow entities (e.g. lists of metrics/params)
        if isinstance(obj, list):
            res = [self.to_string(elem) for elem in obj[:self._MAX_LIST_LEN]]
            if len(obj) > self._MAX_LIST_LEN:
                res.append("...")
            return "[%s]" % ", ".join(res)
        return self.printer.pformat(obj)

    def _entity_to_string(self, entity):
        return ", ".join(["%s=%s" % (key, self.to_string(value)) for key, value in entity])
