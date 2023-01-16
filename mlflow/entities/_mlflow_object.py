from abc import abstractmethod
import pprint


class _MLflowObject:
    def __iter__(self):
        # Iterate through list of properties and yield as key -> value
        for prop in self._properties():
            yield prop, self.__getattribute__(prop)

    @classmethod
    def _get_properties_helper(cls):
        return sorted([p for p in cls.__dict__ if isinstance(getattr(cls, p), property)])

    @classmethod
    def _properties(cls):
        return cls._get_properties_helper()

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


class _MLflowObjectPrinter:
    def __init__(self):
        super().__init__()
        self.printer = pprint.PrettyPrinter()

    def to_string(self, obj):
        if isinstance(obj, _MLflowObject):
            return "<{}: {}>".format(get_classname(obj), self._entity_to_string(obj))
        return self.printer.pformat(obj)

    def _entity_to_string(self, entity):
        return ", ".join(["{}={}".format(key, self.to_string(value)) for key, value in entity])
