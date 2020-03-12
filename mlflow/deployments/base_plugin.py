import sys
import abc


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class BasePlugin(ABC):

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def delete(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def list(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def describe(self, *args, **kwargs):
        pass
