from abc import ABCMeta, abstractmethod


class RunContextProvider(object):
    """
    Abstract base class for context provider objects specifying custom tags at run-creation time
    (e.g. tags specifying the git repo with which the run is associated).

    When a run is created via the fluent ``mlflux.start_run`` method, mlflux iterates through all
    registered RunContextProviders. For each context provider where ``in_context`` returns ``True``,
    mlflux calls the ``tags`` method on the context provider to compute context tags for the run.
    All context tags are then merged together and set on the newly-created run.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def in_context(self):
        """
        Determine if mlflux is running in this context.

        :return: bool indicating if in this context
        """
        pass

    @abstractmethod
    def tags(self):
        """
        Generate context-specific tags.

        :return: dict of tags
        """
        pass
