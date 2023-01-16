from mlflow.tracking.context.abstract_context import RunContextProvider


class PluginRunContextProvider(RunContextProvider):
    """RunContextProvider provided through plugin system"""

    def in_context(self):
        return False

    def tags(self):
        return {"test": "tag"}
