from abc import ABCMeta, abstractmethod


class FlavorBackend:
    """
    Abstract class for Flavor Backend.
    This class defines the API interface for local model deployment of MLflow model flavors.
    """

    __metaclass__ = ABCMeta

    def __init__(self, config, **kwargs):  # pylint: disable=unused-argument
        self._config = config

    @abstractmethod
    def predict(self, model_uri, input_path, output_path, content_type, json_format):
        """
        Generate predictions using a saved MLflow model referenced by the given URI.
        Input and output are read from and written to a file or stdin / stdout.

        :param model_uri: URI pointing to the MLflow model to be used for scoring.
        :param input_path: Path to the file with input data. If not specified, data is read from
                           stdin.
        :param output_path: Path to the file with output predictions. If not specified, data is
                            written to stdout.
        :param content_type: Specifies the input format. Can be one of {``json``, ``csv``}
        :param json_format: Only applies if ``content_type == json``. Specifies how is the input
                            data encoded in json. Can be one of {``split``, ``records``} mirroring
                            the behavior of Pandas orient attribute. The default is ``split`` which
                            expects dict like data: ``{'index' -> [index], 'columns' -> [columns],
                            'data' -> [values]}``, where index is optional.
                            For more information see
                            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html
        """
        pass

    @abstractmethod
    def serve(self, model_uri, port, host, enable_mlserver):
        """
        Serve the specified MLflow model locally.

        :param model_uri: URI pointing to the MLflow model to be used for scoring.
        :param port: Port to use for the model deployment.
        :param host: Host to use for the model deployment. Defaults to ``localhost``.
        :param enable_mlserver: Whether to use MLServer or the local scoring server.
        """
        pass

    def prepare_env(self, model_uri):
        """
        Performs any preparation necessary to predict or serve the model, for example
        downloading dependencies or initializing a conda environment. After preparation,
        calling predict or serve should be fast.
        """
        pass

    @abstractmethod
    def can_score_model(self):
        """
        Check whether this flavor backend can be deployed in the current environment.

        :return: True if this flavor backend can be applied in the current environment.
        """
        pass

    def can_build_image(self):
        """
        :return: True if this flavor has a `build_image` method defined for building a docker
                 container capable of serving the model, False otherwise.
        """
        return callable(getattr(self.__class__, "build_image", None))
