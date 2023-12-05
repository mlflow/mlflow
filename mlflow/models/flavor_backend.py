from abc import ABCMeta, abstractmethod

from mlflow.utils.annotations import developer_stable


@developer_stable
class FlavorBackend(metaclass=ABCMeta):
    """
    Abstract class for Flavor Backend.
    This class defines the API interface for local model deployment of MLflow model flavors.
    """

    def __init__(self, config, **kwargs):  # pylint: disable=unused-argument
        self._config = config

    @abstractmethod
    def predict(self, model_uri, input_path, output_path, content_type):
        """
        Generate predictions using a saved MLflow model referenced by the given URI.
        Input and output are read from and written to a file or stdin / stdout.

        :param model_uri: URI pointing to the MLflow model to be used for scoring.
        :param input_path: Path to the file with input data. If not specified, data is read from
                           stdin.
        :param output_path: Path to the file with output predictions. If not specified, data is
                            written to stdout.
        :param content_type: Specifies the input format. Can be one of {``json``, ``csv``}
        """
        pass

    @abstractmethod
    def serve(
        self,
        model_uri,
        port,
        host,
        timeout,
        enable_mlserver,
        synchronous=True,
        stdout=None,
        stderr=None,
    ):
        """
        Serve the specified MLflow model locally.

        :param model_uri: URI pointing to the MLflow model to be used for scoring.
        :param port: Port to use for the model deployment.
        :param host: Host to use for the model deployment. Defaults to ``localhost``.
        :param timeout: Timeout in seconds to serve a request. Defaults to 60.
        :param enable_mlserver: Whether to use MLServer or the local scoring server.
        :param synchronous: If True, wait until server process exit and return 0, if process exit
                            with non-zero return code, raise exception.
                            If False, return the server process `Popen` instance immediately.
        :param stdout: Redirect server stdout
        :param stderr: Redirect server stderr
        """
        pass

    def prepare_env(self, model_uri, capture_output=False):
        """
        Performs any preparation necessary to predict or serve the model, for example
        downloading dependencies or initializing a conda environment. After preparation,
        calling predict or serve should be fast.
        """
        pass

    @abstractmethod
    def build_image(self, model_uri, image_name, install_mlflow, mlflow_home, enable_mlserver):
        raise NotImplementedError

    @abstractmethod
    def generate_dockerfile(
        self, model_uri, output_path, install_mlflow, mlflow_home, enable_mlserver
    ):
        raise NotImplementedError

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
