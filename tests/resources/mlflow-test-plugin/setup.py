from setuptools import setup, find_packages


setup(
    name="mlflow-test-plugin",
    version="0.0.1",
    description="Test plugin for MLflow",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with MLflow
    install_requires=["mlflow"],
    entry_points={
        # Define a Tracking Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflow.tracking_store": "file-plugin=mlflow_test_plugin.file_store:PluginFileStore",
        # Define a ArtifactRepository plugin for artifact URIs with scheme 'file-plugin'
        "mlflow.artifact_repository": "file-plugin=mlflow_test_plugin.local_artifact:PluginLocalArtifactRepository",  # pylint: disable=line-too-long
        # Define a RunContextProvider plugin. The entry point name for run context providers
        # is not used, and so is set to the string "unused" here
        "mlflow.run_context_provider": "unused=mlflow_test_plugin.run_context_provider:PluginRunContextProvider",  # pylint: disable=line-too-long
        # Define a DefaultExperimentProvider plugin. The entry point name for
        # default experiment providers is not used, and so is set to the string "unused" here
        "mlflow.default_experiment_provider": "unused=mlflow_test_plugin.default_experiment_provider:PluginDefaultExperimentProvider",  # pylint: disable=line-too-long
        # Define a RequestHeaderProvider plugin. The entry point name for request header providers
        # is not used, and so is set to the string "unused" here
        "mlflow.request_header_provider": "unused=mlflow_test_plugin.request_header_provider:PluginRequestHeaderProvider",  # pylint: disable=line-too-long
        # Define a Model Registry Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflow.model_registry_store": "file-plugin=mlflow_test_plugin.sqlalchemy_store:PluginRegistrySqlAlchemyStore",  # pylint: disable=line-too-long
        # Define a MLflow Project Backend plugin called 'dummy-backend'
        "mlflow.project_backend": "dummy-backend=mlflow_test_plugin.dummy_backend:PluginDummyProjectBackend",  # pylint: disable=line-too-long
        # Define a MLflow model deployment plugin for target 'faketarget'
        "mlflow.deployments": "faketarget=mlflow_test_plugin.fake_deployment_plugin",
        # Define a Mlflow model evaluator with name "dummy_evaluator"
        "mlflow.model_evaluator": "dummy_evaluator=mlflow_test_plugin.dummy_evaluator:DummyEvaluator",  # pylint: disable=line-too-long
    },
)
