from setuptools import setup, find_packages


setup(
    name="mlflow-test-plugin",
    version="0.0.1",
    description="Test plugin for MLflow",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with MLflow
    install_requires=[],  # Remove mlflow dependency, it can cause tests to run against pypi
    entry_points={
        # Define a Tracking Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflow.tracking_store": "file-plugin=mlflow_test_plugin.file_store:PluginFileStore",
        # Define a ArtifactRepository plugin for artifact URIs with scheme 'file-plugin'
        "mlflow.artifact_repository": "file-plugin=mlflow_test_plugin.local_artifact:PluginLocalArtifactRepository",  # noqa
        # Define a RunContextProvider plugin. The entry point name for run context providers
        # is not used, and so is set to the string "unused" here
        "mlflow.run_context_provider": "unused=mlflow_test_plugin.run_context_provider:PluginRunContextProvider",  # noqa
        # Define a Model Registry Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflow.model_registry_store": "file-plugin=mlflow_test_plugin.sqlalchemy_store:PluginRegistrySqlAlchemyStore",  # noqa
        # Define a MLflow Project Backend plugin called 'dummy-backend'
        "mlflow.project_backend": "dummy-backend=mlflow_test_plugin.dummy_backend:PluginDummyProjectBackend",  # noqa
        # Define a MLflow model deployment plugin for target 'faketarget'
        "mlflow.deployments": "faketarget=mlflow_test_plugin.fake_deployment_plugin",
    },
)
