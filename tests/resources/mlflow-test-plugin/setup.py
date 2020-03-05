from setuptools import setup, find_packages


setup(
    name="mflow-test-plugin",
    version="0.0.1",
    description="Test plugin for MLflow",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with MLflow
    install_requires=["mlflow"],
    entry_points={
        # Define a Tracking Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflow.tracking_store": "file-plugin=mlflow_test_plugin:PluginFileStore",
        # Define a ArtifactRepository plugin for artifact URIs with scheme 'file-plugin'
        "mlflow.artifact_repository":
            "file-plugin=mlflow_test_plugin:PluginLocalArtifactRepository",
        # Define a RunContextProvider plugin. The entry point name for run context providers
        # is not used, and so is set to the string "unused" here
        "mlflow.run_context_provider": "unused=mlflow_test_plugin:PluginRunContextProvider",
        # Define a Model Registry Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflow.model_registry_store":
            "file-plugin=mlflow_test_plugin:PluginRegistrySqlAlchemyStore",
    },
)
