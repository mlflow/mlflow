from setuptools import setup, find_packages


setup(
    name="mlflux-test-plugin",
    version="0.0.1",
    description="Test plugin for mlflux",
    packages=find_packages(),
    # Require mlflux as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with mlflux
    install_requires=["mlflux"],
    entry_points={
        # Define a Tracking Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflux.tracking_store": "file-plugin=mlflow_test_plugin.file_store:PluginFileStore",
        # Define a ArtifactRepository plugin for artifact URIs with scheme 'file-plugin'
        "mlflux.artifact_repository": "file-plugin=mlflow_test_plugin.local_artifact:PluginLocalArtifactRepository",  # noqa
        # Define a RunContextProvider plugin. The entry point name for run context providers
        # is not used, and so is set to the string "unused" here
        "mlflux.run_context_provider": "unused=mlflow_test_plugin.run_context_provider:PluginRunContextProvider",  # noqa
        # Define a RequestHeaderProvider plugin. The entry point name for request header providers
        # is not used, and so is set to the string "unused" here
        "mlflux.request_header_provider": "unused=mlflow_test_plugin.request_header_provider:PluginRequestHeaderProvider",  # noqa
        # Define a Model Registry Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflux.model_registry_store": "file-plugin=mlflow_test_plugin.sqlalchemy_store:PluginRegistrySqlAlchemyStore",  # noqa
        # Define a mlflux Project Backend plugin called 'dummy-backend'
        "mlflux.project_backend": "dummy-backend=mlflow_test_plugin.dummy_backend:PluginDummyProjectBackend",  # noqa
        # Define a mlflux model deployment plugin for target 'faketarget'
        "mlflux.deployments": "faketarget=mlflow_test_plugin.fake_deployment_plugin",
    },
)
