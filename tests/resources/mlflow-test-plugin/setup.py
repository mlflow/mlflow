from setuptools import find_packages, setup

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
        "mlflow.artifact_repository": (
            "file-plugin=mlflow_test_plugin.local_artifact:PluginLocalArtifactRepository"
        ),
        # Define a RunContextProvider plugin. The entry point name for run context providers
        # is not used, and so is set to the string "unused" here
        "mlflow.run_context_provider": (
            "unused=mlflow_test_plugin.run_context_provider:PluginRunContextProvider"
        ),
        # Define a DefaultExperimentProvider plugin. The entry point name for
        # default experiment providers is not used, and so is set to the string "unused" here
        "mlflow.default_experiment_provider": (
            "unused=mlflow_test_plugin.default_experiment_provider:PluginDefaultExperimentProvider"
        ),
        # Define a RequestHeaderProvider plugin. The entry point name for request header providers
        # is not used, and so is set to the string "unused" here
        "mlflow.request_header_provider": (
            "unused=mlflow_test_plugin.request_header_provider:PluginRequestHeaderProvider"
        ),
        # Define a RequestAuthProvider plugin. The entry point name for request auth providers
        # is not used, and so is set to the string "unused" here
        "mlflow.request_auth_provider": (
            "unused=mlflow_test_plugin.request_auth_provider:PluginRequestAuthProvider"
        ),
        # Define a Model Registry Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflow.model_registry_store": (
            "file-plugin=mlflow_test_plugin.sqlalchemy_store:PluginRegistrySqlAlchemyStore"
        ),
        # Define a MLflow Project Backend plugin called 'dummy-backend'
        "mlflow.project_backend": (
            "dummy-backend=mlflow_test_plugin.dummy_backend:PluginDummyProjectBackend"
        ),
        # Define a MLflow model deployment plugin for target 'faketarget'
        "mlflow.deployments": "faketarget=mlflow_test_plugin.fake_deployment_plugin",
        # Define a MLflow model evaluator with name "dummy_evaluator"
        "mlflow.model_evaluator": (
            "dummy_evaluator=mlflow_test_plugin.dummy_evaluator:DummyEvaluator"
        ),
        # Define a custom MLflow application with name custom_app
        "mlflow.app": "custom_app=mlflow_test_plugin.app:custom_app",
        # Define an MLflow dataset source called "dummy_source"
        "mlflow.dataset_source": (
            "dummy_source=mlflow_test_plugin.dummy_dataset_source:DummyDatasetSource"
        ),
        # Define an MLflow dataset constructor called "from_dummy"
        "mlflow.dataset_constructor": "from_dummy=mlflow_test_plugin.dummy_dataset:from_dummy",
    },
)
