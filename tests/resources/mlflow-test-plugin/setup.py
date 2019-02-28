from setuptools import setup, find_packages


setup(
    name="mflow-test-plugin",
    version="0.0.1",
    description="Test plugin for MLflow",
    packages=find_packages(),
    install_requires=["mlflow"],
    entry_points={
        "mlflow.tracking_store": "file-plugin=mlflow_test_plugin:PluginFileStore",
        "mlflow.artifact_repository":
            "file-plugin=mlflow_test_plugin:PluginLocalArtifactRepository",
    },
)
