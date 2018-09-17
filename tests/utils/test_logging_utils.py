import mlflow.logging


def test_logging_methods():
    mlflow.logging.debug("HI!")
    mlflow.logging.info("HI!")
    mlflow.logging.warn("HI!")
    mlflow.logging.error("HI!")
    mlflow.logging.fatal("HI!")


def test_set_verbosity():
    verbosities = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
    for verbosity in verbosities:
        mlflow.logging.set_verbosity(verbosity)
