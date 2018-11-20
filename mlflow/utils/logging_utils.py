from __future__ import print_function

import sys
import logging
import logging.config

import mlflow

# Logging format example:
# 2018/11/20 12:36:37 INFO mlflow.sagemaker: Creating new SageMaker endpoint
LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

DEFAULT_LOGGER_NAME = mlflow.__name__


def _configure_mlflow_loggers():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'mlflow_formatter': {
                'format': LOGGING_LINE_FORMAT,
                'datefmt': LOGGING_DATETIME_FORMAT,
            },
        },
        'handlers': {
            'mlflow_handler': {
                'level': 'INFO',
                'formatter': 'mlflow_formatter',
                'class': 'logging.StreamHandler',
                'stream': sys.stderr,
            },
        },
        'loggers': {
            DEFAULT_LOGGER_NAME: {
                'handlers': ['mlflow_handler'],
                'level': 'INFO',
                'propagate': False,
            },
        },
    })


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
