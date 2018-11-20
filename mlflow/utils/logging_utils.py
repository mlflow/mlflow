import sys
import logging
import logging.config

import mlflow


LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

DEFAULT_LOGGER_NAME = mlflow.__name__


def _configure_default_logger():
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
