import sys
import logging
import logging.config

import mlflow


LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

DEFAULT_LOGGER_NAME = mlflow.__name__ 


def _configure_default_logger():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': LOGGING_LINE_FORMAT,
                'datefmt': LOGGING_DATETIME_FORMAT,
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': sys.stderr,
            },
        },
        'loggers': {
            DEFAULT_LOGGER_NAME : {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': False, 
            },
        },
    })
