import sys

import logging.config


LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


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
            '': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': True
            },
        },
    })
