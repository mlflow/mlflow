import logging
from dataclasses import dataclass
from typing import Any, Optional

from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

_logger = logging.getLogger(__name__)


@dataclass
class AutoLoggingConfig:
    """
    A dataclass to hold common autologging configuration options.
    """

    log_input_examples: bool
    log_model_signatures: bool
    log_traces: bool
    extra_tags: Optional[dict[str, Any]] = None
    log_models: bool = True

    @classmethod
    def init(cls, flavor_name: str):
        config_dict = AUTOLOGGING_INTEGRATIONS.get(flavor_name, {})
        # NB: These defaults are only used when the autolog() function for the
        # flavor does not specify the corresponding configuration option
        return cls(
            log_models=config_dict.get("log_models", False),
            log_input_examples=config_dict.get("log_input_examples", False),
            log_model_signatures=config_dict.get("log_model_signatures", False),
            log_traces=config_dict.get("log_traces", True),
            extra_tags=config_dict.get("extra_tags", None),
        )
