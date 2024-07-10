from typing import Any, Dict, List, Optional, Union

CHAT_ENGINE_NAME = "chat"
QUERY_ENGINE_NAME = "query"
RETRIEVER_ENGINE_NAME = "retriever"
SUPPORTED_ENGINES = {CHAT_ENGINE_NAME, QUERY_ENGINE_NAME, RETRIEVER_ENGINE_NAME}


# NOTE: will add this in the next PR
def create_engine_wrapper(index, engine_type: str, model_config: Optional[Dict[str, Any]] = None):
    class _EngineWrapper:
        def __init__(self, index, engine_type, model_config):
            self.index = index
            self.engine_type = engine_type
            self.model_config = model_config

        def predict(self, data, params: Optional[Dict[str, Any]] = None) -> Union[List[str], str]:
            return "placeholder"

    return _EngineWrapper(index, engine_type, model_config)
