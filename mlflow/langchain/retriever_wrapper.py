"""Chain for wrapping a retriever."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.schema import BaseRetriever, Document
from pydantic import Extra, Field


class RetrieverWrapper(Chain):
    input_key: str = "query"  #: :meta private:
    output_key: str = "source_documents"  #: :meta private:
    retriever: BaseRetriever = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.
        :meta private:
        """
        return [self.output_key]

    def _get_docs(self, question: str) -> List[Document]:
        """Get documents from the retriever."""
        return self.retriever.get_relevant_documents(question)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run _get_docs on input query.
        Returns the retrieved documents under the key 'source_documents'.
        Example:
        .. code-block:: python
        chain = MlflowRetrieverEvaluator(retriever=...)
        res = chain({'query': 'This is my query'})
        docs = res['source_documents']
        """
        question = inputs[self.input_key]
        docs = self._get_docs(question)
        list_of_str_page_context = [doc.page_content for doc in docs]
        return {self.output_key: json.dumps(list_of_str_page_context)}

    async def _aget_docs(self, question: str) -> List[Document]:
        """Get documents from the retriever."""
        return await self.retriever.aget_relevant_documents(question)

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run _get_docs on input query.
        Returns the retrieved documents under the key 'source_documents'.
        Example:
        .. code-block:: python
        chain = MlflowRetrieverEvaluator(retriever=...)
        res = chain({'query': 'This is my query'})
        docs = res['source_documents']
        """
        question = inputs[self.input_key]
        docs = await self._aget_docs(question)
        list_of_str_page_context = [doc.page_content for doc in docs]
        return {self.output_key: json.dumps(list_of_str_page_context)}

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "retriever_wrapper"

    @classmethod
    def load(cls, file: Union[str, Path], **kwargs: Any) -> RetrieverWrapper:
        """Load a RetrieverWrapper from a file."""
        # Convert file to Path object.
        file_path = Path(file) if isinstance(file, str) else file
        # Load from either json or yaml.
        if file_path.suffix == ".json":
            with open(file_path) as f:
                config = json.load(f)
        elif file_path.suffix in (".yaml", ".yml"):
            with open(file_path) as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError("File type must be json or yaml")

        # Override default 'verbose' and 'memory' for the chain
        if "verbose" in kwargs:
            config["verbose"] = kwargs.pop("verbose")
        if "memory" in kwargs:
            config["memory"] = kwargs.pop("memory")

        if "_type" not in config:
            raise ValueError("Must specify a chain Type in config")
        config_type = config.pop("_type")

        if config_type != "retriever_wrapper":
            raise ValueError(f"Loading {config_type} chain not supported")

        retriever = kwargs.pop("retriever", None)
        if retriever is None:
            raise ValueError("`retriever` must be present.")

        return cls(
            retriever=retriever,
            **config,
        )
