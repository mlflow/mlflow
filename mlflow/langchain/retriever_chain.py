"""Chain for wrapping a retriever."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ConfigDict, Field

from mlflow.langchain._compat import (
    import_async_callback_manager_for_chain_run,
    import_base_retriever,
    import_callback_manager_for_chain_run,
    import_document,
    try_import_chain,
)

AsyncCallbackManagerForChainRun = import_async_callback_manager_for_chain_run()
CallbackManagerForChainRun = import_callback_manager_for_chain_run()
BaseRetriever = import_base_retriever()
Document = import_document()
Chain = try_import_chain()

if Chain is None:
    raise ImportError(
        "Chain class not found. MLflow's retriever_chain functionality requires langchain<1.0.0. "
        "For langchain 1.0.0+, please use LangGraph instead."
    )


class _RetrieverChain(Chain):
    """
    Chain that wraps a retriever for use with MLflow.

    The MLflow ``langchain`` flavor provides the functionality to log a retriever object and
    evaluate it individually. This is useful if you want to evaluate the quality of the
    relevant documents returned by a retriever object without directing these documents
    through a large language model (LLM) to yield a summarized response.

    In order to log the retriever object in the ``langchain`` flavor, the retriever object
    needs to be wrapped within a ``_RetrieverChain``.

    See ``examples/langchain/retriever_chain.py`` for how to log the ``_RetrieverChain``.

    Args:
        retriever: The retriever to wrap.
    """

    input_key: str = "query"
    output_key: str = "source_documents"
    retriever: BaseRetriever = Field(exclude=True)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys."""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys."""
        return [self.output_key]

    def _get_docs(self, question: str) -> list[Document]:
        """Get documents from the retriever."""
        return self.retriever.get_relevant_documents(question)

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run _get_docs on input query.
        Returns the retrieved documents under the key 'source_documents'.

        Example:

        .. code-block:: python

            chain = _RetrieverChain(retriever=...)
            res = chain({"query": "This is my query"})
            docs = res["source_documents"]
        """
        question = inputs[self.input_key]
        docs = self._get_docs(question)
        list_of_str_page_content = [doc.page_content for doc in docs]
        return {self.output_key: json.dumps(list_of_str_page_content)}

    async def _aget_docs(self, question: str) -> list[Document]:
        """Get documents from the retriever."""
        return await self.retriever.aget_relevant_documents(question)

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run _get_docs on input query.
        Returns the retrieved documents under the key 'source_documents'.

        Example:

        .. code-block:: python

            chain = _RetrieverChain(retriever=...)
            res = chain({"query": "This is my query"})
            docs = res["source_documents"]
        """
        question = inputs[self.input_key]
        docs = await self._aget_docs(question)
        list_of_str_page_content = [doc.page_content for doc in docs]
        return {self.output_key: json.dumps(list_of_str_page_content)}

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "retriever_chain"

    @classmethod
    def load(cls, file: str | Path, **kwargs: Any) -> _RetrieverChain:
        """Load a _RetrieverChain from a file."""
        # Convert file to Path object.
        file_path = Path(file) if isinstance(file, str) else file
        # Load from either json or yaml.
        if file_path.suffix == ".json":
            with open(file_path) as f:
                config = json.load(f)
        elif file_path.suffix in (".yaml", ".yml"):
            with open(file_path) as f:
                # This is to ignore certain tags that are not supported
                # with pydantic >= 2.0
                yaml.add_multi_constructor(
                    "tag:yaml.org,2002:python/object",
                    lambda loader, suffix, node: None,
                    Loader=yaml.SafeLoader,
                )
                config = yaml.load(f, yaml.SafeLoader)
        else:
            raise ValueError("File type must be json or yaml")

        # Override default 'verbose' and 'memory' for the chain
        if verbose := kwargs.pop("verbose", None):
            config["verbose"] = verbose
        if memory := kwargs.pop("memory", None):
            config["memory"] = memory

        if "_type" not in config:
            raise ValueError("Must specify a chain Type in config")
        config_type = config.pop("_type")

        if config_type != "retriever_chain":
            raise ValueError(f"Loading {config_type} chain not supported")

        retriever = kwargs.pop("retriever", None)
        if retriever is None:
            raise ValueError("`retriever` must be present.")

        config.pop("retriever", None)

        return cls(
            retriever=retriever,
            **config,
        )
