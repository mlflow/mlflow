from langchain_core.documents import Document as LangchainDocument
from llama_index.core.schema import NodeWithScore, TextNode

from mlflow.entities import Document


def test_from_langchain_document():
    langchain_document = LangchainDocument(page_content="Hello", metadata={"key": "value"})
    document = Document.from_langchain_document(langchain_document)
    assert document.page_content == "Hello"
    assert document.metadata == {"key": "value"}
    assert document.id is None


def test_from_llama_index_node_with_score():
    text_node = TextNode(text="Hello", metadata={"key": "value"})
    node_with_score = NodeWithScore(node=text_node, score=0.5)
    document = Document.from_llama_index_node_with_score(node_with_score)
    assert document.page_content == "Hello"
    assert document.metadata == {"score": 0.5, "key": "value"}
    assert document.id == node_with_score.node_id
