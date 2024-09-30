from langchain_core.documents import Document as LangchainDocument

from mlflow.entities import Document


def test_from_langchain_document():
    langchain_document = LangchainDocument(page_content="Hello", metadata={"key": "value"})
    document = Document.from_langchain_document(langchain_document)
    assert document.page_content == "Hello"
    assert document.metadata == {"key": "value"}
    assert document.id is None
