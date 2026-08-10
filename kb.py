import chromadb
from langchain_core.tools import tool

chroma_client = chromadb.Client()
knowledge_base = chroma_client.create_collection("domain_knowledge")

documents = []

doc_ids = [f"doc_{i}" for i in range(len(documents))]
knowledge_base.add(documents=documents, ids=doc_ids)


@tool("search_templates")
def search_templates(query: str) -> str:
    """Пошук інформації у базі знань.

    Використовуйте цей інструмент,
    коли потрібна інформація з предметної області.

    Args:
        query: Пошуковий запит.

    Returns:
        Топ-3 релевантних документів з бази знань.
    """

    results = knowledge_base.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    return "\n---\n".join(docs)
