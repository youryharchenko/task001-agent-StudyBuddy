import chromadb
from langchain_core.tools import tool

chroma_client = chromadb.PersistentClient(path="./chroma_db")
knowledge_base = chroma_client.get_collection("domain_knowledge")


@tool("search_info")
def search_info(query: str) -> str:
    """Пошук інформації у базі знань.

    Використовуйте цей інструмент,
    коли потрібна інформація з предметної області.

    Args:
        query: Пошуковий запит.

    Returns:
        Топ-3 релевантних документів з бази знань.
    """

    results = knowledge_base.query(query_texts=[query], n_results=3)
    if results["documents"]:
        docs = results["documents"][0]
        return "\n---\n".join(docs)
    else:
        return "не знайдено релевантних документів"
