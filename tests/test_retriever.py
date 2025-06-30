# tests/test_retriever.py
def test_retriever_precision(rag_service):
    retriever = rag_service._smart_retriever()

    docs = retriever.invoke("porcentaje cobra la plataforma")

    assert any("fees.md" in d.metadata["source"] for d in docs)

def test_recommender_diversity(rec_service):
    recs = rec_service.recommend("borja", k=3)
    assert len({r["title"] for r in recs}) == len(recs)
