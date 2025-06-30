from fastapi import APIRouter, Depends
from ...services.rag import RAGService
from ...services.recommender import RecommendationService
from ...deps import get_rag, get_rec      # ← import the real functions
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

router = APIRouter()

class AskReq(BaseModel):
    question: str
    user_id: str = "anonymous"

class RecReq(BaseModel):
    user_id: str = "anonymous"
    top_k: int = 3

""" @router.post("/ask")
def ask(
    req: AskReq,
    rag: RAGService = Depends(get_rag),
    rec: RecommendationService = Depends(get_rec),
):
    answer, sources = rag.ask(req.question, req.user_id)   # pass uid
    rec.log_sources(req.user_id, sources)                  # ⬅ keeps profile fresh
    return {"answer": answer, "sources": sources} """

@router.post("/ask")
def ask(req: AskReq,
        rag: RAGService         = Depends(get_rag),
        rec: RecommendationService = Depends(get_rec)):
    answer, sources = rag.ask(req.question, req.user_id)
    rec.log_sources(req.user_id, sources)             # keep as-is
    rec.log_query  (req.user_id, req.question)        # 🆕
    return {"answer": answer, "sources": sources}

@router.get("/ask_stream")
async def ask_stream(question: str,
                     user_id: str = "anonymous",
                     rag: RAGService = Depends(get_rag)):
    async def event_generator():
        async for chunk in rag.ask_stream(question, user_id):
             yield f"data: {chunk}\n\n"
    return StreamingResponse(event_generator(),
                              media_type="text/event-stream")

@router.post("/recommend")
def recommend(
    req: RecReq,
    rec: RecommendationService = Depends(get_rec)
):
    return {"recommendations": rec.recommend(req.user_id, req.top_k)}
