from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
from .api.v1.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(title="Shakers RAG Demo")
    app.include_router(api_router, prefix="/api")

    # static UI
    static_dir = Path(__file__).parent / "static"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return (static_dir / "index.html").read_text(encoding="utf-8")

    return app
