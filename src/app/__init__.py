# src/app/__init__.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles     
from pathlib import Path
from .api.v1.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(title="Shakers RAG Demo")
    app.include_router(api_router, prefix="/api")

    # localiza la carpeta donde está index.html
    static_dir = Path(__file__).resolve().parent / "static"

    # ① monta todo el directorio en /static
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # ② sirve index.html en la raíz
    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(static_dir / "index.html")

    return app
