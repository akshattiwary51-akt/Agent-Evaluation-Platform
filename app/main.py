from fastapi import Depends, FastAPI

from app.api.analysis_routes import router as analysis_router
from app.api.auth import require_auth
from app.api.routes import health, router
from app.web.routes import router as web_router

app = FastAPI(title="Agent Evaluation Platform", version="0.5.0")
app.include_router(health)  # open: liveness/readiness probes
for r in (router, analysis_router, web_router):
    app.include_router(r, dependencies=[Depends(require_auth)])
