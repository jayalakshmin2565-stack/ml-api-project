from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uuid
import time

from prometheus_fastapi_instrumentator import Instrumentator

from app.logging_config import logger
from app.v1 import router as v1_router
from app.v2 import router as v2_router
from app.config import settings


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.MODEL_VERSION
)


# ---------------------------------------------------------
# API Routers
# ---------------------------------------------------------

app.include_router(v1_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")


# ---------------------------------------------------------
# Prometheus Monitoring
# ---------------------------------------------------------

Instrumentator(
    should_ignore_untemplated=True
).instrument(app).expose(app)


# ---------------------------------------------------------
# Request Logging Middleware
# ---------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    try:
        response = await call_next(request)

        duration = time.time() - start_time
        path = request.url.path

        logger.info(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={path} "
            f"status_code={response.status_code} "
            f"duration={duration:.4f}s"
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as exc:
        duration = time.time() - start_time
        path = request.url.path

        logger.error(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={path} "
            f"duration={duration:.4f}s "
            f"error={exc}"
        )

        raise


# ---------------------------------------------------------
# Global ValueError Handler
# ---------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"value_error "
        f"request_id={request_id} "
        f"error={exc}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Invalid prediction data",
            "request_id": request_id
        }
    )