import logging
import numpy as np
import functools
import anyio
from asyncio import Lock
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager
from niti_core import run_simulation, VariableRequest, TraceConfig, Simulation, warmup_simulation_cache

from logging_utils import user_id_ctx, request_id_ctx
import uuid
import time
from fastapi import Request

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up PolicyEngine on server startup so the first real API request is sub-0.3s
    logger.info("Warming up PolicyEngine simulation cache...")
    try:
        warmup_simulation_cache()
        logger.info("Warmup complete.")
    except Exception:
        logger.exception("Failed to warm up simulation")
    yield

app = FastAPI(
    title="Niti Engine",
    description="Standalone PolicyEngine calculation service",
    lifespan=lifespan
)

@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """
    Extract user_id and request_id from headers and set them in contextvars.
    Generates a new request_id if not provided by the caller.
    """
    user_id = request.headers.get("X-User-ID", "anonymous")
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    user_token = user_id_ctx.set(user_id)
    req_token = request_id_ctx.set(request_id)
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Manual access log with context and duration
        host = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        version = request.scope.get("http_version", "1.1")
        logger.info(f'{host} - "{method} {path} HTTP/{version}" {response.status_code} ({process_time_ms:.2f}ms)')
        
        # Return the request ID in the response headers
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        user_id_ctx.reset(user_token)
        request_id_ctx.reset(req_token)

class VariableRequestModel(BaseModel):
    name: str
    map_to: Optional[str] = None

class TraceConfigModel(BaseModel):
    enabled: bool = False
    roots: list[str] = Field(default_factory=list)

class CalculateRequest(BaseModel):
    situation: dict
    variables: list[VariableRequestModel]
    year: int
    trace_config: Optional[TraceConfigModel] = None

class CalculateResponse(BaseModel):
    arrays: dict[str, list[Any]]
    trace: Optional[dict] = None

# Create a global lock to serialize calculate requests
calculate_lock = Lock()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/calculate", response_model=CalculateResponse)
async def calculate_endpoint(req: CalculateRequest):
    try:
        # Convert pydantic models to dataclasses used by niti_core
        core_vars = [
            VariableRequest(name=v.name, map_to=v.map_to)
            for v in req.variables
        ]

        core_trace = None
        if req.trace_config and req.trace_config.enabled:
            logger.info(f"Trace config received: enabled={req.trace_config.enabled}, roots={req.trace_config.roots}")
            core_trace = TraceConfig(
                enabled=req.trace_config.enabled,
                roots=tuple(req.trace_config.roots)
            )
            
        # Run simulation sequentially under lock, offloading CPU-bound tasks to threadpool.
        # PolicyEngine is not thread-safe, so we don't run multiple simulations in parallel.
        logger.debug(f"Queueing simulation for year {req.year}, situation={req.situation} with variables={core_vars} and trace_config={core_trace}")
        
        async with calculate_lock:
            func = functools.partial(
                run_simulation,
                req.situation,
                core_vars,
                req.year,
                trace_config=core_trace
            )
            result = await anyio.to_thread.run_sync(func)
        
        arrays_serializable = {
            k: np.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10).tolist() if hasattr(v, "tolist") else v
            for k, v in result.arrays.items()
        }
        
        return CalculateResponse(
            arrays=arrays_serializable,
            trace=result.trace
        )
    except Exception as e:
        logger.exception("Error during simulation")
        raise HTTPException(status_code=500, detail=str(e))
