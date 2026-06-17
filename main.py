import logging
import numpy as np
import functools
import anyio
from asyncio import Lock
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Response
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager
from niti_core import run_simulation, VariableRequest, TraceConfig, Simulation, warmup_simulation_cache

from logging_utils import user_id_ctx, request_id_ctx
import uuid
import time
import gc
import os

GC_CALCULATE_THRESHOLD2 = int(os.environ.get("GC_CALCULATE_THRESHOLD2", "25"))
GC_CALCULATE_THRESHOLD1 = int(os.environ.get("GC_CALCULATE_THRESHOLD1", "5"))
GC_HEALTH_THRESHOLD = int(os.environ.get("GC_HEALTH_THRESHOLD", "120"))

calculate_counter = 0
health_counter = 0

async def cleanup_memory_task(is_calculate: bool):
    global calculate_counter, health_counter
    if not is_calculate and GC_HEALTH_THRESHOLD <= 0:
        return
        
    async with calculate_lock:
        start = time.perf_counter()
        gc_type = None
        
        if is_calculate:
            calculate_counter += 1
            if calculate_counter % GC_CALCULATE_THRESHOLD1 == 0:
                gc.collect(1)  # Gen 1 GC (Gen 0-1)
                gc_type = "Gen 1 (Gen 0-1)"
            elif GC_CALCULATE_THRESHOLD2 > 0 and calculate_counter >= GC_CALCULATE_THRESHOLD2:
                gc.collect()  # Full GC (Gen 2)
                calculate_counter = 0
                health_counter = 0
                gc_type = "Full (Gen 0-2)"
            else:
                gc.collect(0)  # Gen 0 GC (Gen 0)
                gc_type = "Gen 0 (Gen 0)"
        else:
            health_counter += 1
            if health_counter >= GC_HEALTH_THRESHOLD:
                gc.collect()  # Full GC (Gen 2)
                calculate_counter = 0
                health_counter = 0
                gc_type = "Full (Gen 0-2)"
                
        if gc_type:
            duration_ms = (time.perf_counter() - start) * 1000
            trigger_type = "calculate" if is_calculate else "health"
            logger.info(
                f"Garbage collection ({gc_type}) triggered by {trigger_type} "
                f"completed in {duration_ms:.2f}ms. "
                f"Counters: calc={calculate_counter}, health={health_counter}"
            )

logger = logging.getLogger(__name__)

# Prometheus Metrics instrumentation
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Histogram

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint", "status"]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Disable automatic garbage collection
    gc.set_threshold(0)
    logger.info("Disabled automatic garbage collection (GC).")
    
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
        process_time_s = process_time_ms / 1000.0
        
        path = request.url.path
        # Record metrics (skip health checks, metrics, and root to avoid noise)
        if path not in ["/health", "/metrics", "/"]:
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=path,
                status=response.status_code
            ).observe(process_time_s)
        
        # Manual access log with context and duration
        host = request.client.host if request.client else "unknown"
        method = request.method
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
async def health_check(background_tasks: BackgroundTasks):
    background_tasks.add_task(cleanup_memory_task, False)
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/calculate", response_model=CalculateResponse)
async def calculate_endpoint(req: CalculateRequest, background_tasks: BackgroundTasks):
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
        
        background_tasks.add_task(cleanup_memory_task, True)
        
        return CalculateResponse(
            arrays=arrays_serializable,
            trace=result.trace
        )
    except Exception as e:
        logger.exception("Error during simulation")
        raise HTTPException(status_code=500, detail=str(e))
