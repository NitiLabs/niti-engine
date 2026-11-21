import asyncio
import contextvars
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager
from niti_core import (
    DEFAULT_WARMUP_YEARS,
    Simulation,
    TraceConfig,
    VariableRequest,
    run_simulation,
    warmup_simulation_year,
)
from prometheus_client import multiprocess, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, Histogram

from logging_utils import user_id_ctx, request_id_ctx
import uuid
import time
import gc
import os
import importlib.metadata

def get_policyengine_us_version() -> str:
    try:
        return importlib.metadata.version("policyengine-us")
    except Exception:
        try:
            import policyengine_us
            return getattr(policyengine_us, "__version__", "unknown")
        except Exception:
            return "unknown"

POLICYENGINE_US_VERSION = get_policyengine_us_version()

GC_CALCULATE_THRESHOLD2 = int(os.environ.get("GC_CALCULATE_THRESHOLD2", "25"))
GC_CALCULATE_THRESHOLD1 = int(os.environ.get("GC_CALCULATE_THRESHOLD1", "5"))
GC_IDLE_DELAY_SECONDS = float(os.environ.get("GC_IDLE_DELAY_SECONDS", "1.0"))
GC_PERIODIC_FULL_INTERVAL_SECONDS = float(os.environ.get("GC_PERIODIC_FULL_INTERVAL_SECONDS", "3600"))
SLOW_CALCULATE_THRESHOLD_MS = float(os.environ.get("SLOW_CALCULATE_THRESHOLD_MS", "400"))
LOG_LEVEL = (os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
PROMETHEUS_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

_GC_TYPE_LABELS = {
    0: "Gen 0 (Gen 0)",
    1: "Gen 1 (Gen 0-1)",
    2: "Full (Gen 0-2)",
}

# Single-worker executor serializes all PolicyEngine simulations and manual GC
# onto one pinned OS thread (PolicyEngine is not thread-safe) while keeping the
# asyncio event loop free to flush HTTP responses, /health, and /metrics.
sim_executor: Optional[ThreadPoolExecutor] = None

async def run_in_sim_executor(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(
        sim_executor,
        functools.partial(ctx.run, fn, *args, **kwargs),
    )

def _collect_gc_sync(gen: int) -> float:
    start = time.perf_counter()
    gc.collect(gen)
    return (time.perf_counter() - start) * 1000

class GCManager:
    """Debounces and coalesces manual GC onto sim_executor.

    All state lives on the asyncio event loop:
    - Waits GC_IDLE_DELAY_SECONDS after the last in-flight /calculate finishes
      so multi-request flows aren't interrupted by GC between calls.
    - Coalesces pending GC needs, with higher generations overriding lower ones
      (Gen 2 > Gen 1 > Gen 0).
    - Forces immediate GC if uncollected calculations reach GC_CALCULATE_THRESHOLD2
      so sustained traffic with no idle gaps cannot defer GC indefinitely.
    - Runs a periodic Full (Gen 0-2) GC if GC_PERIODIC_FULL_INTERVAL_SECONDS elapses
      without one.
    """
    def __init__(self) -> None:
        self.calculate_counter = 0
        self.uncollected_calculates = 0
        self._in_flight = 0
        self._pending_gen: Optional[int] = None
        self._pending_triggers: set[str] = set()
        self._gc_running = False
        self._idle_timer: Optional[asyncio.TimerHandle] = None
        self._periodic_task: Optional[asyncio.Task] = None
        self._gc_task: Optional[asyncio.Task] = None
        self._last_full_gc_time = time.monotonic()

    def start(self) -> None:
        self._last_full_gc_time = time.monotonic()
        if GC_PERIODIC_FULL_INTERVAL_SECONDS > 0:
            self._periodic_task = asyncio.create_task(
                self._periodic_full_gc_loop(),
                context=contextvars.Context(),
            )

    async def stop(self) -> None:
        self._cancel_idle_timer()
        for task in (self._periodic_task, self._gc_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._periodic_task = None
        self._gc_task = None

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def on_calculate_start(self) -> None:
        self._in_flight += 1
        self._cancel_idle_timer()

    def on_calculate_end(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)
        self.calculate_counter += 1
        self.uncollected_calculates += 1
        if GC_CALCULATE_THRESHOLD2 > 0 and self.calculate_counter >= GC_CALCULATE_THRESHOLD2:
            needed_gen = 2
            self.calculate_counter = 0
        elif GC_CALCULATE_THRESHOLD1 > 0 and self.calculate_counter % GC_CALCULATE_THRESHOLD1 == 0:
            needed_gen = 1
        else:
            needed_gen = 0
        self._merge_need(needed_gen, "calculate")

        # Under sustained traffic with no idle gap, bound memory by forcing GC once
        # uncollected calculations reach GC_CALCULATE_THRESHOLD2.
        if GC_CALCULATE_THRESHOLD2 > 0 and self.uncollected_calculates >= GC_CALCULATE_THRESHOLD2:
            self._cancel_idle_timer()
            self._dispatch_gc()
        elif self._in_flight == 0:
            self._arm_idle_timer()

    def _merge_need(self, gen: int, trigger: str) -> None:
        if self._pending_gen is None or gen > self._pending_gen:
            self._pending_gen = gen
        self._pending_triggers.add(trigger)

    def _arm_idle_timer(self) -> None:
        self._cancel_idle_timer()
        if GC_IDLE_DELAY_SECONDS <= 0:
            self._dispatch_gc()
            return
        loop = asyncio.get_running_loop()
        self._idle_timer = loop.call_later(
            GC_IDLE_DELAY_SECONDS,
            self._on_idle_timer,
            context=contextvars.Context(),
        )

    def _on_idle_timer(self) -> None:
        self._idle_timer = None
        if self._in_flight == 0 and self._pending_gen is not None:
            self._dispatch_gc()

    def _dispatch_gc(self) -> None:
        if self._gc_running or self._pending_gen is None or sim_executor is None:
            return
        self._gc_running = True
        self._gc_task = asyncio.create_task(
            self._run_gc(),
            context=contextvars.Context(),
        )

    async def _run_gc(self) -> None:
        try:
            if sim_executor is None or self._pending_gen is None:
                return
            gen = self._pending_gen
            triggers = "+".join(sorted(self._pending_triggers))
            collected_calcs = self.uncollected_calculates
            self._pending_gen = None
            self._pending_triggers.clear()
            self.uncollected_calculates = 0
            if gen == 2:
                self.calculate_counter = 0

            duration_ms = await run_in_sim_executor(_collect_gc_sync, gen)
            if gen == 2:
                self._last_full_gc_time = time.monotonic()
            logger.info(
                f"Garbage collection ({_GC_TYPE_LABELS[gen]}) triggered by {triggers} "
                f"completed in {duration_ms:.2f}ms. "
                f"Counters: calc={self.calculate_counter}, collected_calcs={collected_calcs}"
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error during garbage collection")
        finally:
            self._gc_running = False
            self._gc_task = None
            if self._pending_gen is not None and self._in_flight == 0 and self._idle_timer is None:
                self._arm_idle_timer()

    async def _periodic_full_gc_loop(self) -> None:
        while True:
            elapsed = time.monotonic() - self._last_full_gc_time
            sleep_for = GC_PERIODIC_FULL_INTERVAL_SECONDS - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
                continue

            self._merge_need(2, "periodic")
            if self._in_flight == 0:
                self._cancel_idle_timer()
                self._dispatch_gc()

            if self._gc_task is not None:
                await asyncio.shield(self._gc_task)
            else:
                await asyncio.sleep(min(1.0, GC_PERIODIC_FULL_INTERVAL_SECONDS))

gc_manager = GCManager()

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

if PROMETHEUS_MULTIPROC_DIR:
    os.makedirs(PROMETHEUS_MULTIPROC_DIR, exist_ok=True)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint", "status"]
)

def _collect_and_freeze() -> None:
    gc.collect(2)
    # Freeze the loaded objects to speed up future gen 2 GCs.
    gc.freeze()


async def _run_background_warmup() -> None:
    """Warm up each representative tax year in its own executor task so incoming
    /calculate requests can interleave between years instead of waiting for all years.
    """
    start = time.perf_counter()
    logger.info(f"Starting background PolicyEngine simulation warmup for years {DEFAULT_WARMUP_YEARS}...")
    try:
        for year in DEFAULT_WARMUP_YEARS:
            if sim_executor is None:
                return
            year_start = time.perf_counter()
            await run_in_sim_executor(warmup_simulation_year, year)
            logger.info(f"Warmed up PolicyEngine cache for {year} in {(time.perf_counter() - year_start) * 1000:.2f}ms.")
        if sim_executor is None:
            return
        await run_in_sim_executor(_collect_and_freeze)
        logger.info(f"Background warmup and gc.freeze() complete in {(time.perf_counter() - start) * 1000:.2f}ms.")
    except asyncio.CancelledError:
        logger.info("Background warmup cancelled during shutdown.")
        raise
    except Exception:
        logger.exception("Failed to warm up simulation")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim_executor
    # Disable automatic garbage collection
    gc.set_threshold(0)
    logger.info("Disabled automatic garbage collection (GC).")
    logger.info(f"policyengine-us version: {POLICYENGINE_US_VERSION}")

    sim_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="niti-sim")

    # Run warmup in the background so uvicorn starts listening and passes /health immediately;
    # any early /calculate requests queue cleanly behind the active warmup year on sim_executor.
    warmup_task = asyncio.create_task(_run_background_warmup(), context=contextvars.Context())
    gc_manager.start()
    try:
        yield
    finally:
        if not warmup_task.done():
            warmup_task.cancel()
            try:
                await warmup_task
            except asyncio.CancelledError:
                pass
        await gc_manager.stop()
        if sim_executor is not None:
            sim_executor.shutdown(wait=True)
            sim_executor = None

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
    period: Optional[str] = None

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

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "pe_version": POLICYENGINE_US_VERSION,
    }

@app.get("/metrics")
async def metrics():
    if PROMETHEUS_MULTIPROC_DIR:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

def _execute_simulation(
    situation: dict,
    core_vars: list[VariableRequest],
    year: int,
    core_trace: Optional[TraceConfig],
    profile: bool,
) -> tuple[dict[str, list[Any]], Optional[dict]]:
    start = time.perf_counter()
    pr = None
    if profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()

    try:
        result = run_simulation(
            situation,
            core_vars,
            year,
            trace_config=core_trace,
        )
        arrays_serializable = {
            k: np.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10).tolist() if hasattr(v, "tolist") else v
            for k, v in result.arrays.items()
        }
        if pr is not None:
            pr.disable()
            with open("/app/tmp/api_profile_stats.txt", "w") as f:
                ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
                ps.print_stats(150)
            logger.info("Saved API profile to tmp/api_profile_stats.txt")
        return arrays_serializable, result.trace
    except Exception:
        if pr is not None:
            try:
                pr.disable()
            except Exception:
                pass
        raise
    finally:
        compute_ms = (time.perf_counter() - start) * 1000
        if compute_ms > SLOW_CALCULATE_THRESHOLD_MS and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Slow calculate request ({compute_ms:.2f}ms > {SLOW_CALCULATE_THRESHOLD_MS:.0f}ms): "
                f"year={year}, situation={situation}, variables={core_vars}, trace_config={core_trace}"
            )

@app.post("/calculate", response_model=CalculateResponse)
async def calculate_endpoint(req: CalculateRequest, request: Request):
    profile = request.headers.get("X-Profile") == "true"

    try:
        # Convert pydantic models to dataclasses used by niti_core
        core_vars = [
            VariableRequest(name=v.name, map_to=v.map_to, period=v.period)
            for v in req.variables
        ]

        core_trace = None
        if req.trace_config and req.trace_config.enabled:
            logger.info(f"Trace config received: enabled={req.trace_config.enabled}, roots={req.trace_config.roots}")
            core_trace = TraceConfig(
                enabled=req.trace_config.enabled,
                roots=tuple(req.trace_config.roots)
            )

        gc_manager.on_calculate_start()
        try:
            arrays_serializable, trace = await run_in_sim_executor(
                _execute_simulation,
                req.situation,
                core_vars,
                req.year,
                core_trace,
                profile,
            )
        finally:
            gc_manager.on_calculate_end()

        return CalculateResponse(
            arrays=arrays_serializable,
            trace=trace
        )
    except Exception as e:
        logger.exception("Error during simulation")
        raise HTTPException(status_code=500, detail=str(e))
