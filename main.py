import logging
import numpy as np
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager
from niti_core import run_simulation, VariableRequest, TraceConfig, Simulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up PolicyEngine on server startup so the first real API request is sub-0.3s
    logger.info("Warming up PolicyEngine simulation cache...")
    try:
        _warmup_sim = Simulation(situation={"people": {"p": {"age": {"2025": 30}}}})
        _warmup_sim.calculate("income_tax", 2025)
        logger.info("Warmup complete.")
    except Exception:
        logger.exception("Failed to warm up simulation")
    yield

app = FastAPI(
    title="Niti Engine",
    description="Standalone PolicyEngine calculation service",
    lifespan=lifespan
)

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

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/calculate", response_model=CalculateResponse)
def calculate_endpoint(req: CalculateRequest):
    try:
        # Convert pydantic models to dataclasses used by niti_core
        core_vars = [
            VariableRequest(name=v.name, map_to=v.map_to)
            for v in req.variables
        ]
        
        core_trace = None
        if req.trace_config:
            core_trace = TraceConfig(
                enabled=req.trace_config.enabled,
                roots=tuple(req.trace_config.roots)
            )
            
        # Run simulation
        result = run_simulation(
            situation=req.situation,
            variables=core_vars,
            year=req.year,
            trace_config=core_trace
        )
        
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
