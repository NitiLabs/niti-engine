"""
niti_core: Optimized version of PolicyEngine for household use case.
"""

import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── PolicyEngine imports (isolated here) ─────────────────────────────────
from policyengine_us import Simulation
from policyengine_core.simulations.simulation_macro_cache import SimulationMacroCache

import niti_reforms  # noqa: F401  – registers custom ACA variable globally
from policyengine_us.system import system

import policyengine_us.tools.geography.county_helpers

logger = logging.getLogger(__name__)

# ── Monkey-patch ─────────────────────────────────────────────────────────
# 1. Lazy-caching county FIPS dataset loading.
# This prevents filesystem reading/parsing overhead on subsequent requests.
_orig_load_county_fips = policyengine_us.tools.geography.county_helpers.load_county_fips_dataset
_cached_county_fips_df = None

def _lazy_load_county_fips_dataset():
    global _cached_county_fips_df
    if _cached_county_fips_df is None:
        start_time = time.time()
        df = _orig_load_county_fips()
        if df is None or len(df) <= 3200:
            row_count = len(df) if df is not None else 0
            raise RuntimeError(
                f"County FIPS dataset load error: expected > 3200 rows, but got {row_count} rows."
            )
        _cached_county_fips_df = df
        duration = time.time() - start_time
        logger.info(f"Loaded county FIPS dataset in {duration:.4f} seconds. Size/Shape: {_cached_county_fips_df.shape}")
    return _cached_county_fips_df

policyengine_us.tools.geography.county_helpers.load_county_fips_dataset = _lazy_load_county_fips_dataset

# Patch the dynamic module namespaces of the registered county formulas inside system:
_county_var = system.variables.get("county")
if _county_var:
    for formula in _county_var.formulas.values():
        if "load_county_fips_dataset" in formula.__globals__:
            formula.__globals__["load_county_fips_dataset"] = _lazy_load_county_fips_dataset


# ── Public data structures ───────────────────────────────────────────────

@dataclass(frozen=True)
class VariableRequest:
    """One variable the caller wants calculated.

    Attributes:
        name:    PolicyEngine variable name (e.g. ``"income_tax"``).
        map_to:  Optional entity to map to (e.g. ``"tax_unit"``).
                 Passed straight through to ``Simulation.calculate(…, map_to=)``.
        period:  Optional period to calculate the variable for.
    """
    name: str
    map_to: Optional[str] = None
    period: Optional[str] = None


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for PolicyEngine execution tracing.

    Attributes:
        enabled: Whether tracing is enabled.
        roots:   Tuple of variable-name prefixes whose trace nodes should
                 always be kept (e.g. ``("income_tax", "ca_income_tax",
                 "aca_ptc")``).  Year is appended automatically.
    """
    enabled: bool = False
    roots: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class SimulationResult:
    """The output of ``run_simulation``.

    Attributes:
        arrays:  Mapping of variable name → NumPy array of results.
                 Keys match the ``VariableRequest.name`` values that were
                 passed in.  Array length is 1 for a single-point calc
                 or N for an axes scan.
        trace:   If tracing was requested, the filtered flat-trace dict
                 from PolicyEngine's tracer.  ``None`` otherwise.
    """
    arrays: dict = field(default_factory=dict)
    trace: Optional[dict] = None


# ── Core function ────────────────────────────────────────────────────────

def run_simulation(
    situation: dict,
    variables: list[VariableRequest],
    year: int,
    *,
    trace_config: Optional[TraceConfig] = None,
) -> SimulationResult:
    """Create a PolicyEngine Simulation and calculate the requested variables.

    This is the single entry point that the rest of tax-api should use.
    It owns the full Simulation lifecycle: creation → calculation → teardown.

    Args:
        situation:    A raw PolicyEngine situation dictionary (people, tax_units,
                      families, households, and optionally axes).
        variables:    The list of variables to calculate.
        year:         The tax year (int, e.g. 2024).
        trace_config: Optional TraceConfig object containing trace enablement
                      and root prefix filter configuration.

    Returns:
        A ``SimulationResult`` containing the requested arrays and an
        optional trace.
    """
    run_trace = trace_config.enabled if trace_config is not None else False
    sim = Simulation(situation=situation, tax_benefit_system=system, trace=run_trace)

    result = SimulationResult()

    for var in variables:
        # Skip variables that don't exist in the system (e.g. tx_income_tax
        # for Texas which has no income tax).  The caller sees a missing key
        # in ``result.arrays`` and treats it as None / not-applicable.
        if var.name not in sim.tax_benefit_system.variables:
            continue
        kwargs = {}
        if var.map_to is not None:
            kwargs["map_to"] = var.map_to
        calc_year = int(var.period) if var.period is not None else year
        result.arrays[var.name] = sim.calculate(var.name, calc_year, **kwargs)

    if run_trace and trace_config is not None:
        result.trace = _filter_trace(
            sim.tracer.get_serialized_flat_trace(),
            trace_config.roots,
            year,
        )

    return result


def _filter_trace(
    full_trace: dict,
    trace_roots: tuple[str, ...],
    year: int,
) -> dict:
    """Prune the raw PolicyEngine trace to only the reachable, non-trivial nodes.

    ``trace_roots`` are the variable-name prefixes whose trace nodes should
    always be kept (e.g. ``("income_tax", "ca_income_tax", "aca_ptc")``).
    All other root nodes whose value is zero / False are discarded.  Then
    we walk the dependency graph from surviving roots and return only the
    reachable subset.

    This typically shrinks the trace from ~3,000–5,000 nodes / 2–5 MB down
    to a few hundred nodes / 100–300 KB, which matters when the trace is
    serialized over the wire from niti-engine → tax-api.
    """
    # e.g. ("income_tax", "ca_income_tax", "aca_ptc") → ("income_tax<", ...)
    always_keep_prefixes = tuple(
        f"{root}<" for root in trace_roots
    )

    # Identify which trace keys are dependency targets (i.e. not roots).
    all_dependencies: set[str] = set()
    for node in full_trace.values():
        if "dependencies" in node:
            all_dependencies.update(node["dependencies"])

    # Walk root nodes and keep those that are either always-keep or non-trivial.
    surviving_roots: set[str] = set()
    for key, node in full_trace.items():
        if key in all_dependencies:
            continue  # Not a root
        if key.startswith(always_keep_prefixes):
            surviving_roots.add(key)
            continue
        val = node.get("value")
        if val in (0, 0.0, False):
            continue
        if isinstance(val, list) and len(val) == 1 and val[0] in (0, False):
            continue
        surviving_roots.add(key)

    # BFS from surviving roots to collect the full reachable subgraph.
    reachable: set[str] = set()
    queue = list(surviving_roots)
    queued = set(surviving_roots)
    while queue:
        current_key = queue.pop(0)
        reachable.add(current_key)
        node = full_trace.get(current_key)
        if node and "dependencies" in node:
            for dep in node["dependencies"]:
                if dep not in queued:
                    queued.add(dep)
                    queue.append(dep)

    return {k: v for k, v in full_trace.items() if k in reachable}


def warmup_simulation_cache():
    """Run a comprehensive warmup simulation for representative years (2025, 2026)
    to force NumExpr JIT compilation and lazy-loading of parameters, geography datasets,
    and state/ACA variables.
    """
    for year in (2025, 2026):
        year_str = str(year)
        warmup_situation = {
            'people': {'p': {'age': {year_str: 45}, 'employment_income': {year_str: 10000.0}}},
            'tax_units': {'t': {'members': ['p'], 'filing_status': {year_str: 'SINGLE'}}},
            'families': {'f': {'members': ['p']}},
            'households': {'h': {'members': ['p'], 'state_code': {year_str: 'CA'}, 'county_fips': {year_str: '06085'}}}
        }
        
        sim = Simulation(situation=warmup_situation, tax_benefit_system=system)
        
        warmup_variables = [
            'income_tax',
            'niti_aca_repayment_limit',
            'niti_aca_premium_tax_credit',
            'niti_net_aca_tax',
            'aca_magi',
            'aca_required_contribution_percentage',
            'slcsp',
            'gross_medicare_part_b_premium',
            'base_part_b_premium',
            'income_adjusted_part_d_premium_surcharge',
            'ca_income_tax'
        ]
        
        for var_name in warmup_variables:
            if var_name in system.variables:
                kwargs = {}
                if var_name in ('gross_medicare_part_b_premium', 'base_part_b_premium', 'income_adjusted_part_d_premium_surcharge'):
                    kwargs["map_to"] = "tax_unit"
                try:
                    sim.calculate(var_name, year, **kwargs)
                except Exception:
                    pass
