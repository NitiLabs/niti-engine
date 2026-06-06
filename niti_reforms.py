import numpy as np
from policyengine_us.model_api import Variable, Reform, TaxUnit, YEAR, ParameterNode
from policyengine_us.variables.household.demographic.tax_unit.filing_status import filing_status

aca_limits_data = {
    "tier_1": {
        "max_fpl": {"values": {"2022-01-01": 2.0}},
        "single_limit": {"values": {"2022-01-01": 325, "2023-01-01": 350, "2024-01-01": 375, "2025-01-01": 375}},
        "joint_limit": {"values": {"2022-01-01": 650, "2023-01-01": 700, "2024-01-01": 750, "2025-01-01": 750}}
    },
    "tier_2": {
        "max_fpl": {"values": {"2022-01-01": 3.0}},
        "single_limit": {"values": {"2022-01-01": 825, "2023-01-01": 900, "2024-01-01": 950, "2025-01-01": 975}},
        "joint_limit": {"values": {"2022-01-01": 1650, "2023-01-01": 1800, "2024-01-01": 1900, "2025-01-01": 1950}}
    },
    "tier_3": {
        "max_fpl": {"values": {"2022-01-01": 4.0}},
        "single_limit": {"values": {"2022-01-01": 1400, "2023-01-01": 1500, "2024-01-01": 1575, "2025-01-01": 1625}},
        "joint_limit": {"values": {"2022-01-01": 2800, "2023-01-01": 3000, "2024-01-01": 3150, "2025-01-01": 3250}}
    }
}

niti_aca_repayment_limits = ParameterNode(name="niti_aca_repayment_limits", data=aca_limits_data)

class niti_aca_repayment_limit(Variable):
    value_type = float
    entity = TaxUnit
    label = "ACA Repayment Limit based on Federal Poverty Line"
    definition_period = YEAR
    
    def formula(tax_unit, period, parameters):
        fpl_pct = tax_unit("aca_magi_fraction", period)
        is_single = tax_unit("filing_status", period) == filing_status.possible_values.SINGLE
        
        limit = parameters.niti_aca_repayment_limits
        
        limit_tier_1 = np.where(is_single, limit.tier_1.single_limit(period), limit.tier_1.joint_limit(period))
        limit_tier_2 = np.where(is_single, limit.tier_2.single_limit(period), limit.tier_2.joint_limit(period))
        limit_tier_3 = np.where(is_single, limit.tier_3.single_limit(period), limit.tier_3.joint_limit(period))
        
        return np.select(
            [
                fpl_pct < limit.tier_1.max_fpl(period), 
                fpl_pct < limit.tier_2.max_fpl(period), 
                fpl_pct < limit.tier_3.max_fpl(period)
            ],
            [limit_tier_1, limit_tier_2, limit_tier_3],
            default=np.inf
        )

from policyengine_us.system import system

# Register the custom variable and parameter node with the system instead of defining a reform. This helps with caching etc.
if "niti_aca_repayment_limits" not in system.parameters.children:
    system.parameters.add_child("niti_aca_repayment_limits", niti_aca_repayment_limits)

if "niti_aca_repayment_limit" not in system.variables:
    system.add_variable(niti_aca_repayment_limit)

# Optimize first_county_in_state.formula directly on the registered system variable instance.
# Rebuilding the state-to-county mapping on every evaluation executes 150k pure-Python string matching loops, adding ~1.0s.
# We replace it with a pre-computed O(1) cache lookup.
if "first_county_in_state" in system.variables:
    from policyengine_us.variables.household.demographic.geographic.county.county_enum import County
    from policyengine_us.variables.household.demographic.geographic.state_code import StateCode

    _state_to_first_county_cache = {}
    for _state in StateCode:
        _state_abbr = _state.value
        _state_counties = [
            _c for _c in County
            if _c != County.UNKNOWN and _c.value.endswith(f", {_state_abbr}")
        ]
        if _state_counties:
            _state_to_first_county_cache[_state_abbr] = min(_state_counties, key=lambda c: c.value)

    def _patched_first_county_in_state_formula(household, period, parameters):
        state_code_str = household("state_code_str", period)
        result = [
            _state_to_first_county_cache.get(state_abbr, County.UNKNOWN)
            for state_abbr in state_code_str
        ]
        return np.array(result)

    _var_instance = system.variables["first_county_in_state"]
    for _date in list(_var_instance.formulas.keys()):
        _var_instance.formulas[_date] = _patched_first_county_in_state_formula

# ─────────────────────────────────────────────────────────────────────────────
# Massachusetts Public Pension Exemption Reform
# ─────────────────────────────────────────────────────────────────────────────
if "ma_gross_income" in system.variables:
    _orig_ma_gross_income_formula = system.variables["ma_gross_income"].formulas[next(iter(system.variables["ma_gross_income"].formulas.keys()))]
    
    def _patched_ma_gross_income_formula(tax_unit, period, parameters):
        orig_val = _orig_ma_gross_income_formula(tax_unit, period, parameters)
        public_pension = tax_unit.sum(tax_unit.members("taxable_public_pension_income", period))
        return np.maximum(0.0, orig_val - public_pension)
        
    _var = system.variables["ma_gross_income"]
    for _date in list(_var.formulas.keys()):
        _var.formulas[_date] = _patched_ma_gross_income_formula

