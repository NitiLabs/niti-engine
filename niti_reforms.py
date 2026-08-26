import numpy as np
from policyengine_us.model_api import Variable, Reform, TaxUnit, Person, YEAR, ParameterNode
from policyengine_us.variables.household.demographic.tax_unit.filing_status import filing_status

aca_limits_data = {
    "tier_1": {
        "max_fpl": {"values": {"2022-01-01": 2.0}},
        "single_limit": {"values": {"2022-01-01": 325, "2023-01-01": 350, "2024-01-01": 375, "2025-01-01": 375, "2026-01-01": float('inf')}},
        "joint_limit": {"values": {"2022-01-01": 650, "2023-01-01": 700, "2024-01-01": 750, "2025-01-01": 750, "2026-01-01": float('inf')}}
    },
    "tier_2": {
        "max_fpl": {"values": {"2022-01-01": 3.0}},
        "single_limit": {"values": {"2022-01-01": 825, "2023-01-01": 900, "2024-01-01": 950, "2025-01-01": 975, "2026-01-01": float('inf')}},
        "joint_limit": {"values": {"2022-01-01": 1650, "2023-01-01": 1800, "2024-01-01": 1900, "2025-01-01": 1950, "2026-01-01": float('inf')}}
    },
    "tier_3": {
        "max_fpl": {"values": {"2022-01-01": 4.0}},
        "single_limit": {"values": {"2022-01-01": 1400, "2023-01-01": 1500, "2024-01-01": 1575, "2025-01-01": 1625, "2026-01-01": float('inf')}},
        "joint_limit": {"values": {"2022-01-01": 2800, "2023-01-01": 3000, "2024-01-01": 3150, "2025-01-01": 3250, "2026-01-01": float('inf')}}
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


# ─────────────────────────────────────────────────────────────────────────────
# US Government Interest & Dividends Split Patch
# ─────────────────────────────────────────────────────────────────────────────
class us_govt_interest_income_person(Variable):
    value_type = float
    entity = Person
    label = "Interest from U.S. government obligations"
    definition_period = YEAR


class us_govt_dividends_person(Variable):
    value_type = float
    entity = Person
    label = "Dividends from U.S. government obligations"
    definition_period = YEAR


if "us_govt_interest_income_person" not in system.variables:
    system.add_variable(us_govt_interest_income_person)

if "us_govt_dividends_person" not in system.variables:
    system.add_variable(us_govt_dividends_person)

if "us_govt_interest_person" in system.variables:
    def _patched_us_govt_interest_person_formula(person, period, parameters):
        interest = person("us_govt_interest_income_person", period)
        dividends = person("us_govt_dividends_person", period)
        return interest + dividends

    _var = system.variables["us_govt_interest_person"]
    _var.formulas = {
        "2015-01-01": _patched_us_govt_interest_person_formula
    }


# ─────────────────────────────────────────────────────────────────────────────
# Custom Niti ACA Variables
# ─────────────────────────────────────────────────────────────────────────────

class niti_aca_yearly_premium_paid(Variable):
    value_type = float
    entity = TaxUnit
    label = "Total out-of-pocket premium paid over the year"
    definition_period = YEAR
    default_value = 0.0


class niti_aca_enrolled_months(Variable):
    value_type = int
    entity = TaxUnit
    label = "Months enrolled in ACA"
    definition_period = YEAR
    default_value = 12


class niti_advance_premium_tax_credit(Variable):
    value_type = float
    entity = TaxUnit
    label = "Advance premium tax credit reported"
    definition_period = YEAR
    default_value = 0.0


class niti_aca_premium_tax_credit(Variable):
    value_type = float
    entity = TaxUnit
    label = "ACA Premium Tax Credit (Gross)"
    definition_period = YEAR
    
    def formula(tax_unit, period, parameters):
        months = tax_unit("niti_aca_enrolled_months", period)
        
        # Check if anyone in the tax unit is eligible for PTC
        any_eligible = tax_unit.sum(tax_unit.members("is_aca_ptc_eligible", period)) > 0
        
        # 1. SLCSP Benchmark for the period We multiply by the enrollment fraction
        # to get the actual benchmark for the period.
        annual_slcsp = tax_unit("slcsp", period)
        ptc_benchmark = annual_slcsp * (months / 12.0)
        
        # 2. Scale up if advance premium tax credit reported is larger than SLCSP
        aptc = tax_unit("niti_advance_premium_tax_credit", period)
        ptc_benchmark = np.maximum(ptc_benchmark, aptc)
        
        # 3. Required contribution for the period
        magi = tax_unit("aca_magi", period)
        applicable_pct = tax_unit("aca_required_contribution_percentage", period)
        contribution_for_period = (magi * applicable_pct) * (months / 12.0)
        
        total_gross_ptc = np.maximum(0.0, ptc_benchmark - contribution_for_period)
        
        # 4. Cap at total premium price (paid + advance)
        premium_paid = tax_unit("niti_aca_yearly_premium_paid", period)
        annual_plan_premium = premium_paid + aptc
        total_gross_ptc = np.minimum(total_gross_ptc, annual_plan_premium)
        
        # ACA eligibility depends on enrolled months > 0 and having eligible members
        return np.where(any_eligible & (months > 0), total_gross_ptc, 0.0)


class niti_net_aca_tax(Variable):
    value_type = float
    entity = TaxUnit
    label = "Net ACA Tax Impact (Repayment Owed or Refund Claimed)"
    definition_period = YEAR
    
    def formula(tax_unit, period, parameters):
        gross_ptc = tax_unit("niti_aca_premium_tax_credit", period)
        aptc = tax_unit("niti_advance_premium_tax_credit", period)
        
        # Negative tax impact (refund)
        refund_impact = aptc - gross_ptc
        
        # Positive tax impact (repayment capped by limit)
        repayment_limit = tax_unit("niti_aca_repayment_limit", period)
        raw_repayment = aptc - gross_ptc
        repayment_impact = np.minimum(aptc, np.minimum(raw_repayment, repayment_limit))
        
        return np.where(gross_ptc < aptc, repayment_impact, refund_impact)


for var_class in [
    niti_aca_yearly_premium_paid,
    niti_aca_enrolled_months,
    niti_advance_premium_tax_credit,
    niti_aca_premium_tax_credit,
    niti_net_aca_tax
]:
    if var_class.__name__ not in system.variables:
        system.add_variable(var_class)


# ─────────────────────────────────────────────────────────────────────────────
# Temporary Fix: Include taxable_roth_conversions in State Incomes & Exclusions
# ─────────────────────────────────────────────────────────────────────────────

def _patch_parameter_source_list(param, var_name="taxable_roth_conversions"):
    """Appends var_name to ParameterAtInstant values so Simulation clones inherit it."""
    for item in getattr(param, "values_list", []):
        if isinstance(item.value, list) and var_name not in item.value:
            item.value = list(item.value) + [var_name]

# Category A: State Gross Income / AGI definitions
# https://github.com/PolicyEngine/policyengine-us/issues/9352
_patch_parameter_source_list(system.parameters.gov.states.al.tax.income.agi.gross_income_sources)
_patch_parameter_source_list(system.parameters.gov.states.ar.tax.income.gross_income.sources.individual)
_patch_parameter_source_list(system.parameters.gov.states.ar.tax.income.gross_income.sources.joint)
_patch_parameter_source_list(system.parameters.gov.states.nj.tax.income.gross_income.non_negative_sources)

# Category B: Senior / Retirement income exclusions
# https://github.com/PolicyEngine/policyengine-us/issues/9353
_patch_parameter_source_list(system.parameters.gov.states.co.tax.income.subtractions.pension.income_sources)
_patch_parameter_source_list(system.parameters.gov.states.ga.tax.income.agi.exclusions.retirement.sources)
_patch_parameter_source_list(system.parameters.gov.states.ky.tax.income.exclusions.pension_income.other_retirement_income_sources)

# Alabama Senior Retirement Exemption (Schedule RS / Ala. Code § 40-18-19(a)(13))
if "al_retirement_exemption_person" in system.variables:
    def _patched_al_retirement_exemption_formula(person, period, parameters):
        p = parameters(period).gov.states.al.tax.income.exemptions.retirement
        retirement_income = (
            person("taxable_retirement_distributions", period)
            + person("taxable_pension_income", period)
            + person("taxable_roth_conversions", period)
        )
        return np.minimum(retirement_income, p.cap)
    _al_ret_var = system.variables["al_retirement_exemption_person"]
    for _d in list(_al_ret_var.formulas.keys()):
        _al_ret_var.formulas[_d] = _patched_al_retirement_exemption_formula


# Arkansas Senior Retirement Exemption
if "ar_retirement_or_disability_benefits_exemption_person" in system.variables:
    def _patched_ar_retirement_exemption_formula(person, period, parameters):
        p = parameters(period).gov.irs.income.exemption.traditional_distribution
        head_or_spouse = person("is_tax_unit_head_or_spouse", period)
        disability_benefits_and_taxable_pensions = (
            person("disability_benefits", period) + person("taxable_pension_income", period)
        )
        ira_age_eligible = person("age", period) >= p.age_threshold
        age_eligible_ira_distributions = ira_age_eligible * (
            person("taxable_ira_distributions", period) + person("taxable_roth_conversions", period)
        )
        return head_or_spouse * (
            disability_benefits_and_taxable_pensions + age_eligible_ira_distributions
        )
    _ar_var = system.variables["ar_retirement_or_disability_benefits_exemption_person"]
    for _d in list(_ar_var.formulas.keys()):
        _ar_var.formulas[_d] = _patched_ar_retirement_exemption_formula

# Wisconsin Senior Retirement Exemption
if "wi_retirement_income_exclusion_amount" in system.variables:
    def _patched_wi_retirement_formula(tax_unit, period, parameters):
        p = parameters(period).gov.states.wi.tax.income.subtractions.retirement_income.exclusion
        person = tax_unit.members
        age = person("age", period)
        head_or_spouse = person("is_tax_unit_head_or_spouse", period)
        eligible = (age >= p.min_age) * head_or_spouse
        pension = person("taxable_pension_income", period)
        ira = person("taxable_ira_distributions", period) + person("taxable_roth_conversions", period)
        person_ret_income = (pension + ira) * eligible
        filing_status = tax_unit("filing_status", period)
        is_joint = filing_status == filing_status.possible_values.JOINT
        both_eligible = tax_unit.sum(eligible) >= 2
        total_ret_income = tax_unit.sum(person_ret_income)
        joint_both_amount = np.minimum(p.max_amount.joint, total_ret_income)
        per_person_capped = np.minimum(p.max_amount.single, person_ret_income)
        standard_amount = tax_unit.sum(per_person_capped)
        return np.where(is_joint & both_eligible, joint_both_amount, standard_amount)

    _wi_var = system.variables["wi_retirement_income_exclusion_amount"]
    for _d in list(_wi_var.formulas.keys()):
        _wi_var.formulas[_d] = _patched_wi_retirement_formula

