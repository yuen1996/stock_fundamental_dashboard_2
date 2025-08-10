# Central config for thresholds keyed EXACTLY as ratios return from utils/calculations.py
RATIO_THRESHOLDS = {
    "ROE (%)": 12,                # good if >= 12
    "Current Ratio": 2,           # good if >= 2
    "Dividend Yield (%)": 4,      # good if >= 4%
    "P/E": 15,                    # good if <= 15
    "P/B": 2,                     # good if <= 2
    "Debt-Asset Ratio (%)": 50,   # good if <= 50
}


# =========================
# Industry-aware scoring (new)
# =========================

# Master switches
USE_INDUSTRY_SCORING   = True     # turn new engine on/off
MIN_SCORE_INDUSTRY     = 65       # composite threshold to PASS
MIN_VALUATION_ENTRY    = 50       # minimum valuation-at-entry score to PASS
FD_RATE                = 0.035    # Malaysia fixed deposit baseline (UI can override)

# Block weights (sum to 100)
BLOCK_WEIGHTS = {
    "cashflow_first": 30,
    "ttm_vs_lfy":     15,
    "growth_quality": 25,
    "valuation_entry":15,
    "dividend":       10,
    "momentum":        5,
}

# Winsorization caps (tame outlier inputs before scoring)
CAPS = {
    "pe": 80, "pb": 10, "ev_ebitda": 30, "nd_ebitda": 10, "coverage": 30,
    "gm": 0.85, "nm": 0.60, "cash_conv": 2.0, "yield": 0.20,
}

# Map free-text Industry to our buckets (keyword-based; fallback = "General")
INDUSTRY_BUCKET_MAP = {
    "bank": "Financials", "finance": "Financials", "insurance": "Financials",
    "reit": "REITs", "trust": "REITs",
    "retail": "Retail", "consumer": "Retail",
    "manufactur": "Manufacturing", "industrial": "Manufacturing",
    "utility": "Utilities",
    "energy": "Energy/Materials", "material": "Energy/Materials",
    "tech": "Tech", "software": "Tech", "semicon": "Tech",
    "health": "Healthcare", "pharma": "Healthcare",
    "telco": "Telco",
}

# Bucket profiles: ramps (lo→hi) and which metrics to use at entry
BUCKET_PROFILES = {
    "Manufacturing": {
        "value":   {"pe": (15, 7, True), "ev_ebitda": (12, 6, True), "fcf_yield": (0.02, 0.08)},
        "quality": {"gm": (0.12, 0.35), "cash_conv": (0.60, 1.20)},
        "cash":    {"nd_ebitda": (4, 0, True), "coverage": (2, 12), "debt_fcf": (8, 0, True)},
        "entry":   ["pe", "ev_ebitda", "fcf_yield"],
        "fair_bands": {"pe": (9, 13), "ev_ebitda": (7, 10)},
    },
    "Retail": {
        "value":   {"pe": (18, 9, True), "fcf_yield": (0.015, 0.06)},
        "quality": {"gm": (0.10, 0.30), "cash_conv": (0.55, 1.10)},
        "cash":    {"coverage": (2, 10), "debt_fcf": (8, 0, True)},
        "entry":   ["pe", "fcf_yield"],
        "fair_bands": {"pe": (11, 15)},
    },
    "Financials": {
        "value":   {"pb": (1.6, 0.8, True), "pe": (15, 8, True)},
        "quality": {"roe": (0.08, 0.18), "nm": (0.15, 0.35)},
        "cash":    {"coverage": (2, 8)},
        "entry":   ["pb", "pe"],
        "fair_bands": {"pb": (1.0, 1.3)},
    },
    "REITs": {
        "value":   {"yield": (0.04, 0.08), "pb": (1.4, 0.9, True)},
        "quality": {"nm": (0.12, 0.30), "cash_conv": (0.70, 1.20)},
        "cash":    {"coverage": (1.8, 4.5), "debt_fcf": (10, 4, True)},
        "entry":   ["yield", "pb"],
        "fair_bands": {"pb": (0.9, 1.2)},
    },
    "Utilities": {
        "value":   {"pe": (18, 10, True), "yield": (0.035, 0.06)},
        "quality": {"gm": (0.20, 0.40), "cash_conv": (0.70, 1.20)},
        "cash":    {"coverage": (2, 8), "debt_fcf": (10, 4, True)},
        "entry":   ["pe", "yield"],
        "fair_bands": {"pe": (12, 16)},
    },
    "Energy/Materials": {
        "value":   {"ev_ebitda": (14, 6, True), "pe": (16, 8, True)},
        "quality": {"gm": (0.10, 0.35), "cash_conv": (0.60, 1.20)},
        "cash":    {"nd_ebitda": (5, 0, True), "coverage": (2, 10)},
        "entry":   ["ev_ebitda", "pe"],
        "fair_bands": {"ev_ebitda": (7, 9)},
    },
    "Tech": {
        "value":   {"pe": (25, 12, True), "fcf_yield": (0.00, 0.05)},
        "quality": {"gm": (0.30, 0.60), "cash_conv": (0.50, 1.00)},
        "cash":    {"coverage": (1.5, 8), "nd_ebitda": (3, 0, True)},
        "entry":   ["pe", "fcf_yield"],
        "fair_bands": {"pe": (16, 22)},
    },
    "Healthcare": {
        "value":   {"pe": (25, 12, True), "fcf_yield": (0.00, 0.05)},
        "quality": {"gm": (0.30, 0.60), "cash_conv": (0.55, 1.05)},
        "cash":    {"coverage": (1.5, 8)},
        "entry":   ["pe", "fcf_yield"],
        "fair_bands": {"pe": (17, 23)},
    },
    "Telco": {
        "value":   {"pe": (20, 10, True), "yield": (0.03, 0.06)},
        "quality": {"gm": (0.30, 0.55), "cash_conv": (0.60, 1.10)},
        "cash":    {"coverage": (2, 8), "debt_fcf": (9, 3, True)},
        "entry":   ["pe", "yield"],
        "fair_bands": {"pe": (12, 16)},
    },
    "General": {
        "value":   {"pe": (18, 9, True), "fcf_yield": (0.02, 0.08)},
        "quality": {"gm": (0.15, 0.40), "cash_conv": (0.60, 1.30)},
        "cash":    {"nd_ebitda": (4, 0, True), "coverage": (2, 12), "debt_fcf": (8, 0, True)},
        "entry":   ["pe", "fcf_yield"],
        "fair_bands": {"pe": (11, 15)},
    },
}

# SENSE alert thresholds (for warnings even if scores look good)
ALERTS = {
    "jump_factor": 2.0,        # ≥100% spike (x2) YoY or TTM
    "np_margin_drop_pp": 3,    # Net Profit Margin drop ≥ 3 percentage points
    "rev_drop_pct": -0.10,     # Revenue down ≥ 10% YoY
    "debt_jump_pct": 0.30,     # Total debt up ≥ 30% YoY
    "payout_jump_pp": 20,      # Payout ratio up ≥ 20pp YoY
    "accruals_warn": 0.25,     # (Net Profit − CFO) / Assets > 25% (LFY)
    "dilution_5y": 0.25,       # Shares up > 25% over 5y
    "coverage_low": 2.0,       # Interest coverage < 2 (TTM)
    "cfo_np_ratio_low": 0.60,  # CFO / Net Profit (LFY) < 0.6
}

# Optional: reduce block confidence if severe alerts are present
ALERTS_CONF_DEGRADE = {"critical": 0.20, "warning": 0.10, "info": 0.05}
