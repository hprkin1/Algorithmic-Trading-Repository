"""Universe for systematic pairs selection (v2). Pairs are only formed WITHIN a group.

Chosen 2026-10-30 on economic grounds (same industry / same exposure), before any v2 results.
Caveat: these are companies that still exist today (survivorship bias flatters results somewhat;
ETFs are much less affected than single stocks).
"""
STOCK_GROUPS: dict[str, list[str]] = {
    "integrated_oil": ["XOM", "CVX", "COP", "OXY"],
    "e_and_p": ["EOG", "DVN", "APA", "COP", "OXY"],
    "oil_services": ["SLB", "HAL", "BKR"],
    "big_banks": ["JPM", "BAC", "C", "WFC"],
    "regional_banks": ["USB", "PNC", "TFC", "FITB", "KEY", "RF", "HBAN", "MTB"],
    "investment_banks": ["GS", "MS", "SCHW"],
    "insurance_pc": ["TRV", "ALL", "PGR", "CB", "HIG"],
    "insurance_life": ["MET", "PRU", "AIG", "AFL", "PFG"],
    "payments": ["V", "MA", "AXP", "COF"],
    "asset_managers": ["BLK", "TROW", "BEN", "IVZ"],
    "rails": ["UNP", "CSX", "NSC"],
    "parcel": ["UPS", "FDX"],
    "airlines": ["DAL", "UAL", "LUV", "ALK"],
    "defense": ["LMT", "NOC", "GD", "RTX"],
    "machinery": ["CAT", "DE", "CMI", "PCAR"],
    "industrial_conglomerates": ["HON", "MMM", "ETN", "EMR", "ITW", "PH"],
    "utilities": ["DUK", "SO", "AEP", "D", "EXC", "XEL", "ED", "PEG", "NEE", "WEC"],
    "beverages": ["KO", "PEP"],
    "household_products": ["PG", "CL", "KMB", "CLX", "CHD"],
    "packaged_food": ["GIS", "CPB", "HSY", "MDLZ", "SJM", "CAG"],
    "tobacco": ["MO", "PM"],
    "home_improvement": ["HD", "LOW"],
    "big_box_retail": ["WMT", "TGT", "COST", "DG", "DLTR"],
    "off_price_retail": ["TJX", "ROST"],
    "restaurants": ["MCD", "YUM", "SBUX", "DRI"],
    "pharma": ["JNJ", "PFE", "MRK", "LLY", "BMY", "AMGN", "GILD"],
    "managed_care": ["UNH", "ELV", "CI", "HUM", "CVS"],
    "med_devices": ["MDT", "ABT", "SYK", "BSX", "BDX", "ZBH"],
    "semis": ["INTC", "TXN", "ADI", "MCHP", "QCOM", "MU", "AMAT", "LRCX", "KLAC"],
    "software_it": ["MSFT", "ORCL", "ADBE", "IBM", "ACN", "INTU"],
    "telecom": ["T", "VZ"],
    "chemicals": ["APD", "ECL", "SHW", "PPG", "EMN"],
    "metals_mining": ["FCX", "NEM", "NUE", "STLD"],
    "homebuilders": ["DHI", "LEN", "PHM", "TOL", "NVR"],
    "reits": ["SPG", "PLD", "PSA", "O", "VTR", "WELL", "BXP"],  # AVB, EQR: no usable Yahoo history (Oct 2026)
    "waste": ["WM", "RSG"],
    "media": ["DIS", "CMCSA"],
}

ETF_GROUPS: dict[str, list[str]] = {
    "us_sectors": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
    "us_broad": ["SPY", "DIA", "IWM", "MDY", "QQQ"],
    "developed_countries": ["EWA", "EWC", "EWG", "EWJ", "EWU", "EWQ", "EWL", "EWD", "EWN", "EWP", "EWI"],
    "asia_em_countries": ["EWH", "EWS", "EWT", "EWY", "EWZ", "EWW", "EZA", "EEM"],
    "banks_etf": ["KBE", "KRE", "XLF"],
    "energy_etf": ["XLE", "XOP", "OIH"],
    "health_etf": ["XLV", "IBB", "XBI"],
    "bonds": ["IEF", "TLT", "AGG", "LQD", "TIP", "SHY"],
    "metals": ["GLD", "SLV", "GDX", "GDXJ"],
}


def all_groups() -> dict[str, list[str]]:
    return {**{f"stock:{k}": v for k, v in STOCK_GROUPS.items()}, **{f"etf:{k}": v for k, v in ETF_GROUPS.items()}}


def all_symbols() -> list[str]:
    return sorted({s for g in all_groups().values() for s in g})
