"""Generate the synthetic-client supplement (CLT-006, CLT-008).

The source portfolios.xlsx has 8 real clients; the assignment says 10. This script
writes data/portfolios/synthetic_supplement.xlsx with two clearly-synthetic clients
using the SAME schema and asset_class/sector vocabulary as the real file (see
docs/data_assumptions.md §1). The original file is never touched; the repository
unions the two at load time.

Idempotent: python -m scripts.generate_synthetic_clients
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

OUT = Path("data/portfolios/synthetic_supplement.xlsx")

COLUMNS = [
    "client_id", "symbol", "security_name", "asset_class",
    "quantity", "purchase_date", "Purchase Price", "sector",
]

# CLT-006 — balanced/moderate: individual stocks + core ETFs + cash.
# CLT-008 — income tilt: dividend stocks + dividend/muni/international ETFs + cash.
# Cash rows follow the source convention: quantity = dollar balance, price ≈ $50 noise.
ROWS = [
    # client_id, symbol, security_name, asset_class, qty, purchase_date, price, sector
    ("CLT-006", "JPM",  "JPMorgan Chase & Co",                  "Individual Stock",   320, datetime(2022, 3, 14), 131.25, "Financials"),
    ("CLT-006", "HD",   "Home Depot Inc",                       "Individual Stock",   150, datetime(2021, 6, 8),  305.40, "Consumer Discretionary"),
    ("CLT-006", "ABBV", "AbbVie Inc",                           "Individual Stock",   280, datetime(2023, 2, 21), 152.80, "Healthcare"),
    ("CLT-006", "VOO",  "Vanguard S&P 500 ETF",                 "Large Cap ETF",      210, datetime(2020, 11, 16), 330.15, "Broad Market"),
    ("CLT-006", "BND",  "Vanguard Total Bond Market ETF",       "Bond ETF",           950, datetime(2021, 8, 9),   85.42, "Government Bonds"),
    ("CLT-006", "VEA",  "Vanguard FTSE Developed Markets ETF",  "International ETF",  780, datetime(2022, 10, 17), 41.80, "International Developed"),
    ("CLT-006", "CASH", "Money Market Fund",                    "Cash Equivalent",  34750, datetime(2024, 2, 20),  50.35, "Cash"),

    ("CLT-008", "KO",   "The Coca-Cola Company",                "Individual Stock",   600, datetime(2020, 7, 22),  47.85, "Consumer Staples"),
    ("CLT-008", "PEP",  "PepsiCo Inc",                          "Individual Stock",   220, datetime(2021, 10, 5), 158.90, "Consumer Staples"),
    ("CLT-008", "VZ",   "Verizon Communications Inc",           "Individual Stock",   900, datetime(2022, 4, 11),  51.30, "Communication Services"),
    ("CLT-008", "DUK",  "Duke Energy Corporation",              "Individual Stock",   350, datetime(2021, 3, 29),  96.20, "Utilities"),
    ("CLT-008", "SCHD", "Schwab U.S. Dividend Equity ETF",      "Dividend ETF",       850, datetime(2021, 12, 13), 76.45, "Dividend Stocks"),
    ("CLT-008", "MUB",  "iShares National Muni Bond ETF",       "Municipal Bond ETF", 380, datetime(2022, 8, 24), 106.80, "Municipal Bonds"),
    ("CLT-008", "VXUS", "Vanguard Total International Stock ETF", "International ETF", 520, datetime(2023, 7, 18),  57.90, "International"),
    ("CLT-008", "CASH", "High-Yield Savings",                   "Cash Equivalent",  48200, datetime(2024, 6, 5),   52.10, "Cash"),
]


def main() -> None:
    df = pd.DataFrame(ROWS, columns=COLUMNS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT, sheet_name="Potfolios", index=False)  # same (sic) sheet name as the source
    print(f"wrote {OUT} — {len(df)} rows, clients: {sorted(df.client_id.unique())}")


if __name__ == "__main__":
    main()
