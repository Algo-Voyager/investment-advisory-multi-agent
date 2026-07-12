"""Seed data/profiles/CLT-XXX.json — SYNTHETIC client risk profiles.

The source portfolios.xlsx contains only holdings; risk tolerance, horizon and
income needs are invented here as a documented assumption (docs/data_assumptions.md §5).

Deliberate mismatches for the Risk agent demo:
- CLT-009: profiled CONSERVATIVE but holds 10 speculative small/mid-cap tech names
  → portfolio far riskier than tolerance.
- CLT-003: profiled AGGRESSIVE but holds a plain index-fund + cash book
  → portfolio far more conservative than tolerance.
Everyone else is roughly aligned.

Idempotent: python -m scripts.seed_profiles
"""

import json
from pathlib import Path

PROFILES_DIR = Path("data/profiles")

PROFILES = [
    # client_id, name, tolerance, risk_score, horizon_yrs, income_needs, goals
    ("CLT-001", "Margaret Whitfield", "conservative", 3, 8, True,
     "Capital preservation with steady income approaching retirement"),
    ("CLT-002", "Daniel Osei", "aggressive", 8, 20, False,
     "Long-horizon growth via technology and innovation exposure"),
    ("CLT-003", "Priya Raghavan", "aggressive", 9, 25, False,     # ← MISMATCH: book is index+cash
     "Maximum growth; comfortable with large drawdowns"),
    ("CLT-004", "Tom Becker", "moderate", 5, 15, False,
     "Balanced index-fund accumulation for mid-career wealth building"),
    ("CLT-005", "Elena Sokolova", "aggressive", 8, 18, False,
     "Concentrated mega-cap technology growth"),
    ("CLT-006", "Rajesh Nair", "moderate", 5, 12, False,
     "Diversified core portfolio with selective blue-chip stocks"),
    ("CLT-007", "Grace Lindqvist", "moderate", 6, 10, True,
     "Value-oriented equity income from established companies"),
    ("CLT-008", "Harold Kim", "conservative", 3, 6, True,        # income tilt, near retirement
     "Dividend income and municipal bond stability"),
    ("CLT-009", "Sofia Marchetti", "conservative", 2, 5, True,   # ← MISMATCH: book is speculative tech
     "Capital preservation; cannot tolerate significant losses"),
    ("CLT-010", "Ahmed Al-Rashid", "moderate", 6, 15, False,
     "Sustainable/ESG-aligned growth with healthcare stability"),
]


def main() -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    for client_id, name, tolerance, score, horizon, income, goals in PROFILES:
        profile = {
            "_note": "synthetic — not provided in source data",
            "client_id": client_id,
            "name": name,
            "risk_tolerance": tolerance,
            "risk_score": score,
            "time_horizon_years": horizon,
            "income_needs": income,
            "goals": goals,
        }
        path = PROFILES_DIR / f"{client_id}.json"
        path.write_text(json.dumps(profile, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
