from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mammography_pipeline as m


def synthetic_frame(rows_per_year: int = 36) -> pd.DataFrame:
    """Synthetic invitations only; no real identifiers or empirical distributions."""
    rng = np.random.default_rng(2026)
    records = []
    for year in range(2015, 2025):
        for i in range(rows_per_year):
            gp, gyn, dental, other = rng.poisson([.8, .6, 1.4, .3])
            count = int(rng.integers(0, 4))
            interval = int(rng.choice(4, p=[.15, .15, .15, .55]))
            records.append({
                "invitation_id": f"SYNTH-{year}-{i:04d}", "person_id": f"SYNTH-P-{i:04d}",
                "invitation_date": pd.Timestamp(year, i % 12 + 1, 10),
                "age_band": "65-69" if i % 2 else "70-74",
                "territory": ["SYNTH_AREA_A", "SYNTH_AREA_B", None][i % 3],
                "prior_invitation_count": year - 2014, "invitation_sequence": year - 2014,
                "months_last_invitation": 12. if year > 2015 else np.nan,
                "first_invitation": float(year == 2015),
                "insurance_duration_years": year - 1999,
                "gp_registered": float(i % 4 != 0), "gyn_registered": float(i % 3 != 0),
                "gp_reg_duration_band": ">5" if i % 4 != 0 else None,
                "gyn_reg_duration_band": "2-5" if i % 3 != 0 else None,
                "prior_screen_count_5y": count, "months_since_last_screen": float(12 + i % 40) if count else np.nan,
                "no_screen_observed_5y": float(count == 0), "healthcare_contact_12m_cnt": int(rng.poisson(10)),
                "gp_prevention_24m_cnt": gp, "gyn_prevention_24m_cnt": gyn,
                "dental_prevention_24m_cnt": dental, "other_screening_36m_cnt": other,
                "screening_90d": float(interval == 0), "screening_180d": float(interval <= 1),
                "screening_365d": float(interval <= 2),
            })
    return pd.DataFrame(records)


@pytest.fixture
def cfg(tmp_path):
    return m.Config(
        date_format="%Y-%m-%d", outcome_data_end="2026-01-02", test_year=2024,
        assume_t0_snapshots=True, folds=2, models="logistic,decision_tree",
        feature_sets="minimal,raw_snapshot", preset="quick", budget_seconds=30,
        n_jobs=1, bootstrap=10, output_dir=str(tmp_path / "run"),
        provenance="SYNTHETIC mechanics test only; not empirical evidence",
    )


@pytest.fixture
def synthetic(cfg):
    return m.normalize_schema(synthetic_frame(), cfg, require_targets=True)
