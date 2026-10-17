"""Create T0 behavioral features and audit complete profile intersections."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from common import connect, parser, read_cohort, save_table, source, summarize, write_json


def build_features(results, rebuild=False):
    started = time.monotonic()
    with connect(results) as con:
        if con.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='features'").fetchone()[0]:
            if not rebuild:
                raise FileExistsError("Features already exist; use --rebuild to replace derived features.")
            con.execute("DROP TABLE features")
        expressions = []
        for lag in (0, 30, 90):
            # An elapsed full year + lag must end strictly before this invitation.
            window = (f"PARTITION BY person_id ORDER BY julianday(invitation_date) "
                      f"RANGE BETWEEN UNBOUNDED PRECEDING AND {366 + lag} PRECEDING")
            expressions.extend([
                f"COUNT(screening_365d) OVER ({window}) AS prior_known_{lag}",
                f"COALESCE(SUM(screening_365d) OVER ({window}),0) AS prior_yes_{lag}",
                f"COALESCE(SUM(CASE WHEN screening_365d IS NOT NULL THEN screening_90d END) "
                f"OVER ({window}),0) AS prior_early_{lag}",
                f"COALESCE(SUM(CASE WHEN screening_365d=1 AND screening_90d=0 THEN 1 ELSE 0 END) "
                f"OVER ({window}),0) AS prior_late_{lag}",
            ])
        query = """
            CREATE TABLE features AS SELECT *,
                ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY invitation_date)-1 AS observed_prior,
                CAST(julianday(invitation_date) - julianday(LAG(invitation_date) OVER (
                    PARTITION BY person_id ORDER BY invitation_date)) AS INTEGER) AS prior_gap_days,
        """ + ", ".join(expressions) + " FROM invitations"
        con.execute(query)
        con.execute("CREATE UNIQUE INDEX feature_person_date ON features(person_id, invitation_date)")
        con.commit()
        latest = read_cohort(con)
        index_year = int(latest["invitation_date"].dt.year.max())
        latest["prevention_domains"] = latest[list(source.DOMAINS)].gt(0).sum(axis=1)
        latest["gp_registration_gap"] = latest["gp_registered"].eq(1) & latest["gp_prevention_24m_cnt"].eq(0)
        latest["gyn_registration_gap"] = latest["gyn_registered"].eq(1) & latest["gyn_prevention_24m_cnt"].eq(0)
        latest["contact_band"] = pd.cut(latest["healthcare_contact_12m_cnt"],
                                        [-1, 0, 2, 10, 30, np.inf],
                                        labels=["0", "1_2", "3_10", "11_30", "31_plus"])
        latest["observed_prior_band"] = latest["observed_prior"].clip(upper=3).map(
            {0: "0", 1: "1", 2: "2", 3: "3_plus"})
        all_summary = summarize(latest, [])
        save_table(results, "cohort_summary.csv", all_summary)
        tables = []
        axes = ["history_profile", "care_profile", "screen_recency", "invitation_status",
                "registration_profile", "age_band", "territory", "contact_band", "prevention_domains",
                "gp_registration_gap", "gyn_registration_gap", "observed_prior_band"]
        for feature in axes:
            table = summarize(latest, [feature]).rename(columns={feature: "level"})
            table.insert(0, "feature", feature)
            tables.append(table)
        save_table(results, "feature_associations.csv", pd.concat(tables, ignore_index=True))
        cross = summarize(latest, ["history_profile", "care_profile"])
        save_table(results, "candidate_profiles.csv", cross)
        save_table(results, "screening_care_profiles.csv", summarize(latest, ["screen_recency", "care_profile"]))
        save_table(results, "invitation_history_profiles.csv",
                   summarize(latest, ["invitation_status", "history_profile"]))
        continuous = latest[[*source.DOMAINS, "healthcare_contact_12m_cnt",
                             "prior_screen_count_5y", "months_since_last_screen",
                             "prior_gap_days", "observed_prior"]].describe(
                                 percentiles=[.1, .25, .5, .75, .9, .99]).T
        continuous.index.name = "feature"
        continuous.to_csv(results / "feature_quantiles.csv")
        overlap = con.execute("""
            SELECT SUM(prior_gap_days <= 365), SUM(prior_gap_days <= 395),
                   SUM(observed_prior=0 AND first_invitation=0)
            FROM features
        """).fetchone()
    registry = [
        ("care_profile", "T0", "GP/GYN prevention >0; otherwise dental/other >0; otherwise all four zero and contacts <=2 or >2; missing is separate."),
        ("history_profile", "before T0", "Only prior invitations whose 365d+30d window ended strictly before T0; any early response, any delayed response, >=2 zero windows, one zero window, or none."),
        ("screen_recency", "T0", "No observed lookback screening / <24 / 24-35 / 36-59 / >=60 months; not a diagnosis or clinical eligibility rule."),
        ("invitation_status", "T0", "Source first flag kept distinct from first row observed in extract."),
        ("registration_profile", "T0", "GP/GYN valid registration combinations; not actual doctor availability."),
        ("prevention_domains", "T0", "Number of domains with observed prevention, not sum of incompatible window counts."),
        ("gp_registration_gap", "T0", "Registered GP and no recorded GP prevention in 24m."),
        ("gyn_registration_gap", "T0", "Registered gynecology and no recorded GYN prevention in 24m."),
        ("observed_prior", "before T0", "Count of earlier rows in this extract only; left-truncated."),
        ("prior_gap_days", "before T0", "Exact dispatch-date gap, not source months counter."),
        ("contact_band", "T0", "0 / 1-2 / 3-10 / 11-30 / 31+ provider-day combinations in 12m."),
        ("prior_known_0/30/90", "before T0", "Known prior 365-day labels, after full window plus lag. No current/future labels."),
        ("prior_yes_0/30/90", "before T0", "Sum of recorded prior positive 365-day flags; overlapping windows not independent events."),
        ("prior_early_0/30/90", "before T0", "Prior 90-day flags counted only after that invitation's 365-day window matures."),
        ("prior_late_0/30/90", "before T0", "Prior 365d=1 and 90d=0; missing 90d is not classified as late."),
    ]
    pd.DataFrame(registry, columns=["feature", "availability", "definition"]).to_csv(
        results / "feature_registry.csv", index=False)
    result = {
        "main_cohort": "last calendar year, latest invitation per person WITHIN that year; no future-year selection",
        "index_year": index_year,
        "people": len(latest), "profile_cells": len(cross),
        "outcomes_not_used_for_current_assignment": True,
        "history_lag_days": 30, "history_lag_status": "conservative assumption, sensitivity 0/90; not verified source delay",
        "prior_gaps_le_365": overlap[0], "prior_gaps_le_395": overlap[1],
        "first_observed_but_source_repeat_rows": overlap[2],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    write_json(results / "features.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--rebuild", action="store_true")
    args = cli.parse_args()
    build_features(args.results, args.rebuild)
