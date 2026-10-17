"""Discover supported rule intersections and measure stability, not latent clusters."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import (
    add_profiles, connect, parser, read_cohort, read_json, save_table,
    source, summarize, write_json,
)

MERGEABLE = ("low_contact", "care_no_prevention")


def action_history(frame: pd.DataFrame) -> pd.Series:
    result = frame["history_profile"].replace({
        "early_response": "past_response", "delayed_response": "past_response",
        "response_timing_unknown": "past_response",
    }).copy()
    unobserved = result.eq("unobserved")
    result.loc[unobserved & frame["first_invitation"].eq(1)] = "unobserved_first"
    result.loc[unobserved & frame["first_invitation"].eq(0)] = "unobserved_repeat"
    return result


def discover_mapping(frame: pd.DataFrame, minimum: int = 500) -> dict[str, str]:
    """Pool only substantively adjacent sparse cells; never optimize on outcomes."""
    history = action_history(frame)
    not_recent = ~frame["months_since_last_screen"].lt(24)
    counts = pd.crosstab(history[not_recent], frame.loc[not_recent, "care_profile"])
    mapping = {}
    for h in counts.index:
        sparse = any(0 < counts.loc[h].get(c, 0) < minimum for c in MERGEABLE)
        for care in frame["care_profile"].unique():
            target = "no_prevention" if sparse and care in MERGEABLE else care
            mapping[f"{h}__{care}"] = f"{h}__{target}"
    return mapping


def assign(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    raw = action_history(frame) + "__" + frame["care_profile"]
    result = raw.map(mapping).fillna(raw)
    result.loc[frame["months_since_last_screen"].lt(24)] = "recent_screen"
    return result.rename("group_id")


def sample_size(p0: float, difference: float, alpha: float = .05, power: float = .8) -> int | None:
    """Normal approximation for equal-size independent patient-level trial arms."""
    p1 = p0 + difference
    if not 0 < p0 < p1 < 1:
        return None
    pbar = (p0 + p1) / 2
    numerator = (norm.ppf(1 - alpha / 2) * math.sqrt(2 * pbar * (1 - pbar))
                 + norm.ppf(power) * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1)))
    return math.ceil(numerator * numerator / (difference * difference))


def group_characteristics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group in frame.groupby("group_id", observed=True):
        row = {"group_id": group_id, "n": len(group)}
        for field in ["healthcare_contact_12m_cnt", "prior_screen_count_5y", "months_since_last_screen",
                      "observed_prior", "prior_known_30", *source.DOMAINS]:
            row[f"{field}_median"] = group[field].median()
            row[f"{field}_known"] = group[field].notna().sum()
        flags = {
            "no_screen_lookback": group["no_screen_observed_5y"].eq(1),
            "source_first": group["first_invitation"].eq(1),
            "no_gp_registration": group["gp_registered"].eq(0),
            "no_gyn_registration": group["gyn_registered"].eq(0),
            "gp_without_prevention": group["gp_registered"].eq(1) & group["gp_prevention_24m_cnt"].eq(0),
            "gyn_without_prevention": group["gyn_registered"].eq(1) & group["gyn_prevention_24m_cnt"].eq(0),
            "prior_early_response": group["prior_early_30"].gt(0),
            "prior_late_response": group["prior_late_30"].gt(0),
            "age_40_49": group["age_band"].isin(["40-44", "45-49"]),
            "age_65_plus": group["age_band"].isin(["65-69", "70-74", "75+"]),
        }
        row.update({f"{name}_fraction": values.mean() for name, values in flags.items()})
        row["modal_age_band"] = str(group["age_band"].mode().iloc[0])
        row["modal_age_fraction"] = group["age_band"].eq(row["modal_age_band"]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    """Age/year descriptive standardization on COMMON supported strata only."""
    frame = frame.copy()
    frame["year"] = frame["invitation_date"].dt.year
    frame["age_stratum"] = np.select(
        [frame["age_band"].isin(["40-44", "45-49"]),
         frame["age_band"].isin(["50-54", "55-59"]),
         frame["age_band"].isin(["60-64", "65-69"]),
         frame["age_band"].isin(["70-74", "75+"])],
        ["40_49", "50_59", "60_69", "70_plus"], default="unknown",
    )
    baseline = frame.groupby("age_stratum", observed=True)["screening_180d"].agg(["count", "mean"])
    rows = []
    for group_id, group in frame.groupby("group_id", observed=True):
        cells = group.groupby("age_stratum", observed=True)["screening_180d"].agg(["count", "mean"])
        common = cells.index[(cells["count"] >= 20) & cells.index.isin(baseline.index)]
        weights = baseline.loc[common, "count"]
        covered = weights.sum() / baseline["count"].sum()
        if len(common):
            weights = weights / weights.sum()
            rate = (cells.loc[common, "mean"] * weights).sum()
            reference = (baseline.loc[common, "mean"] * weights).sum()
        else:
            rate = reference = math.nan
        rows.append({
            "group_id": group_id, "n": len(group),
            "age_standardized_180d": rate, "same_strata_reference_180d": reference,
            "reference_population_coverage": covered, "supported_age_strata": len(common),
        })
    return pd.DataFrame(rows)


def run(results, minimum=500):
    started = time.monotonic()
    features = read_json(results / "features.json")
    with connect(results) as con:
        frame = read_cohort(con, year=features["index_year"])
        mapping = discover_mapping(frame, minimum)
        frame["group_id"] = assign(frame, mapping)
        summary = summarize(frame, ["group_id"])
        summary["share"] = summary["n"] / len(frame)
        summary["no_recorded_screen_365d"] = summary["known_365d"] - summary["attended_365d"]
        summary["share_of_nonattendance"] = summary["no_recorded_screen_365d"] / summary["no_recorded_screen_365d"].sum()
        summary["operational_support"] = summary["n"].ge(minimum)
        overall = summarize(frame, []).iloc[0]
        summary["difference_180d_pp"] = 100 * (summary["rate_180d"] - overall["rate_180d"])
        summary["extra_per_1pp"] = summary["n"] * .01
        summary["scenario_only_5pp_extra"] = summary["n"] * .05
        summary["recorded_90_180_interval"] = summary["attended_180d"] - summary["attended_90d"]
        summary["recorded_180_365_interval"] = summary["attended_365d"] - summary["attended_180d"]
        save_table(results, "groups.csv", summary)
        save_table(results, "group_characteristics.csv", group_characteristics(frame))
        save_table(results, "age_standardization.csv", standardize(frame))
        context = []
        for axis in ["age_band", "territory", "screen_recency", "registration_profile", "history_profile"]:
            table = summarize(frame, ["group_id", axis]).rename(columns={axis: "level"})
            table.insert(1, "feature", axis)
            context.append(table)
        save_table(results, "group_context.csv", pd.concat(context, ignore_index=True))
        frame[["person_id", "invitation_id", "group_id"]].to_sql(
            "main_assignments", con, if_exists="replace", index=False)
        con.execute("CREATE UNIQUE INDEX main_assignment_key ON main_assignments(invitation_id)")
        con.commit()
        for index, chunk in enumerate(pd.read_sql_query("SELECT * FROM features", con, chunksize=50000)):
            chunk = add_profiles(chunk)
            chunk["group_id"] = assign(chunk, mapping)
            chunk[["person_id", "invitation_id", "group_id"]].to_sql(
                "invitation_assignments", con, if_exists="replace" if index == 0 else "append", index=False)
        con.execute("CREATE UNIQUE INDEX full_assignment_key ON invitation_assignments(invitation_id)")
        con.commit()
        all_invitation_summary = pd.read_sql_query("""
            SELECT a.group_id, COUNT(*) AS n, COUNT(DISTINCT a.person_id) AS people,
                   COUNT(i.screening_90d) AS known_90d, SUM(i.screening_90d) AS attended_90d,
                   AVG(i.screening_90d) AS rate_90d,
                   COUNT(i.screening_180d) AS known_180d, SUM(i.screening_180d) AS attended_180d,
                   AVG(i.screening_180d) AS rate_180d,
                   COUNT(i.screening_365d) AS known_365d, SUM(i.screening_365d) AS attended_365d,
                   AVG(i.screening_365d) AS rate_365d
            FROM invitation_assignments a JOIN invitations i USING (invitation_id)
            GROUP BY a.group_id
        """, con)
        save_table(results, "all_invitation_groups.csv", all_invitation_summary)
        annual = []
        for year in range(int(read_json(results / "audit.json")["first_invitation_date"][:4]),
                          features["index_year"] + 1):
            yearly = read_cohort(con, year=year)
            yearly["group_id"] = assign(yearly, mapping)
            table = summarize(yearly, ["group_id"])
            table.insert(0, "year", year)
            annual.append(table)
        save_table(results, "annual_groups.csv", pd.concat(annual, ignore_index=True))
        sensitivity = []
        for lag in (0, 30, 90):
            for threshold in (0, 2, 5):
                changed = add_profiles(frame, low_contact=threshold, lag=lag)
                changed["group_id"] = assign(changed, mapping)
                table = summarize(changed, ["group_id"])
                table.insert(0, "contact_threshold", threshold)
                table.insert(0, "history_lag_days", lag)
                table["assignment_changed_fraction"] = changed["group_id"].ne(frame["group_id"]).mean()
                sensitivity.append(table)
        save_table(results, "threshold_sensitivity.csv", pd.concat(sensitivity, ignore_index=True))
        cohort_sensitivity = []
        for policy in ("earliest", "latest"):
            other = read_cohort(con, policy)
            other["group_id"] = assign(other, mapping)
            table = summarize(other, ["group_id"])
            table.insert(0, "policy", policy + "_across_years_selection_biased")
            cohort_sensitivity.append(table)
        save_table(results, "cohort_sensitivity.csv", pd.concat(cohort_sensitivity, ignore_index=True))
    # Future trial inputs are scenarios, not predictions or measured effects.
    power = []
    tests = int(summary["group_id"].ne("recent_screen").sum())
    for row in summary.to_dict("records"):
        if row["group_id"] == "recent_screen":
            continue
        for delta in (.02, .05):
            for alpha in (.05, .05 / tests):
                per_arm = sample_size(row["rate_180d"], delta, alpha)
                power.append({
                    "group_id": row["group_id"], "n": row["n"], "baseline_recorded_180d": row["rate_180d"],
                    "scenario_absolute_increase": delta, "alpha_two_sided": alpha, "power": .8,
                    "per_arm": per_arm, "total": 2 * per_arm if per_arm else None,
                    "fits_entire_historical_group": bool(per_arm and 2 * per_arm <= row["n"]),
                })
    save_table(results, "trial_power_scenarios.csv", pd.DataFrame(power))
    support = []
    for candidate_min in (250, 500, 1000, 2000):
        alternate = assign(frame, discover_mapping(frame, candidate_min))
        counts = alternate.value_counts()
        support.append({
            "minimum_support": candidate_min, "groups": len(counts),
            "groups_below_support": int(counts.lt(candidate_min).sum()),
            "people_below_support": int(counts[counts.lt(candidate_min)].sum()),
        })
    pd.DataFrame(support).to_csv(results / "support_sensitivity.csv", index=False)
    metadata = {
        "index_year": features["index_year"], "people": len(frame), "groups": len(summary),
        "minimum_support": minimum, "all_people_assigned_once": bool(summary["n"].sum() == len(frame)),
        "all_source_invitations_assigned": int(all_invitation_summary["n"].sum()),
        "operational_groups": int(summary["operational_support"].sum()),
        "rule": "recent-screening safety override, then history_action x care_profile; sparse adjacent zero-prevention cells pooled",
        "outcomes_used_to_choose_rules": False, "status": "exploratory descriptive partition; not validated latent classes",
        "rate_status": "recorded only; follow-up cutoff unknown",
        "selection_warning": "All feature exploration is exploratory; primary fixed-year cohort avoids conditioning on future invitation disappearance.",
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "mapping": mapping,
    }
    write_json(results / "groups.json", metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "mapping"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--minimum-support", type=int, default=500)
    args = cli.parse_args()
    if args.minimum_support < 20:
        cli.error("Minimum support must be at least 20.")
    run(args.results, args.minimum_support)
