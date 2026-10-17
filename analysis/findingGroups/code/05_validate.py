"""Independently verify stored histories, main assignments and published totals."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import BASE, code_hash, connect, parser, read_cohort, read_json, summarize, write_json

groups_module = importlib.import_module("03_groups")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(results: Path, documentation: Path):
    audit = read_json(results / "audit.json")
    groups = read_json(results / "groups.json")
    manifest = read_json(results / "run_manifest.json")
    table = pd.read_csv(results / "groups.csv").set_index("group_id")
    with connect(results) as con:
        total = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        require(total == audit["invitations"], "Not all invitations have features.")
        assignments, people = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT person_id) FROM invitation_assignments").fetchone()
        require(assignments == total and people == audit["people"], "Incomplete full-data grouping.")
        full_table = pd.read_csv(results / "all_invitation_groups.csv").set_index("group_id")
        full_counts = pd.read_sql_query(
            "SELECT group_id, COUNT(*) AS n FROM invitation_assignments GROUP BY group_id",
            con).set_index("group_id")
        require(full_counts["n"].equals(full_table.loc[full_counts.index, "n"]), "Full-data totals mismatch.")
        for lag in (0, 30, 90):
            differences = con.execute(f"""
                SELECT COUNT(*) FROM features f
                WHERE prior_known_{lag} != (
                    SELECT COUNT(p.screening_365d) FROM invitations p
                    WHERE p.person_id=f.person_id
                      AND p.invitation_date < date(f.invitation_date, '-{365+lag} days')
                ) OR prior_yes_{lag} != (
                    SELECT COALESCE(SUM(p.screening_365d),0) FROM invitations p
                    WHERE p.person_id=f.person_id
                      AND p.invitation_date < date(f.invitation_date, '-{365+lag} days')
                )
            """).fetchone()[0]
            require(differences == 0, f"Independent history mismatch for lag {lag}.")
        frame = read_cohort(con, year=groups["index_year"])
        frame["group_id"] = groups_module.assign(frame, groups["mapping"])
        actual = summarize(frame, ["group_id"]).set_index("group_id")
        require(set(actual.index) == set(table.index), "Missing or extra published group.")
        for col in ["n", "people", *[f"attended_{h}d" for h in (90, 180, 365)]]:
            require(actual[col].equals(table.loc[actual.index, col]), f"Aggregate mismatch: {col}.")
        for horizon in (90, 180, 365):
            require(np.allclose(actual[f"rate_{horizon}d"], table.loc[actual.index, f"rate_{horizon}d"]),
                    f"Rate mismatch: {horizon}.")
        require(len(frame) == groups["people"] == table["n"].sum(), "Main cohort coverage mismatch.")
        stored = pd.read_sql_query("SELECT * FROM main_assignments", con)
        require(len(stored) == len(frame) and stored["person_id"].nunique() == len(frame),
                "Patient assignment is not one-to-one.")
        merged = frame[["invitation_id", "group_id"]].merge(stored, on="invitation_id", validate="one_to_one")
        require(merged["group_id_x"].equals(merged["group_id_y"]), "Persisted assignment mismatch.")
    require(manifest["final_code_sha256"] == code_hash(), "Reports use a stale code version.")
    require(manifest["models_fitted"] == 0, "ML must remain outside this workflow.")
    for path in results.glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        require(not columns.intersection({"person_id", "invitation_id"}), f"Identifier export: {path.name}.")
    text = (documentation / "interpretation" / "groups.md").read_text(encoding="utf-8")
    require(text.count("\n## ") + int(text.startswith("## ")) == len(table), "Incorrect group-card count.")
    for group_id in table.index:
        require(text.count(f"`{group_id}`") == 1, f"Group card missing/duplicated: {group_id}.")
    require(all(line.startswith("## ") or line.startswith("- ") or not line.strip() for line in text.splitlines()),
            "Interpretation must contain only consecutive group cards.")
    require(table["n"].ge(groups["minimum_support"]).all(), "Unsupported group needs explicit review.")
    result = {
        "status": "passed", "full_data_history_checks": total * 3,
        "main_people": len(frame), "groups": len(table),
        "all_source_invitations_assigned": assignments, "all_source_people_covered": people,
        "persisted_assignment_matches": True, "aggregate_counts_and_rates_match": True,
        "identifier_columns_in_public_csv": False, "group_cards_match": True,
        "code_hash_matches": True, "models_fitted": 0,
    }
    write_json(results / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--documentation", type=Path, default=BASE)
    args = cli.parse_args()
    validate(args.results, args.documentation)
