"""Stream the explicitly named CSV into a private validated SQLite workspace."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import time

import pandas as pd

from common import code_hash, parser, source, write_json


def audit(data: Path, results: Path, chunk_size: int = 50000) -> dict:
    started = time.monotonic()
    if not data.is_file():
        raise FileNotFoundError(data)
    results.mkdir(parents=True, exist_ok=True)
    private = results / "private"
    private.mkdir(exist_ok=True, mode=0o700)
    database = private / "analysis.sqlite"
    if database.exists():
        raise FileExistsError(f"Use a new results directory, or explicitly remove this derived workspace: {database}")
    with data.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    names = source.canonical_names(header, {})
    required = set(source.KEYS) | set(source.NUMERIC) | set(source.BINARY) | set(source.CATEGORICAL)
    missing = required - set(names)
    if missing:
        raise ValueError(f"Required analysis columns absent: {sorted(missing)}")
    unknown = set(names) - required
    digest = hashlib.sha256()
    with data.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    cfg = source.Config(data_mode="full", allow_full_data=True)
    rows = 0
    with sqlite3.connect(database) as con:
        con.execute("PRAGMA cache_size=-32768")
        for raw in pd.read_csv(data, dtype=str, encoding="utf-8-sig", keep_default_na=False, chunksize=chunk_size):
            frame = source.normalize_schema(raw, cfg, require_targets=True)
            frame = frame.loc[:, [c for c in names if c in required]]
            frame["invitation_date"] = frame["invitation_date"].dt.strftime("%Y-%m-%d")
            frame.to_sql("invitations", con, if_exists="append", index=False, chunksize=2000)
            rows += len(frame)
        if not rows:
            raise ValueError("No invitations found.")
        # Unique indexes also detect duplicate keys spanning chunk boundaries.
        con.execute("CREATE UNIQUE INDEX invitation_key ON invitations(invitation_id)")
        con.execute("CREATE UNIQUE INDEX person_date ON invitations(person_id, invitation_date)")
        con.commit()
        counts = con.execute("""
            SELECT COUNT(*), COUNT(DISTINCT person_id), MIN(invitation_date), MAX(invitation_date)
            FROM invitations
        """).fetchone()
        fields = []
        for name in names:
            if name not in required:
                fields.append({"name": name, "role": "unknown", "inclusion": "excluded"})
                continue
            missing_n, distinct = con.execute(
                f'SELECT SUM("{name}" IS NULL), COUNT(DISTINCT "{name}") FROM invitations'
            ).fetchone()
            role = ("identifier" if name in source.KEYS[:2] else "date" if name == "invitation_date"
                    else "outcome" if name in source.TARGETS.values() else "T0_snapshot")
            definition = source.DEFINITIONS.get(name, ("", "", role))
            fields.append({
                "name": name, "role": role, "missing_n": missing_n, "missing_fraction": missing_n / rows,
                "distinct": distinct, "units": definition[1], "definition": definition[2],
                "inclusion": "audit_only_ambiguous" if name in source.AMBIGUOUS else role,
            })
        pd.DataFrame(fields).to_csv(results / "schema_audit.csv", index=False)
        numeric = []
        for name in sorted(source.NUMERIC):
            n, minimum, maximum, mean = con.execute(
                f'SELECT COUNT("{name}"), MIN("{name}"), MAX("{name}"), AVG("{name}") FROM invitations'
            ).fetchone()
            numeric.append(dict(feature=name, known=n, minimum=minimum, maximum=maximum, mean=mean))
        pd.DataFrame(numeric).to_csv(results / "numeric_audit.csv", index=False)
        counter_disagreement = con.execute("""
            SELECT SUM(prior_invitation_count != invitation_sequence),
                   SUM(prior_screen_count_5y=0 AND no_screen_observed_5y=0),
                   SUM(first_invitation=1 AND prior_invitation_count>1)
            FROM invitations
        """).fetchone()
        pd.read_sql_query("""
            SELECT substr(invitation_date,1,4) AS year, COUNT(*) AS invitations,
                   COUNT(DISTINCT person_id) AS people,
                   AVG(screening_90d) AS recorded_90d,
                   AVG(screening_180d) AS recorded_180d,
                   AVG(screening_365d) AS recorded_365d
            FROM invitations GROUP BY year ORDER BY year
        """, con).to_csv(results / "invitation_year_audit.csv", index=False)
        pd.read_sql_query("""
            SELECT invitations_per_person, COUNT(*) AS people FROM (
                SELECT COUNT(*) AS invitations_per_person FROM invitations GROUP BY person_id
            ) GROUP BY invitations_per_person
        """, con).to_csv(results / "history_coverage.csv", index=False)
    result = {
        "status": "validated", "data_filename": data.name, "data_sha256": digest.hexdigest(),
        "bytes": data.stat().st_size, "invitations": counts[0], "people": counts[1],
        "first_invitation_date": counts[2], "last_invitation_date": counts[3],
        "unknown_excluded_columns": sorted(unknown),
        "source_count_sequence_disagreements": counter_disagreement[0],
        "zero_5y_count_but_older_screen": counter_disagreement[1],
        "source_first_with_counter_above_one": counter_disagreement[2],
        "outcome_cutoff": None, "rate_status": "recorded_flags_only_followup_unverified",
        "snapshot_recording_latency": "unknown; T0 snapshot interpretation assumed",
        "utc_completed": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "code_sha256": code_hash(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "pytest"]},
    }
    write_json(results / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--chunk-size", type=int, default=50000)
    args = cli.parse_args()
    if args.chunk_size < 1:
        cli.error("--chunk-size must be positive")
    audit(args.data, args.results, args.chunk_size)
