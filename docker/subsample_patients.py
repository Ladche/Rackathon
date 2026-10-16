"""Stream a seeded WHOLE-PATIENT subsample of a large invitation CSV.

All invitations of every selected person are kept, so patient histories stay
complete. Unlike --sample-cap (applied after the whole file is materialized),
this never holds the full file in memory.
"""
import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--people", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--person-column", default="PERSON_ID")
    ap.add_argument("--encoding", default="utf-8-sig")
    a = ap.parse_args()

    kw = dict(dtype=str, keep_default_na=False, encoding=a.encoding, chunksize=200_000)
    people = set()
    for chunk in pd.read_csv(a.input, usecols=[a.person_column], **kw):
        people.update(chunk[a.person_column])
    ordered = np.array(sorted(people))
    if a.people >= len(ordered):
        chosen = set(ordered)
    else:
        chosen = set(np.random.default_rng(a.seed).choice(ordered, a.people, replace=False))

    rows = 0
    first = True
    for chunk in pd.read_csv(a.input, **kw):
        part = chunk[chunk[a.person_column].isin(chosen)]
        part.to_csv(a.output, mode="w" if first else "a", header=first, index=False)
        first = False
        rows += len(part)
    print(f"    Subsample: {len(chosen)} of {len(ordered)} people, {rows} invitations -> {a.output}")


if __name__ == "__main__":
    main()
