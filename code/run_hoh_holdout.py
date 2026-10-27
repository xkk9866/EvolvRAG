"""Held-out HoH slice. The public dataset exposes only a train split.

The lexical margin stays at 0.85, the value fixed before the first 1000 rows
were scored. This script scores the next 1000 qualifying training rows and
does not choose the margin. It is not an official test set and it is not a
generation score.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import run_hoh as hoh

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "hoh_rows_2000.json"
OUT = ROOT / "results" / "hoh_holdout.json"
TARGET = 2000


def fetch():
    if CACHE.exists():
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if len(cached) >= TARGET:
            return cached[:TARGET]
    rows = []
    offset = 0
    while len(rows) < TARGET:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={hoh.DATASET}&config=default&split=train&offset={offset}&length={hoh.PAGE}"
        )
        payload = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=120) as handle:
                    payload = json.loads(handle.read().decode("utf-8"))
                break
            except Exception as exc:
                print("retry", offset, attempt, type(exc).__name__, flush=True)
                time.sleep(2 + attempt * 2)
        if payload is None:
            break
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            row = item.get("row", item)
            outdated = row.get("outdated_infos") or []
            if not outdated:
                continue
            rows.append(
                {
                    "question": row.get("question") or "",
                    "answer": row.get("answer") or "",
                    "evidence": row.get("evidence") or "",
                    "time": str(row.get("last_modified_time") or ""),
                    "outdated": [
                        {
                            "answer": old.get("answer") or "",
                            "evidence": old.get("evidence") or "",
                            "time": str(old.get("last_modified_time") or ""),
                        }
                        for old in outdated
                        if old.get("evidence")
                    ],
                }
            )
            if len(rows) >= TARGET:
                break
        offset += hoh.PAGE
        print("offset", offset, "kept", len(rows), flush=True)
        if offset > 20000:
            break
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def main():
    rows = fetch()
    first = hoh.evaluate(rows[:1000])
    second = hoh.evaluate(rows[1000:2000]) if len(rows) >= 2000 else None
    report = {
        "source": hoh.DATASET,
        "splits_exposed": ["train"],
        "declared_margin": hoh.MARGIN,
        "first_1000": first,
        "next_1000": second,
        "n_fetched": len(rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
