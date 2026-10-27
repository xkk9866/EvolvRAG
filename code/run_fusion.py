"""The clearance switch on Stock and Flight.

Both corpora name the source, the day, and a value that conflicts across
sources. Stock gold is the NASDAQ page for the NASDAQ-100. Flight gold is
the airline page for the released flights. The organization of a stock key
is the attribute. The organization of a flight key is the airline. The
switch is the society rule: the organization record is used when it clears
more sources above 0.55. Nothing here retunes that rule.

A source the CRH unit drops below 0.55 can still clear on the additive
count of the same labels when every other claim agrees. That repair is a
separate row. It does not rename the switch.
"""

from __future__ import annotations

import json
import math
import random
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fusion"
OUT = ROOT / "results" / "fusion.json"
PRIOR = 0.40
ALPHA = 4.0
TAU = 0.55
ETA = 0.5
EPS = 1e-6
FLOOR = 0.18
CAP = 0.88

STOCK_ATTRS = (
    ("last", 3, 2, "num"),
    ("open", 4, 3, "num"),
    ("volume", 6, 5, "num"),
    ("high", 7, 6, "num"),
    ("low", 8, 7, "num"),
)
FLIGHT_ATTRS = (
    ("actual_dep", 3, 2, "time"),
    ("dep_gate", 4, 3, "gate"),
    ("actual_arr", 6, 5, "time"),
    ("arr_gate", 7, 6, "gate"),
)
TIME_FORMATS = (
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%Y %I:%M%p",
    "%m/%d/%y %I:%M %p",
    "%m/%d/%y %I:%M%p",
    "%m/%d/%Y %H:%M",
    "%m/%d/%y %H:%M",
)


def clip(weight):
    return min(CAP, max(FLOOR, weight))


def norm_num(text):
    raw = (text or "").strip().lower().replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    if not raw or raw in {"-", "n/a", "na", "none"}:
        return None
    try:
        return f"{float(raw):.4f}"
    except ValueError:
        return None


def norm_gate(text):
    raw = re.sub(r"\s+", "", (text or "")).upper()
    if not raw or raw in {"-", "N/A", "NA"}:
        return None
    return raw


def norm_time(text):
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"\s+(EST|EDT|CST|CDT|MST|MDT|PST|PDT|UTC)\b", "", raw, flags=re.I)
    raw = re.sub(r"\s*\([^)]*\)", "", raw).strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return None


def normalize(kind, text):
    if kind == "num":
        return norm_num(text)
    if kind == "time":
        return norm_time(text)
    return norm_gate(text)


def crh_fit(rows):
    by_slot = defaultdict(list)
    counts = defaultdict(int)
    for slot, value, issuer in rows:
        by_slot[slot].append((value, issuer))
        counts[issuer] += 1
    if not counts:
        return {}, {}, {}
    log_w = {issuer: 1.0 for issuer in counts}
    unit = {issuer: PRIOR for issuer in counts}
    truth = {}
    loss = {issuer: 0 for issuer in counts}
    for _ in range(6):
        truth = {}
        for slot, pairs in by_slot.items():
            scores = defaultdict(float)
            for value, issuer in pairs:
                scores[value] += log_w[issuer]
            truth[slot] = max(scores, key=lambda value: (scores[value], value))
        loss = {issuer: 0 for issuer in counts}
        for slot, pairs in by_slot.items():
            answer = truth[slot]
            for value, issuer in pairs:
                if value != answer:
                    loss[issuer] += 1
        peak = max(loss.values()) if loss else 0.0
        for issuer, n in counts.items():
            if peak <= 0:
                log_w[issuer] = 1.0
                raw_unit = 1.0
            else:
                log_w[issuer] = -math.log((loss[issuer] + EPS) / (peak + EPS))
                perfect = -math.log(EPS / (n + EPS))
                raw = -math.log((loss[issuer] + EPS) / (n + EPS))
                raw_unit = 0.0 if perfect <= 0 else raw / perfect
            if n < 2:
                unit[issuer] = PRIOR
            else:
                unit[issuer] = clip((PRIOR * ALPHA + raw_unit * n) / (ALPHA + n))
    learned = {issuer: unit[issuer] for issuer, n in counts.items() if n >= 2}
    hits = {issuer: counts[issuer] - loss.get(issuer, 0) for issuer in counts}
    return truth, learned, hits


def additive_from_hits(hits, n):
    if n < 2:
        return PRIOR
    return clip((PRIOR * ALPHA + (hits / n) * n) / (ALPHA + n))


def lines_of(zip_path, inner):
    with zipfile.ZipFile(zip_path) as handle:
        text = handle.read(inner).decode("utf-8", "replace")
    return [line.split("\t") for line in text.splitlines() if line.strip()]


def stock_days():
    data_zip = DATA / "clean_stock.zip"
    gold_zip = DATA / "nasdaq_truth.zip"
    with zipfile.ZipFile(data_zip) as handle:
        names = sorted(name for name in handle.namelist() if name.endswith(".txt"))
    days = []
    for name in names:
        day = name.rsplit("-", 3)[-3] + "-" + name.rsplit("-", 2)[-2] + "-" + name.rsplit("-", 1)[-1].replace(".txt", "")
        # stock-2011-07-01.txt
        day = Path(name).stem.replace("stock-", "")
        gold_name = f"nasdaq_truth/stock-{day}-nasdaq-com.txt"
        claims = []
        for parts in lines_of(data_zip, name):
            if len(parts) < 9:
                continue
            source, symbol = parts[0], parts[1].lower()
            for attr, data_i, _gold_i, kind in STOCK_ATTRS:
                if data_i >= len(parts):
                    continue
                value = normalize(kind, parts[data_i])
                if value is None:
                    continue
                claims.append((f"{symbol}|{attr}", value, source, attr))
        gold = {}
        for parts in lines_of(gold_zip, gold_name):
            if len(parts) < 8:
                continue
            symbol = parts[0].lower()
            for attr, _data_i, gold_i, kind in STOCK_ATTRS:
                if gold_i >= len(parts):
                    continue
                value = normalize(kind, parts[gold_i])
                if value is None:
                    continue
                gold[f"{symbol}|{attr}"] = value
        days.append({"day": day, "claims": claims, "gold": gold})
    return days


def flight_days():
    data_zip = DATA / "clean_flight.zip"
    gold_zip = DATA / "flight_truth.zip"
    with zipfile.ZipFile(data_zip) as handle:
        names = sorted(name for name in handle.namelist() if name.endswith("-data.txt"))
    days = []
    for name in names:
        day = Path(name).name.replace("-data.txt", "")
        gold_name = f"flight_truth/{day}-truth.txt"
        claims = []
        for parts in lines_of(data_zip, name):
            if len(parts) < 8:
                continue
            source, flight = parts[0], parts[1].upper()
            org = flight.split("-")[0]
            for attr, data_i, _gold_i, kind in FLIGHT_ATTRS:
                if data_i >= len(parts):
                    continue
                value = normalize(kind, parts[data_i])
                if value is None:
                    continue
                claims.append((f"{flight}|{attr}", value, source, org))
        gold = {}
        try:
            gold_rows = lines_of(gold_zip, gold_name)
        except KeyError:
            gold_rows = []
        for parts in gold_rows:
            if len(parts) < 7:
                continue
            flight = parts[0].upper()
            for attr, _data_i, gold_i, kind in FLIGHT_ATTRS:
                if gold_i >= len(parts):
                    continue
                value = normalize(kind, parts[gold_i])
                if value is None:
                    continue
                gold[f"{flight}|{attr}"] = value
        days.append({"day": day, "claims": claims, "gold": gold})
    return days


def rate_fit(rows):
    """Majority label, then the additive weight. One loss does not zero the source."""
    by_slot = defaultdict(list)
    for slot, value, issuer in rows:
        by_slot[slot].append((value, issuer))
    truth = {}
    for slot, pairs in by_slot.items():
        buckets = defaultdict(int)
        for value, _issuer in pairs:
            buckets[value] += 1
        truth[slot] = max(buckets, key=lambda value: (buckets[value], value))
    hits = defaultdict(int)
    counts = defaultdict(int)
    for slot, value, issuer in rows:
        counts[issuer] += 1
        if value == truth[slot]:
            hits[issuer] += 1
    return {issuer: additive_from_hits(hits[issuer], n) for issuer, n in counts.items() if n >= 2}


class Books:
    def __init__(self):
        self.soc = {}
        self.org_hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.crh_soc = {}
        self.crh_hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.used_org = 0
        self.used_sum = 0
        self.decisions = 0
        self.belief = {name: {} for name in ("majority", "society", "org", "hier", "sum", "crh")}

    def org_weight(self, org, issuer, crh=False):
        hits, n = (self.crh_hits if crh else self.org_hits)[org][issuer]
        if crh:
            losses = n - hits
            if n < 2:
                return PRIOR
            perfect = -math.log(EPS / (n + EPS))
            raw = -math.log((losses + EPS) / (n + EPS))
            raw_unit = 0.0 if perfect <= 0 else raw / perfect
            return clip((PRIOR * ALPHA + raw_unit * n) / (ALPHA + n))
        return additive_from_hits(hits, n)

    def weights(self, org, kind):
        crh = kind == "crh"
        soc = self.crh_soc if crh else self.soc
        if kind == "society":
            return soc
        local = {}
        table = self.crh_hits if crh else self.org_hits
        for issuer in table[org]:
            local[issuer] = self.org_weight(org, issuer, crh)
        if kind == "org":
            return local
        if kind == "sum":
            soc_mass = sum(weight for weight in soc.values() if weight + 1e-9 >= TAU)
            org_mass = sum(weight for weight in local.values() if weight + 1e-9 >= TAU)
            if org_mass > soc_mass:
                self.used_sum += 1
                return local
            return soc
        soc_high = sum(weight + 1e-9 >= TAU for weight in soc.values())
        org_high = sum(weight + 1e-9 >= TAU for weight in local.values())
        if org_high >= soc_high + 1:
            if kind == "hier":
                self.used_org += 1
            return local
        return soc

    def update(self, claims):
        rows = [(slot, value, issuer) for slot, value, issuer, _org in claims]
        for issuer, weight in rate_fit(rows).items():
            old = self.soc.get(issuer, PRIOR)
            self.soc[issuer] = (1.0 - ETA) * old + ETA * weight
        _truth, learned, _hits = crh_fit(rows)
        for issuer, weight in learned.items():
            old = self.crh_soc.get(issuer, PRIOR)
            self.crh_soc[issuer] = (1.0 - ETA) * old + ETA * weight
        grouped = defaultdict(list)
        for slot, value, issuer, org in claims:
            grouped[org].append((slot, value, issuer))
        for org, fresh in grouped.items():
            by_slot = defaultdict(list)
            for slot, value, issuer in fresh:
                by_slot[slot].append((value, issuer))
            truth = {}
            for slot, pairs in by_slot.items():
                buckets = defaultdict(int)
                for value, _issuer in pairs:
                    buckets[value] += 1
                truth[slot] = max(buckets, key=lambda value: (buckets[value], value))
            _ctruth, _clearned, chits = crh_fit(fresh)
            seen = set()
            for slot, value, issuer in fresh:
                if (slot, issuer) in seen:
                    continue
                seen.add((slot, issuer))
                self.org_hits[org][issuer][1] += 1
                if value == truth[slot]:
                    self.org_hits[org][issuer][0] += 1
                self.crh_hits[org][issuer][1] += 1
                if _ctruth.get(slot) == value:
                    self.crh_hits[org][issuer][0] += 1
            del chits

    def answer(self, claims, gold, previous_gold):
        by_slot = defaultdict(list)
        org_of = {}
        for slot, value, issuer, org in claims:
            by_slot[slot].append((value, issuer))
            org_of[slot] = org
        scores = {name: {"hit": 0, "n": 0, "changed_hit": 0, "changed_n": 0} for name in self.belief}
        item = {name: [] for name in self.belief}
        changed_item = {name: [] for name in self.belief}
        for slot, answer in gold.items():
            self.decisions += 1
            pairs = by_slot.get(slot, [])
            org = org_of.get(slot)
            if org is None and "|" in slot:
                head = slot.split("|")[0]
                org = head.split("-")[0] if "-" in head else slot.split("|")[1]
            majority = None
            if pairs:
                buckets = defaultdict(set)
                for value, issuer in pairs:
                    buckets[value].add(issuer)
                majority = max(buckets, key=lambda value: (len(buckets[value]), value))
            chosen = {}
            for name in self.belief:
                if name == "majority":
                    best = majority
                else:
                    weights = self.weights(org, name) if org is not None else (self.crh_soc if name == "crh" else self.soc)
                    best = None
                    best_key = None
                    for value, issuer in pairs:
                        weight = weights.get(issuer, PRIOR)
                        if weight + 1e-9 < TAU:
                            continue
                        key = (weight, issuer)
                        if best_key is None or key > best_key:
                            best = value
                            best_key = key
                    if best is None:
                        best = self.belief[name].get(slot)
                chosen[name] = best
                if best is not None:
                    self.belief[name][slot] = best
            changed = previous_gold is not None and slot in previous_gold and previous_gold[slot] != answer
            for name, value in chosen.items():
                hit = int(value == answer)
                scores[name]["n"] += 1
                scores[name]["hit"] += hit
                item[name].append(hit)
                if changed:
                    scores[name]["changed_n"] += 1
                    scores[name]["changed_hit"] += hit
                    changed_item[name].append(hit)
        scores["_item"] = item
        scores["_changed_item"] = changed_item
        return scores


def mean_std(values):
    n = len(values)
    mu = sum(values) / n
    var = sum((value - mu) ** 2 for value in values) / (n - 1)
    return mu, math.sqrt(var)


def perm_p(diffs):
    obs = abs(sum(diffs))
    if obs <= 1e-12:
        return 1.0
    wins = sum(diff > 0 for diff in diffs)
    losses = sum(diff < 0 for diff in diffs)
    rng = random.Random(3157)
    count = 1
    for _ in range(99999):
        plus_wins = rng.getrandbits(wins).bit_count() if wins else 0
        plus_losses = rng.getrandbits(losses).bit_count() if losses else 0
        signed = (2 * plus_wins - wins) - (2 * plus_losses - losses)
        if abs(signed) + 1e-9 >= obs:
            count += 1
    return max(count / 100000.0, 1e-5)


def paired(left, right):
    diffs = [a - b for a, b in zip(left, right)]
    mu, sd = mean_std(diffs)
    return {"mean": mu, "std": sd, "p": perm_p(diffs), "n": len(diffs)}


def majority_value(pairs):
    buckets = defaultdict(set)
    for value, issuer in pairs:
        buckets[value].add(issuer)
    return max(buckets, key=lambda value: (len(buckets[value]), value))


def cleared_speaker(weights, pairs):
    best = None
    best_key = None
    for value, issuer in pairs:
        weight = weights.get(issuer, PRIOR)
        if weight + 1e-9 < TAU:
            continue
        key = (weight, issuer)
        if best_key is None or key > best_key:
            best = {"value": value, "issuer": issuer, "weight": weight}
            best_key = key
    return best


def inject(book, claims, gold, mode):
    by_slot = defaultdict(list)
    for index, (slot, value, issuer, _org) in enumerate(claims):
        by_slot[slot].append((index, value, issuer))
    for issuer in sorted(book.soc):
        if book.soc[issuer] + 1e-9 < 0.70:
            continue
        owned = []
        for slot, rows in by_slot.items():
            for index, value, who in rows:
                if who == issuer:
                    owned.append((slot, index, value, rows))
        if len(owned) < 2:
            continue
        for slot, index, value, rows in sorted(owned):
            answer = gold.get(slot)
            if answer is None:
                continue
            pairs = [(item_value, who) for _index, item_value, who in rows]
            vote = majority_value(pairs)
            if mode == "slip" and value == answer:
                _slot, _value, who, org = claims[index]
                claims[index] = (slot, "INJECTED_LOSS", who, org)
                return {"mode": mode, "issuer": issuer, "slot": slot, "org": org, "old": value, "new": "INJECTED_LOSS"}
            if mode == "pioneer" and value == vote and vote != answer:
                _slot, _value, who, org = claims[index]
                claims[index] = (slot, answer, who, org)
                return {"mode": mode, "issuer": issuer, "slot": slot, "org": org, "old": value, "new": answer, "majority": vote}
    return None


def recovery_trace(days, mode):
    book = Books()
    found = None
    series = []
    for index, day in enumerate(days):
        claims = [tuple(row) for row in day["claims"]]
        if index == 0:
            book.update(claims)
            continue
        if found is None:
            found = inject(book, claims, day["gold"], mode)
            if found is not None:
                found["day"] = day["day"]
                found["day_index"] = index
        if found is None:
            book.update(claims)
            continue
        issuer = found["issuer"]
        org = found["org"]
        slot = found["slot"]
        pairs = [(value, who) for key, value, who, _org in claims if key == slot]
        speaker = cleared_speaker(book.soc, pairs)
        crh_speaker = cleared_speaker(book.crh_soc, pairs)
        before = {
            "day": day["day"],
            "soc": book.soc.get(issuer, PRIOR),
            "crh_soc": book.crh_soc.get(issuer, PRIOR),
            "org": book.org_weight(org, issuer, False),
            "crh_org": book.org_weight(org, issuer, True),
            "spoke": speaker["issuer"] == issuer if speaker else False,
            "spoke_gold": bool(speaker and speaker["value"] == day["gold"].get(slot)),
            "crh_spoke": crh_speaker["issuer"] == issuer if crh_speaker else False,
        }
        book.update(claims)
        before["soc_after"] = book.soc.get(issuer, PRIOR)
        before["crh_soc_after"] = book.crh_soc.get(issuer, PRIOR)
        before["org_after"] = book.org_weight(org, issuer, False)
        before["crh_org_after"] = book.org_weight(org, issuer, True)
        series.append(before)
    return {"found": found, "series": series}


def run_corpus(name, days):
    book = Books()
    totals = {key: {"hit": 0, "n": 0, "changed_hit": 0, "changed_n": 0} for key in book.belief}
    items = {key: [] for key in book.belief}
    changed_items = {key: [] for key in book.belief}
    usable = 0
    for index, day in enumerate(days):
        if index == 0:
            book.update(day["claims"])
            continue
        previous = days[index - 1]["gold"]
        scores = book.answer(day["claims"], day["gold"], previous)
        book.update(day["claims"])
        usable += len(day["gold"])
        for key in book.belief:
            row = scores[key]
            for field in totals[key]:
                totals[key][field] += row[field]
            items[key].extend(scores["_item"][key])
            changed_items[key].extend(scores["_changed_item"][key])
        print(name, day["day"], "gold", len(day["gold"]), "hier", round(scores["hier"]["hit"] / max(scores["hier"]["n"], 1), 3), flush=True)
    summary = {}
    for key, row in totals.items():
        summary[key] = {
            "em": row["hit"] / row["n"] if row["n"] else None,
            "changed": row["changed_hit"] / row["changed_n"] if row["changed_n"] else None,
            "n": row["n"],
            "changed_n": row["changed_n"],
        }
    tests = []
    for left, right in (("society", "majority"), ("society", "crh"), ("hier", "society"), ("sum", "society"), ("sum", "hier")):
        row = paired(items[left], items[right])
        row.update({"left": left, "right": right, "slice": "all"})
        tests.append(row)
        changed = paired(changed_items[left], changed_items[right])
        changed.update({"left": left, "right": right, "slice": "changed"})
        tests.append(changed)
    return {
        "days": len(days) - 1,
        "gold": usable,
        "org_decisions": book.used_org,
        "sum_decisions": book.used_sum,
        "decisions": book.decisions,
        "summary": summary,
        "tests": tests,
    }


def main():
    print("stock", flush=True)
    stock_rows = stock_days()
    stock = run_corpus("stock", stock_rows)
    print("flight", flush=True)
    flight = run_corpus("flight", flight_days())
    print("recovery", flush=True)
    recovery = {
        "slip": recovery_trace(stock_rows, "slip"),
        "pioneer": recovery_trace(stock_rows, "pioneer"),
    }
    payload = {
        "stock": stock,
        "flight": flight,
        "recovery": recovery,
        "tau": TAU,
        "eta": ETA,
        "margin": 1,
        "stream_weight": "additive",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
