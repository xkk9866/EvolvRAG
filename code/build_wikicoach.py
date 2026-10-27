"""WikiCoach: head coaches of football clubs as told by language Wikipedias.

Truth   Wikidata head-coach statements (P286) with start and end dates.
Sources language editions of Wikipedia. At every quarter boundary the claim
        of an edition is the coach named in the club's infobox in the last
        revision before that boundary.
Context the club's country.

Stages (each cached under data/wikicoach/):
  python build_wikicoach.py truth     SPARQL: clubs, countries, coach tenures
  python build_wikicoach.py revs      revision ids at each quarter boundary
  python build_wikicoach.py content   infobox field of those revisions
  python build_wikicoach.py resolve   link targets -> Wikidata ids
  python build_wikicoach.py claims    data/wikicoach/claims.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "wikicoach"
UA = {"User-Agent": "testimony-research/0.1 (academic study; contact via project page)"}
LANGS = ["en", "de", "fr", "es", "it"]
START, END = 2014, 2024          # quarters from 2014Q1 to 2023Q4
FIELDS = {
    "en": ["manager", "head coach", "headcoach", "coach"],
    "de": ["trainer", "cheftrainer"],
    "fr": ["entraîneur", "entraineur"],
    "es": ["entrenador", "director técnico", "técnico"],
    "it": ["allenatore"],
    "pt": ["treinador", "técnico", "tecnico"],
    "nl": ["trainer", "coach"],
    "ru": ["тренер", "главный тренер"],
    "tr": ["teknik direktör", "menajer", "teknikdirektör"],
    "pl": ["trener"],
    "sv": ["tränare", "huvudtränare"],
    "ar": ["المدرب", "مدرب"],
}
PAUSE = 2.0                      # one request at a time, 2 s apart
SESSION = requests.Session()
SESSION.headers.update(UA)


def get(url, params, tries=40):
    """One polite request: a pause after every call, Retry-After honored on 429/503."""
    for i in range(tries):
        try:
            r = SESSION.get(url, params={**params, "maxlag": "60"}, timeout=60)
            if r.status_code == 200:
                data = r.json()
                if data.get("error", {}).get("code") == "maxlag":
                    time.sleep(int(r.headers.get("Retry-After", 5)))
                    continue
                time.sleep(PAUSE)
                return data
            if r.status_code in (429, 503):
                time.sleep(max(int(r.headers.get("Retry-After", 0) or 0), 20 * (i + 1)))
                continue
            r.raise_for_status()
        except (requests.RequestException, ValueError):
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed {url} {params}")


def boundaries():
    out = []
    for y in range(START, END):
        for q in range(4):
            out.append(datetime(y, 1 + 3 * q, 1, tzinfo=timezone.utc))
    return out


# ----------------------------------------------------------------------------- truth
def truth():
    q = """
SELECT ?club ?country ?coach ?start ?end ?sl WHERE {
  ?club wdt:P31 wd:Q476028; wdt:P17 ?country; wikibase:sitelinks ?sl.
  FILTER(?sl >= 18)
  ?club p:P286 ?st. ?st ps:P286 ?coach. ?st pq:P580 ?start.
  OPTIONAL { ?st pq:P582 ?end. }
  FILTER(YEAR(?start) >= 2008)
}"""
    data = get("https://query.wikidata.org/sparql", {"query": q, "format": "json"})
    rows = data["results"]["bindings"]
    clubs = defaultdict(lambda: {"tenures": []})
    for b in rows:
        c = b["club"]["value"].rsplit("/", 1)[1]
        clubs[c]["country"] = b["country"]["value"].rsplit("/", 1)[1]
        clubs[c]["tenures"].append((b["coach"]["value"].rsplit("/", 1)[1], b["start"]["value"][:10],
                                    b["end"]["value"][:10] if "end" in b else None))
    # keep clubs whose dated tenures cover the whole period without gaps or overlaps
    keep = {}
    for c, info in clubs.items():
        ten = sorted(set(info["tenures"]), key=lambda x: x[1])
        cover = coverage(ten)
        if cover >= 0.6:
            keep[c] = {"country": info["country"], "tenures": ten}
    ids = list(keep)
    for i in range(0, len(ids), 50):
        ent = get("https://www.wikidata.org/w/api.php", {"action": "wbgetentities", "ids": "|".join(ids[i:i + 50]),
                                                          "props": "sitelinks|labels", "languages": "en", "format": "json"})
        for qid, e in ent["entities"].items():
            keep[qid]["sitelinks"] = {k[:-4]: v["title"] for k, v in e.get("sitelinks", {}).items()
                                      if k.endswith("wiki") and k[:-4] in LANGS}
            keep[qid]["label"] = e.get("labels", {}).get("en", {}).get("value", qid)
    keep = {c: v for c, v in keep.items() if len(v.get("sitelinks", {})) >= 4}
    (DIR / "truth.json").write_text(json.dumps(keep, ensure_ascii=False, indent=0), encoding="utf-8")
    print("clubs", len(clubs), "kept", len(keep))


def cover_ok(ten):
    """Tenures tile [START, END) with at most a 45-day gap and no overlap > 45 days."""
    lo = datetime(START, 1, 1)
    hi = datetime(END, 1, 1)
    spans = []
    for _coach, s, e in ten:
        try:
            ds = datetime.fromisoformat(s)
            de = datetime.fromisoformat(e) if e else hi
        except ValueError:
            return False
        if de <= lo or ds >= hi:
            continue
        spans.append((ds, de))
    spans.sort()
    if not spans or spans[0][0] > lo.replace(month=2, day=15):
        return False
    cur = spans[0][1]
    for ds, de in spans[1:]:
        if (ds - cur).days > 45 or (cur - ds).days > 45:
            return False
        cur = max(cur, de)
    return cur >= hi


def coach_at(tenures, when):
    """The coach in office at a boundary; None if no tenure or two tenures claim it."""
    found = set()
    for coach, s, e in tenures:
        try:
            ds = datetime.fromisoformat(s)
            de = datetime.fromisoformat(e) if e else datetime(9999, 1, 1)
        except ValueError:
            continue
        if ds <= when < de:
            found.add(coach)
    return found.pop() if len(found) == 1 else None


def coverage(ten):
    bnds = [b.replace(tzinfo=None) for b in boundaries()]
    return sum(coach_at(ten, b) is not None for b in bnds) / len(bnds)


# ----------------------------------------------------------------------------- revisions
def page_revs(lang, title):
    stamps = []
    params = {"action": "query", "prop": "revisions", "titles": title, "rvprop": "ids|timestamp",
              "rvlimit": "max", "rvdir": "newer", "rvstart": f"{START - 1}-01-01T00:00:00Z",
              "rvend": f"{END}-01-01T00:00:00Z", "format": "json", "formatversion": "2"}
    while True:
        data = get(f"https://{lang}.wikipedia.org/w/api.php", params)
        for p in data.get("query", {}).get("pages", []):
            for r in p.get("revisions", []):
                stamps.append((r["timestamp"], r["revid"]))
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
    # the last revision before START-3 carries the page into the window
    pre = get(f"https://{lang}.wikipedia.org/w/api.php",
              {"action": "query", "prop": "revisions", "titles": title, "rvprop": "ids|timestamp", "rvlimit": "1",
               "rvdir": "older", "rvstart": f"{START - 1}-01-01T00:00:00Z", "format": "json", "formatversion": "2"})
    for p in pre.get("query", {}).get("pages", []):
        for r in p.get("revisions", []):
            stamps.append((r["timestamp"], r["revid"]))
    stamps.sort()
    chosen = []
    j = -1
    for b in boundaries():
        while j + 1 < len(stamps) and datetime.fromisoformat(stamps[j + 1][0].replace("Z", "+00:00")) < b:
            j += 1
        chosen.append(stamps[j][1] if j >= 0 else None)
    return chosen


def revs():
    from concurrent.futures import ThreadPoolExecutor
    clubs = json.loads((DIR / "truth.json").read_text(encoding="utf-8"))
    path = DIR / "revs.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [(f"{c}|{lang}", lang, title) for c, info in clubs.items() for lang, title in info["sitelinks"].items()
            if f"{c}|{lang}" not in out]
    todo = [x for x in todo if x[1] in LANGS]
    print("pages to fetch", len(todo), flush=True)
    import threading
    lock = threading.Lock()

    def worker(lang):
        for key, lg, title in todo:
            if lg != lang:
                continue
            try:
                got = page_revs(lg, title)
            except Exception as exc:  # a page that fails is reported and skipped
                print("fail", key, str(exc)[:80], flush=True)
                continue
            with lock:
                out[key] = got
                if len(out) % 50 == 0:
                    path.write_text(json.dumps(out), encoding="utf-8")
                    print("revs", len(out), flush=True)

    for lang in LANGS:                # strictly sequential
        worker(lang)
    path.write_text(json.dumps(out), encoding="utf-8")
    print("revs done", len(out))


# ----------------------------------------------------------------------------- content
LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")


def field_value(text, lang):
    keys = FIELDS[lang]
    for line in text.split("\n"):
        m = re.match(r"\s*\|\s*([^=|]+?)\s*=(.*)$", line)
        if not m:
            continue
        k = m.group(1).strip().lower().replace("_", " ")
        if k in keys:
            v = m.group(2).strip()
            v = re.sub(r"<ref[^>]*/>", "", v)
            v = re.sub(r"<ref.*?</ref>", "", v, flags=re.S)
            v = re.sub(r"<!--.*?-->", "", v, flags=re.S)
            link = LINK.search(v)
            if link:
                return {"link": link.group(1).strip()}
            v = re.sub(r"\{\{[^}]*\}\}", "", v)
            v = re.sub(r"<[^>]+>", "", v).strip(" '\"")
            return {"text": v} if v else None
    return None


def content():
    rv = json.loads((DIR / "revs.json").read_text(encoding="utf-8"))
    path = DIR / "content.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    by_lang = defaultdict(set)
    for key, ids in rv.items():
        lang = key.split("|")[1]
        for r in ids:
            if r is not None and str(r) not in out:
                by_lang[lang].add(r)
    for lang, ids in by_lang.items():
        ids = sorted(ids)
        for i in range(0, len(ids), 50):
            batch = ids[i:i + 50]
            data = get(f"https://{lang}.wikipedia.org/w/api.php",
                       {"action": "query", "prop": "revisions", "revids": "|".join(map(str, batch)),
                        "rvprop": "ids|content", "rvslots": "main", "rvsection": "0",
                        "format": "json", "formatversion": "2"})
            for p in data.get("query", {}).get("pages", []):
                for r in p.get("revisions", []):
                    text = r.get("slots", {}).get("main", {}).get("content", "")
                    out[str(r["revid"])] = field_value(text, lang)
            for b in batch:
                out.setdefault(str(b), None)
            if (i // 50) % 20 == 0:
                path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
                print("content", lang, i, len(ids), flush=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("content done", len(out))


# ----------------------------------------------------------------------------- resolve
def resolve():
    ct = json.loads((DIR / "content.json").read_text(encoding="utf-8"))
    rv = json.loads((DIR / "revs.json").read_text(encoding="utf-8"))
    path = DIR / "resolve.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    need = defaultdict(set)
    for key, ids in rv.items():
        lang = key.split("|")[1]
        for r in ids:
            v = ct.get(str(r)) if r is not None else None
            if v and "link" in v and f"{lang}|{v['link']}" not in out:
                need[lang].add(v["link"])
    for lang, titles in need.items():
        titles = sorted(titles)
        for i in range(0, len(titles), 50):
            batch = titles[i:i + 50]
            data = get("https://www.wikidata.org/w/api.php",
                       {"action": "wbgetentities", "sites": f"{lang}wiki", "titles": "|".join(batch),
                        "props": "sitelinks", "sitefilter": f"{lang}wiki", "normalize": "yes" if len(batch) == 1 else "",
                        "redirects": "yes", "format": "json"})
            got = {}
            for qid, e in data.get("entities", {}).items():
                if qid.startswith("Q"):
                    t = e.get("sitelinks", {}).get(f"{lang}wiki", {}).get("title")
                    if t:
                        got[t.replace("_", " ").lower()] = qid
            for t in batch:
                out[f"{lang}|{t}"] = got.get(t.replace("_", " ").lower())
        print("resolved", lang, len(titles), flush=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    # titles that did not resolve directly may be redirects: query the wiki itself
    for lang in need:
        miss = sorted(t.split("|", 1)[1] for t, q in out.items() if q is None and t.startswith(lang + "|"))
        for i in range(0, len(miss), 50):
            batch = miss[i:i + 50]
            data = get(f"https://{lang}.wikipedia.org/w/api.php",
                       {"action": "query", "titles": "|".join(batch), "redirects": "1", "prop": "pageprops",
                        "ppprop": "wikibase_item", "format": "json", "formatversion": "2"})
            q = data.get("query", {})
            norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
            red = {r["from"]: r["to"] for r in q.get("redirects", [])}
            items = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q.get("pages", [])}
            for t in batch:
                tt = norm.get(t, t)
                tt = red.get(tt, tt)
                if items.get(tt):
                    out[f"{lang}|{t}"] = items[tt]
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("resolve done", len(out), sum(v is not None for v in out.values()))


# ----------------------------------------------------------------------------- claims
def claims():
    clubs = json.loads((DIR / "truth.json").read_text(encoding="utf-8"))
    rv = json.loads((DIR / "revs.json").read_text(encoding="utf-8"))
    ct = json.loads((DIR / "content.json").read_text(encoding="utf-8"))
    rs = json.loads((DIR / "resolve.json").read_text(encoding="utf-8"))
    bnds = boundaries()
    facts, rows = [], []
    for c, info in clubs.items():
        k = len(facts)
        gold = [coach_at([tuple(t) for t in info["tenures"]], b.replace(tzinfo=None)) for b in bnds]
        facts.append({"club": c, "label": info.get("label", c), "country": info["country"], "gold": gold})
        for lang in info["sitelinks"]:
            ids = rv.get(f"{c}|{lang}")
            if not ids:
                continue
            for t, r in enumerate(ids):
                if r is None:
                    continue
                v = ct.get(str(r))
                if not v:
                    continue
                if "link" in v:
                    val = rs.get(f"{lang}|{v['link']}") or f"{lang}:{v['link'].lower()}"
                else:
                    val = f"{lang}:{v['text'].lower()}"
                shown = v["link"] if "link" in v else v["text"]
                rows.append([k, val, lang, t + 1, int(r), shown])
    out = {"langs": LANGS, "boundaries": [b.strftime("%Y-%m-%d") for b in bnds], "facts": facts, "claims": rows}
    (DIR / "claims.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    hit = sum(1 for k, v, s, t, *_ in rows if facts[k]["gold"][t - 1] == v)
    print("facts", len(facts), "claims", len(rows), "agree with gold", round(hit / max(len(rows), 1), 3))


if __name__ == "__main__":
    DIR.mkdir(parents=True, exist_ok=True)
    {"truth": truth, "revs": revs, "content": content, "resolve": resolve, "claims": claims}[sys.argv[1]]()
