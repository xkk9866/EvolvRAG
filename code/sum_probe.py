import sys, re, collections
text = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
for line in text.splitlines():
    if " mean " not in line:
        continue
    head = line.split(" mean ")[0]
    d = collections.defaultdict(list)
    for m in re.finditer(r"(stat|topi|open|sybi|drif|over|unic|coll)(\d+):([\d.]+)/([\d.]+)", line):
        d[m.group(1)].append((float(m.group(3)), float(m.group(4))))
    s = head + " | "
    tot, btot = [], []
    for r, v in d.items():
        a = sum(x for x, _ in v) / len(v); b = sum(y for _, y in v) / len(v)
        tot.append(a); btot.append(b)
        s += f"{r} {a:.3f}/{b:.3f}  "
    print(s, "MEAN", round(sum(tot) / len(tot), 3), "ref", round(sum(btot) / len(btot), 3))
