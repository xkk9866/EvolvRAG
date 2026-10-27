#!/bin/bash
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for r in base domain cliques laggy; do for s in 0 1 2 3 4 5; do
  TESTIMONY_DATA=../data/dynsim/${r}_${s}.json $PY cef_baseline.py ../results/rtm/cef_${r}_${s}.json > ../results/rtm/cef_${r}_${s}.log 2>&1 &
done; done
wait
