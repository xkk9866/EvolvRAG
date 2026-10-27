#!/bin/bash
# DynSim grid: worlds, baselines, TSSM and ablations. One job per line, 20 in parallel.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
mkdir -p ../data/dynsim ../results/dynsim
for r in base domain cliques laggy; do for s in 0 1 2 3 4 5; do
  [ -f ../data/dynsim/${r}_${s}.json ] || $PY dynsim.py $r $s ../data/dynsim/${r}_${s}.json
done; done
jobs() {
for r in base domain cliques laggy; do for s in 0 1 2 3 4 5; do
  echo "base $r $s"; echo "tssm $r $s"
  if [ $s -lt 4 ]; then for v in online rank0 nocopy noself static; do echo "$v $r $s"; done; fi
done; done
}
one() {
  v=$1; r=$2; s=$3
  W=../data/dynsim/${r}_${s}.json; out=../results/dynsim/${v}_${r}_${s}.json
  [ -f "$out" ] && return
  case $v in
    tssm) kw='{"_threads":2}';;
    online) kw='{"_threads":2,"hindsight":false}';;
    rank0) kw='{"_threads":2,"rank":0}';;
    nocopy) kw='{"_threads":2,"copy":false}';;
    noself) kw='{"_threads":2,"span":"time"}';;
    static) kw='{"_threads":2,"hazard":0.001}';;
  esac
  if [ "$v" = base ]; then TESTIMONY_DATA=$W $PY run_wikicoach.py baselines $out > $out.log 2>&1
  else TESTIMONY_DATA=$W $PY run_wikicoach.py tssm "$kw" $out > $out.log 2>&1; fi
}
export -f one; export PY
jobs | xargs -P 20 -L 1 bash -c 'one $0 $1 $2'
echo done > ../results/dynsim/ALL_DONE
