#!/bin/bash
# RTM on DynSim: model and ablations. usage: run_rtmgrid.sh <device> <parallel> [reverse]
cd "$(dirname "$0")"
export PY=${PYTHON:-python} DEV=$1
mkdir -p ../results/rtm
jobs() {
for v in rtm norefresh online rank0 nocopy static; do
for r in base domain cliques laggy; do for s in 0 1 2 3 4 5; do
  if [ $v = rtm ] || [ $s -lt 4 ]; then echo "$v $r $s"; fi
done; done; done
}
one() {
  v=$1; r=$2; s=$3
  out=../results/rtm/${v}_${r}_${s}.json
  [ -f "$out" ] && return
  [ -f "$out.log" ] && return
  touch $out.log
  case $v in
    rtm) kw='{}';; norefresh) kw='{"refresh":false}';; nocopy) kw='{"copy":false}';;
    rank0) kw='{"rank":0}';; online) kw='{"hindsight":false}';; static) kw='{"hazard":0.001}';;
  esac
  OMP_NUM_THREADS=2 RTM_DEVICE=$DEV CUDA_VISIBLE_DEVICES=7 TESTIMONY_DATA=../data/dynsim/${r}_${s}.json $PY run_rtm.py "$kw" $out > $out.log 2>&1 < /dev/null
}
export -f one
if [ "$3" = reverse ]; then jobs | tac; else jobs; fi | xargs -P $2 -L 1 bash -c 'one $0 $1 $2'
