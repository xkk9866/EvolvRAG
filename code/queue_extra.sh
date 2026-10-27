#!/bin/bash
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export NPROC=16
$PY run_main.py tsm stationary,topic,open_rumor,sybil 700-731 variants_mem.json ../results/test_tsm_mem.json > ../logs/test_tsm_mem.log 2>&1
$PY run_main.py tsm drift,collude 700-731 variants_one.json ../results/ood_tsm.json > ../logs/ood_tsm.log 2>&1
$PY run_main.py baselines drift,collude 700-731 ../results/ood_baselines.json > ../logs/ood_baselines.log 2>&1
$PY run_main.py tsm stationary@0.08,topic@0.08,open_rumor@0.08,sybil@0.08,stationary@0.28,topic@0.28,open_rumor@0.28,sybil@0.28 700-715 variants_one.json ../results/rate_tsm.json > ../logs/rate_tsm.log 2>&1
$PY run_main.py baselines stationary@0.08,topic@0.08,open_rumor@0.08,sybil@0.08,stationary@0.28,topic@0.28,open_rumor@0.28,sybil@0.28 700-715 ../results/rate_baselines.json > ../logs/rate_baselines.log 2>&1
echo queue done > ../logs/queue_extra.done
