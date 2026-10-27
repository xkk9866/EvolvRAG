#!/bin/bash
cd "$(dirname "$0")"
mkdir -p ../results/sim
for n in 10 25 50; do for seed in 0 1 2 3 4; do for tag in full norestart sincere_norestart sincere; do
  echo "$n $seed $tag ../results/sim/rec_${n}_${seed}_${tag}.json"
done; done; done | xargs -P 12 -L 1 ${PYTHON:-python} sim_recovery.py > ../logs/sim_recovery.log 2>&1
echo done >> ../logs/sim_recovery.log
