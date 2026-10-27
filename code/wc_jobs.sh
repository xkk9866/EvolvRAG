#!/bin/bash
# WikiCoach: RTM, ablations, TSSM, baselines, partial views, then LLM managers on GPU 7.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
O=../results/wc; mkdir -p $O
run() { name=$1; shift; [ -f $O/$name.json ] || OMP_NUM_THREADS=2 RTM_DEVICE=cpu "$@" > $O/$name.log 2>&1; }
run rtm        $PY run_rtm.py '{"_post":true}' $O/rtm.json &
run norefresh  $PY run_rtm.py '{"refresh":false}' $O/norefresh.json &
run online     $PY run_rtm.py '{"hindsight":false}' $O/online.json &
run rank0      $PY run_rtm.py '{"rank":0}' $O/rank0.json &
run nocopy     $PY run_rtm.py '{"copy":false}' $O/nocopy.json &
run static     $PY run_rtm.py '{"hazard":0.001}' $O/static.json &
run noctxf     $PY run_rtm.py '{"refresh_ctx":false}' $O/noctxf.json &
run tssm       $PY run_wikicoach.py tssm '{"_threads":2,"_post":true}' $O/tssm.json &
run baselines  $PY run_wikicoach.py baselines $O/baselines.json &
run cef        $PY cef_baseline.py $O/cef.json &
for k in 2 3; do for s in 0 1 2; do
  run rtm_k${k}_$s  $PY run_rtm.py '{}' $O/rtm_k${k}_$s.json "k=$k,seed=$s" &
  run tssm_k${k}_$s $PY run_wikicoach.py tssm '{"_threads":2}' $O/tssm_k${k}_$s.json "k=$k,seed=$s" &
  run base_k${k}_$s $PY run_wikicoach.py baselines $O/base_k${k}_$s.json "k=$k,seed=$s" &
done; done
wait
CUDA_VISIBLE_DEVICES=7 GPU_UTIL=0.5 HF_HUB_OFFLINE=1 $PY run_llm_memory.py $O/rtm.json ../models/Qwen2.5-7B-Instruct-AWQ $O/llm_qwen7b.json > $O/llm_qwen7b.log 2>&1
CUDA_VISIBLE_DEVICES=7 GPU_UTIL=0.5 HF_HUB_OFFLINE=1 $PY run_llm_memory.py $O/rtm.json ../models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 $O/llm_llama8b.json > $O/llm_llama8b.log 2>&1
echo done > $O/ALL_DONE
