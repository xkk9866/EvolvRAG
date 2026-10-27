#!/bin/bash
cd "$(dirname "$0")"
LOG=../logs/reader_queue.log
until [ "$(cat ../logs/reader_prepare_*.log | grep -c prepared)" = "64" ]; do sleep 60; done
cat ../results/reader_windows_stationary.jsonl ../results/reader_windows_topic.jsonl ../results/reader_windows_open_rumor.jsonl ../results/reader_windows_sybil.jsonl > ../results/reader_windows_all.jsonl
echo "windows ready $(wc -l < ../results/reader_windows_all.jsonl)" >> $LOG
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
  free=$((24564 - used))
  if [ $free -ge 10000 ]; then break; fi
  echo "$(date) gpu3 free ${free}MiB, waiting" >> $LOG
  sleep 120
done
echo "$(date) starting reader on gpu3 (free ${free}MiB)" >> $LOG
CUDA_VISIBLE_DEVICES=3 GPU_UTIL=0.36 HF_HUB_OFFLINE=1 ${PYTHON:-python} run_reader_tsm.py read ../results/reader_windows_all.jsonl ../models/Qwen2.5-7B-Instruct-AWQ ../results/reader_qwen7b.json >> $LOG 2>&1
echo "$(date) reader exit $?" >> $LOG
