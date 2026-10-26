#!/bin/bash
# Run the Phase 2 grid one job at a time from runs/phase2/queue.txt ("ARM SEED FOLD" per line),
# which `make grid` copies from configs/phase2_queue.txt.
# The queue file is re-read before every job, so pending lines can be edited or removed while it
# runs. A job whose manifest.json exists is complete and skipped; a failed job is recorded and
# not retried.
cd "$(dirname "$0")/.."
Q=runs/phase2/queue.txt
LOG=runs/phase2/queue_log.tsv
[ -f "$LOG" ] || printf "start_utc\tend_utc\tarm\tseed\tfold\texit\tgit_sha\n" > "$LOG"
while true; do
  next=""
  while read -r arm seed fold; do
    [ -z "$arm" ] && continue
    [ -f "runs/phase2/$arm/seed$seed/$fold/manifest.json" ] && continue
    awk -F'\t' -v a="$arm" -v s="$seed" -v f="$fold" '$3==a && $4==s && $5==f && $6!=0 {x=1} END {exit !x}' "$LOG" && continue
    next="$arm $seed $fold"; break
  done < "$Q"
  [ -z "$next" ] && break
  set -- $next
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  nice -n 15 .venv/bin/python -m sk.train --arm "$1" --seed "$2" --fold "$3" \
    > "runs/phase2/logs/$1_seed$2_$3.log" 2>&1
  code=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$start" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" \
    "$code" "$(git rev-parse --short HEAD)" >> "$LOG"
done
echo "queue empty"
