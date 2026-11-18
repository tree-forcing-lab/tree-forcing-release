#!/bin/bash
# Writes the batch-28 DAGs (README_28.md; Qwen3.5-4B K&K and GSM8K, code 5f925cb (branch cot) + the friend's
# 1-GPU patch): jobs28_4b.dag = the review ablations (2501-2533, priorities in joblist_28_4b.txt);
# smoke28_4b.dag = SMOKE_ONLY of one job per distinct arm / data (2501 2504 2510 2516 2522 2525 2528 2531).
# Lines of joblist_28_4b.txt: "<env file>, <priority>, <GPUs>" (1 = the friend's patch: 8 CPUs / 80 GB).
#   bash make_dags_28.sh
set -euo pipefail
cd "$(dirname "$0")"
list() { local f=$1; [ -f "$f" ] || f=../condor/$1; [ -f "$f" ] && echo "$f"; }
node() {   # node <dag> <env> <name> <prio> <ngpu> <smoke>
  local n=j${3%%_*}
  [ -n "$6" ] && { n=${3#smoke_}; n=smoke${n%%_*}; }   # smoke_1201_... -> smoke1201
  printf 'JOB %s train28_4b.sub\nVARS %s envfile="%s" name="%s" prio="%s" smoke="%s" ngpu="%s" %s\nSCRIPT POST %s post_push.sh %s $RETURN\n' \
    "$n" "$n" "$2" "$3" "$4" "$6" "$5" "$(res "$5")" "$n" "$3" >> "$1"
  [ -n "$6" ] || printf 'RETRY %s 1\nPRIORITY %s %s\n' "$n" "$n" "$4" >> "$1"
  echo >> "$1"
}
res() { case "$1" in 1) echo 'cpus="8" mem="80GB" gpumem="40000"' ;; 2) echo 'cpus="12" mem="120GB" gpumem="40000"' ;; *) echo "make_dags_16: $1 GPUs?" >&2; exit 1 ;; esac; }
: > smoke28_4b.dag
for pair in "joblist_28_4b.txt jobs28_4b.dag"; do
  set -- $pair; L=$(list "$1") || { echo "no $1"; exit 1; }; D=$2; : > "$D"
  while IFS=, read -r env prio ngpu; do
    env=$(echo "$env" | xargs); prio=$(echo "$prio" | xargs); ngpu=$(echo "$ngpu" | xargs); name=$(basename "$env" .env)
    [ -n "$env" ] || continue
    node "$D" "$env" "$name" "$prio" "$ngpu" ""
    case " 2501 2504 2510 2516 2522 2525 2528 2531 " in *" ${name%%_*} "*) node smoke28_4b.dag "$env" "smoke_$name" 0 "$ngpu" 1 ;; esac
  done < "$L"
  echo "$D: $(grep -c '^JOB' "$D") jobs"
done
echo "smoke28_4b.dag: $(grep -c '^JOB' smoke28_4b.dag) jobs"
