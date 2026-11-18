#!/bin/bash
# Eval-only executable for the option-intervention sets (2026-10-20, "verify vs compute"): run_eval_iv.sh <env file> <name>
# Env file (in the bundle): MODEL=models/Qwen3.5-4B (the base, from $STAGING/Qwen3.5-4B.tar) and WTAR=<name of a weights
# tar in $STAGING/weights, without .tar> or WTAR='' for the untrained base; SETS="name=path ..." eval sets inside the
# bundle. Lays the job out like run_train5.sh (cotd/ with repo, data, models, logs), runs scripts/eval_student.py once
# on all sets, packs runs/<name>/ (metrics.json, preds_*.jsonl) + the log into results_<name>.tgz. Always writes it.
set -uo pipefail
ENVF=$1; NAME=$2
TOP=${_CONDOR_SCRATCH_DIR:-$PWD}; cd "$TOP"
export HOME=$TOP; W=$HOME/cotd
mkdir -p "$W/logs" "$W/models" "$W/envs/cotd/bin" "$W/tmp" "$W/runs"
export TMPDIR=$W/tmp
pack() {
  [ -n "${MONPID:-}" ] && kill "$MONPID" 2>/dev/null
  local items=("logs/$NAME.out"); [ -f "$W/logs/$NAME.mon" ] && items+=("logs/$NAME.mon"); [ -d "$W/runs" ] && items+=("runs")
  (cd "$W" && tar czf "$TOP/results_$NAME.tgz" --exclude='*/model' --exclude='*.safetensors' --exclude='*.pt' "${items[@]}") || tar czf "$TOP/results_$NAME.tgz" -T /dev/null
}
echo "== $NAME on $(hostname) $(date)" > "$W/logs/$NAME.out"
if ! NGPU_REQ=${NGPU_REQ:-1} python3 - >> "$W/logs/$NAME.out" 2>&1 <<'PY'
import os, sys, torch
need = int(os.environ.get("NGPU_REQ") or 1); n = torch.cuda.device_count()
print(f"preflight: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} sees {n} GPU(s), job asked {need}")
if n < max(1, need): sys.exit("preflight: too few GPUs")
for i in range(n):
    torch.cuda.set_device(i); x = torch.ones(256, 256, device=f"cuda:{i}"); s = float((x @ x).sum()); torch.cuda.synchronize(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"preflight: gpu {i} {torch.cuda.get_device_name(i)} ok (matmul {s:.0f}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free)")
    if abs(s - 256.0 ** 3) > 1 or free < 0.9 * total: sys.exit(f"preflight: gpu {i} wrong result or busy")
PY
then echo "== GPU preflight FAILED on $(hostname): exit 86 (retried on another machine) $(date)" >> "$W/logs/$NAME.out"; pack; exit 86; fi
( while sleep 120; do echo "== $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg) memavail $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)G"
    ps -eo pid,pcpu,rss,etime,args --sort=-pcpu 2>/dev/null | awk 'NR>1 && /python/{printf "  %s %s%% %dG %s %s\n",$1,$2,$3/1048576,$4,substr($0,index($0,$5),90)}' | head -3
  done >> "$W/logs/$NAME.mon" 2>&1 ) &
MONPID=$!
tar xzf "${CODE_TGZ:-cotd_code_data.tgz}" -C "$W" || { echo "bundle extract failed" >> "$W/logs/$NAME.out"; pack; exit 1; }
[ ! -f "$W/CODE_PIN" ] || echo "code pin: $(cat "$W/CODE_PIN")" >> "$W/logs/$NAME.out"
cd "$W"
set -a; . "$ENVF"; set +a
export SLURM_JOB_ID="chtc-$NAME" HF_HUB_OFFLINE=1 HF_HOME="$W/hf" TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
if [ -n "${WTAR:-}" ]; then
  MT=$STAGING/weights/$WTAR.tar; [ -f "$MT" ] || { echo "weights tar missing: $MT" >> "logs/$NAME.out"; pack; exit 1; }
  mkdir -p "models/$WTAR" && tar xf "$MT" -C "models/$WTAR" || { echo "weights extract failed ($MT)" >> "logs/$NAME.out"; pack; exit 1; }
  MDIR=models/$WTAR/model
else
  MB=$(basename "$MODEL"); MT=$STAGING/$MB.tar
  tar xf "$MT" -C models || { echo "model extract failed ($MT)" >> "logs/$NAME.out"; pack; exit 1; }
  MDIR=$MODEL
fi
[ -f "$MDIR/config.json" ] || { echo "no config.json in $MDIR" >> "logs/$NAME.out"; ls -la "$MDIR" >> "logs/$NAME.out" 2>&1; pack; exit 1; }
echo "== model $MDIR ($(du -sh "$MDIR" | cut -f1)); sets: $SETS" >> "logs/$NAME.out"
ln -sf "$(command -v python3)" "envs/cotd/bin/python"
PYTHONPATH=repo/src envs/cotd/bin/python repo/scripts/eval_student.py --model "$MDIR" --eval $SETS --max-len 1536 --bs 8 --out "runs/$NAME" >> "logs/$NAME.out" 2>&1
rc=$?
echo "== exit $rc $(date)" >> "logs/$NAME.out"
rm -rf "models"
pack
exit $rc
