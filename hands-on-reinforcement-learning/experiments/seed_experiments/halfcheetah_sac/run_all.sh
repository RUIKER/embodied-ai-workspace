#!/usr/bin/env bash
# HalfCheetah SAC：默认 3 seeds，并行 3 个进程；已有结果会跳过。
set -u
cd "$(dirname "$0")"
PY="../../../.venv/Scripts/python.exe"
SEEDS="${SEEDS:-0 1 2}"
JOBS="${JOBS:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
mkdir -p results/raw results/logs

run_one() {
    local seed="$1"
    local tag="sac_s${seed}"
    if [ -f "results/raw/${tag}.json" ]; then
        echo "[skip] ${tag}"
        return
    fi
    "$PY" run_one.py --seed "$seed" --out "results/raw/${tag}.json" \
        > "results/logs/${tag}.log" 2>&1
}

for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
    run_one "$seed" &
done
wait
echo "HALFCHEETAH SAC DONE"
