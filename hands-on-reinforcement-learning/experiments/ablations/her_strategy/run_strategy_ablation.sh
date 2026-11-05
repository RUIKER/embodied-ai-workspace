#!/usr/bin/env bash
# HER 目标采样策略消融：future 复用主实验，补跑 final / episode。
set -u
cd "$(dirname "$0")"
PY="../../../.venv/Scripts/python.exe"
RUNNER="../../seed_experiments/her_main/run_one.py"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
JOBS="${JOBS:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
mkdir -p results/raw results/logs

run_one() {
    local strategy="$1" seed="$2"
    local tag="${strategy}_s${seed}"
    if [ -f "results/raw/${tag}.json" ]; then
        echo "[skip] ${tag}"
        return
    fi
    "$PY" "$RUNNER" --seed "$seed" --her 1 --ratio 0.8 \
        --strategy "$strategy" --eval-interval 50 \
        --out "results/raw/${tag}.json" > "results/logs/${tag}.log" 2>&1
}

for strategy in final episode; do
    for seed in $SEEDS; do
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
        run_one "$strategy" "$seed" &
    done
done
wait
echo "STRATEGY ABLATION DONE"
