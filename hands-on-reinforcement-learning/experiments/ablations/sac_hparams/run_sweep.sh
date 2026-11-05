#!/usr/bin/env bash
# HalfCheetah SAC 超参单因素扫描：100k steps × 3 seeds，每次只改一个超参。
# 基线（lr 3e-4 / batch 256 / gamma .99 / init_alpha .01）不重跑，
# 直接取 seed_experiments/halfcheetah_sac 的 1M 结果在 20k 网格上的前 100k 步。
set -u
cd "$(dirname "$0")"
PY="../../../.venv/Scripts/python.exe"
RUNNER="../../seed_experiments/halfcheetah_sac/run_one.py"
SEEDS="${SEEDS:-0 1 2}"
JOBS="${JOBS:-6}"
STEPS="${STEPS:-100000}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
mkdir -p results/raw results/logs

# 名称|附加参数
CONFIGS=(
  "lr1e-4|--actor-lr 1e-4 --critic-lr 1e-4"
  "lr1e-3|--actor-lr 1e-3 --critic-lr 1e-3"
  "batch64|--batch-size 64"
  "batch1024|--batch-size 1024"
  "gamma0.95|--gamma 0.95"
  "alpha1.0|--init-alpha 1.0"
)

run_one() {
    local name="$1" seed="$2" extra="$3"
    local tag="${name}_s${seed}"
    if [ -f "results/raw/${tag}.json" ]; then echo "[skip] ${tag}"; return; fi
    # shellcheck disable=SC2086
    "$PY" "$RUNNER" --seed "$seed" --steps "$STEPS" --eval-interval 20000 \
        --final-eval-episodes 10 $extra \
        --out "results/raw/${tag}.json" > "results/logs/${tag}.log" 2>&1
}

for seed in $SEEDS; do
    for cfg in "${CONFIGS[@]}"; do
        name="${cfg%%|*}"; extra="${cfg#*|}"
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
        run_one "$name" "$seed" "$extra" &
    done
done
wait
echo "SAC HPARAM SWEEP DONE"
