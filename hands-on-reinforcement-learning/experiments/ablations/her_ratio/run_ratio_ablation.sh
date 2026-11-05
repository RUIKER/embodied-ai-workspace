#!/usr/bin/env bash
# her_ratio 消融：补跑 ratio 0.2 / 0.5 / 1.0。
#
# ratio=0（无 HER）和 ratio=0.8 直接复用 seed_experiments/her_main 的结果，
# 不在这里重跑；所以这里只新跑三档。
#
# 用法：
#   bash run_ratio_ablation.sh                     # 默认 seeds 0-9
#   SEEDS="5 6 7 8 9" bash run_ratio_ablation.sh   # 只补指定 seed
#
# 已存在的结果文件会跳过，可以随时中断后重跑。
set -u
cd "$(dirname "$0")"
PY="../../../.venv/Scripts/python.exe"
RUNNER="../../seed_experiments/her_main/run_one.py"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
JOBS="${JOBS:-10}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
mkdir -p results/raw results/logs

run_one() {
    local ratio="$1" seed="$2"
    local tag="ratio${ratio}_s${seed}"
    if [ -f "results/raw/${tag}.json" ]; then
        echo "[skip] ${tag}"
        return
    fi
    "$PY" "$RUNNER" --seed "$seed" --her 1 --ratio "$ratio" \
        --strategy future --eval-interval 50 \
        --out "results/raw/${tag}.json" > "results/logs/${tag}.log" 2>&1
}

for ratio in 0.2 0.5 1.0; do
    for seed in $SEEDS; do
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
        run_one "$ratio" "$seed" &
    done
done
wait
echo "RATIO ABLATION DONE"
