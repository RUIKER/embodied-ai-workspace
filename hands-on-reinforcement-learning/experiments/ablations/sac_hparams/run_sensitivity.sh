#!/usr/bin/env bash
# SAC 超参灵敏度网格（grid.py）：100k steps × 3 seeds，已有结果自动跳过。
#
# 调度顺序：tier 1 → 2 → 3；同一 tier 内 seed 在外层，所以随时中断，
# 已完成的都是整齐的“前 k 个 seed”。
# 用法：
#   bash run_sensitivity.sh                 # 全部
#   TIERS="1" bash run_sensitivity.sh       # 只跑核心 tier
#   JOBS=6 SEEDS="0 1 2" bash run_sensitivity.sh
#   DRY=1 bash run_sensitivity.sh           # 只列出将要跑的 run，不启动
# 预计：tier1 ≈ 4–5h，tier2 ≈ 2–3h，tier3（UTD=4）≈ 1–2h（6 并行，估算，见 README）。
set -u
cd "$(dirname "$0")"
PY="../../../.venv/Scripts/python.exe"
RUNNER="../../seed_experiments/halfcheetah_sac/run_one.py"
SEEDS="${SEEDS:-0 1 2}"
JOBS="${JOBS:-6}"
TIERS="${TIERS:-1 2 3}"
STEPS="${STEPS:-100000}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
mkdir -p results/raw results/logs

# 防睡眠：只在本脚本存活期间生效（SetThreadExecutionState），不改系统电源设置。
WINPID="$(cat /proc/$$/winpid 2>/dev/null || echo 0)"
if [ "$WINPID" != "0" ] && [ -z "${DRY:-}" ]; then
    powershell.exe -NoProfile -Command "
        Add-Type -Namespace W -Name P -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);';
        [void][W.P]::SetThreadExecutionState(0x80000001);
        while (Get-Process -Id $WINPID -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }" \
        > /dev/null 2>&1 &
fi

mapfile -t LINES < <("$PY" grid.py)

run_one() {
    local name="$1" seed="$2" extra="$3"
    local tag="${name}_s${seed}"
    if [ -f "results/raw/${tag}.json" ]; then echo "[skip] ${tag}"; return; fi
    if [ -n "${DRY:-}" ]; then echo "[would run] ${tag}: ${extra}"; return; fi
    # shellcheck disable=SC2086
    "$PY" "$RUNNER" --seed "$seed" --steps "$STEPS" --eval-interval 20000 \
        --final-eval-episodes 10 $extra \
        --out "results/raw/${tag}.json" > "results/logs/${tag}.log" 2>&1
    echo "[done] ${tag} $(date +%H:%M)"
}

for tier in $TIERS; do
    for seed in $SEEDS; do
        for line in "${LINES[@]}"; do
            t="${line%%|*}"; rest="${line#*|}"
            [ "$t" = "$tier" ] || continue
            name="${rest%%|*}"; extra="${rest#*|}"
            while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
            run_one "$name" "$seed" "$extra" &
        done
    done
done
wait
echo "SAC SENSITIVITY DONE $(date +%H:%M)"
