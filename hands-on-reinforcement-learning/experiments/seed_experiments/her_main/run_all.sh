#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY="../../../.venv/Scripts/python.exe"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
mkdir -p results/raw results/logs

run_one() {
    local seed="$1" her="$2"
    local tag
    if [ "$her" = "1" ]; then
        tag="her_r0.8_future_s${seed}"
    else
        tag="noher_s${seed}"
    fi
    "$PY" run_one.py --seed "$seed" --her "$her" \
        --out "results/raw/${tag}.json" > "results/logs/${tag}.log" 2>&1
}
export -f run_one
export PY

for seed in 0 1 2 3 4 5 6 7 8 9; do
    for her in 1 0; do
        while [ "$(jobs -rp | wc -l)" -ge 10 ]; do sleep 1; done
        run_one "$seed" "$her" &
    done
done
wait
echo "ALL DONE"
