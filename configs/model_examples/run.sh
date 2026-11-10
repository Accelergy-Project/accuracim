#!/bin/bash
# Run the model sweeps, one model at a time per GPU (from the repo root).
#   configs/model_examples/run.sh                   # every ADC noise sweep
#   configs/model_examples/run.sh ResNet34 ViT-B16  # these only
#   configs/model_examples/run.sh --bits            # bit searches (bits/)
#   GPUS="0 1" configs/model_examples/run.sh        # several GPUs (default: 0)
set -u
cd "$(dirname "$0")/../.."
dir=configs/model_examples
if [ "${1:-}" = "--bits" ]; then
    dir=$dir/bits; shift
fi
models=("$@")
[ ${#models[@]} -eq 0 ] && models=($(ls $dir/*.yaml | xargs -n1 basename | sed 's/\.yaml$//'))
read -ra gpus <<< "${GPUS:-0}"

# Split CPU threads between parallel runs
threads=$(( $(nproc) / ${#gpus[@]} )); [ "$threads" -lt 1 ] && threads=1
mkdir -p logs/model_examples

for g in "${!gpus[@]}"; do
    (
        for i in "${!models[@]}"; do
            [ $(( i % ${#gpus[@]} )) -eq "$g" ] || continue
            m=${models[$i]}; log=logs/model_examples/$(basename $dir)_$m.txt
            echo "[GPU ${gpus[$g]}] $m -> $log"
            CUDA_VISIBLE_DEVICES=${gpus[$g]} OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
                python -m accuracim --config $dir/$m.yaml > "$log" 2>&1 \
                || echo "[GPU ${gpus[$g]}] $m FAILED (see $log)"
        done
    ) &
done
wait
echo "done"
