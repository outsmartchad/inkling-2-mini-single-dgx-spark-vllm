#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cache="${INKLING_CACHE:-$root/.cache}"
if [[ "$cache" != /* ]]; then
    printf '%s\n' 'INKLING_CACHE must be an absolute path.' >&2
    exit 2
fi
model=thinkingmachines/Inkling-2-Mini
revision=ed40e7da625f594fc9c4990f5f66be24873c22d7
image=vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
port=18905
container_name=inkling-2-mini-local
cache_snapshot="$cache/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$revision"

[[ $# == 1 ]] || { printf '%s\n' "Usage: $0 fetch | serve | benchmark" >&2; exit 2; }
case "$1" in
    fetch)
        mkdir -p "$cache/hf"
        docker pull --platform linux/arm64 "$image"
        docker run --rm --cap-drop ALL --security-opt no-new-privileges \
            --user "$(id -u):$(id -g)" --env HOME=/tmp --env HF_HOME=/hf \
            --env HF_HUB_DISABLE_IMPLICIT_TOKEN=1 --env HF_HUB_DOWNLOAD_TIMEOUT=90 \
            --env HF_XET_NUM_CONCURRENT_RANGE_GETS=8 \
            --volume "$cache/hf:/hf" --volume "$root/bin/fetch.py:/recipe/fetch.py:ro" \
            --entrypoint /usr/bin/python3 "$image" /recipe/fetch.py
        ;;
    serve)
        if [[ ! -s "$cache_snapshot/config.json" || ! -s "$cache_snapshot/mtp.safetensors" ]]; then
            printf '%s\n' 'Run the fetch command to stage the pinned checkpoint.' >&2
            exit 1
        fi
        umask 077
        install -d -m 0700 "$cache/runtime"
        install -d -m 0777 "$cache/runtime/engine-cache"
        exec docker run --rm --name "$container_name" --gpus all --network host --shm-size=4g \
            --cap-drop ALL --security-opt no-new-privileges \
            --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
            --env HF_HOME=/hf --env HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
            --env VLLM_CACHE_ROOT=/engine-cache/vllm --env TRITON_CACHE_DIR=/engine-cache/triton \
            --volume "$cache/hf:/hf:ro" --volume "$cache/runtime/engine-cache:/engine-cache" \
            "$image" "/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$revision" \
            --served-model-name "$model" --host 127.0.0.1 --port "$port" \
            --max-model-len 262144 --max-num-seqs 5 --gpu-memory-utilization 0.70 \
            --max-num-batched-tokens 8192 --safetensors-load-strategy prefetch \
            --disable-prefix-caching --enable-prompt-tokens-details \
            --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
            --limit-mm-per-prompt '{"image":1,"video":0}' \
            --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
        ;;
    benchmark)
        export SPARK_BENCHMARK_URL="http://127.0.0.1:$port" SPARK_BENCHMARK_MODEL="$model"
        mkdir -p "$root/runs"
        run="$(mktemp -d "$root/runs/baseline-XXXXXX")"
        printf 'Results: %s\n' "$run"
        python3 "$root/bin/benchmark.py" >"$run/decode.json"
        python3 "$root/bin/prefill.py" "$run/prefill.json"
        ;;
    *) printf '%s\n' "Usage: $0 fetch | serve | benchmark" >&2; exit 2 ;;
esac
