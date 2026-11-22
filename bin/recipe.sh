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
container_label=io.github.outsmartchad.inkling-recipe=1
cache_snapshot="$cache/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/$revision"

usage() {
    printf '%s\n' "Usage: $0 doctor | fetch | serve | smoke | benchmark | logs | stop"
}

require_command() {
    if ! command -v "$1" >/dev/null; then
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

check_docker() {
    require_command docker
    if ! docker info >/dev/null; then
        printf '%s\n' 'Docker is unavailable. Check the daemon and your account permissions. See docs/troubleshooting.md.' >&2
        exit 1
    fi
}

[[ $# == 1 ]] || { usage >&2; exit 2; }
case "$1" in
    doctor)
        if [[ "$(uname -sm)" != "Linux aarch64" ]]; then
            printf '%s\n' 'Run this recipe on the DGX Spark (Linux aarch64).' >&2
            exit 1
        fi
        require_command python3
        require_command curl
        require_command nvidia-smi
        check_docker
        nvidia-smi -L
        mkdir -p "$cache"
        df -h "$cache"
        if command -v free >/dev/null; then
            free -h
        fi
        printf '%s\n' 'Host checks passed. Review the disk and memory report before proceeding.'
        ;;
    fetch)
        check_docker
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
        require_command python3
        check_docker
        if ! docker image inspect "$image" >/dev/null 2>&1; then
            printf '%s\n' 'The pinned image is missing. Run ./bin/recipe.sh fetch first.' >&2
            exit 1
        fi
        python3 "$root/bin/fetch.py" --check "$cache_snapshot"
        umask 077
        install -d -m 0700 "$cache/runtime"
        install -d -m 0777 "$cache/runtime/engine-cache"
        exec docker run --rm --name "$container_name" --label "$container_label" \
            --gpus all --network host --shm-size=4g \
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
    smoke|benchmark)
        require_command curl
        require_command python3
        export SPARK_BENCHMARK_URL="http://127.0.0.1:$port" SPARK_BENCHMARK_MODEL="$model"
        if [[ "$1" == smoke ]]; then
            exec python3 "$root/bin/benchmark.py" --smoke
        fi
        mkdir -p "$root/runs"
        run="$(mktemp -d "$root/runs/baseline-XXXXXX")"
        printf 'Results: %s\n' "$run"
        python3 "$root/bin/benchmark.py" >"$run/decode.json"
        python3 "$root/bin/prefill.py" "$run/prefill.json"
        ;;
    logs|stop)
        require_command docker
        target="$(docker ps -aq --no-trunc --filter "name=^/$container_name$" --filter "label=$container_label")"
        if [[ -z "$target" ]]; then
            printf '%s\n' 'No container started by this recipe is present. See the foreground serve terminal.' >&2
            exit 1
        fi
        if [[ "$1" == logs ]]; then
            exec docker logs --tail 100 "$target"
        fi
        exec docker stop "$target"
        ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
