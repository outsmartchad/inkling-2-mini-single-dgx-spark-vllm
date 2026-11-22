# Troubleshooting

Run these checks on the Spark from the recipe checkout. `./bin/recipe.sh doctor` checks the ARM64 Linux host, tools, Docker access, and cache filesystem. `./bin/recipe.sh logs` shows the last 100 lines while the container exists. If the container exits on startup, use the output from the foreground `serve` terminal.

## Docker permission denied or GPU unavailable

```bash
docker context show
docker info
nvidia-smi
```

Use the Spark’s local Docker context. If the daemon is running but your user cannot access it, follow NVIDIA’s Docker permissions instructions for your account. If Docker reports that it cannot select a GPU device driver, check NVIDIA Container Toolkit and the host driver using NVIDIA’s runtime guide. [NVIDIA Container Runtime for Docker](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)

After fetching, the following can test GPU access with the same image, independent of vLLM.

```bash
docker run --rm --gpus all --entrypoint nvidia-smi \
  vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e -L
```

## Download fails or checkpoint is incomplete

Check free space on both the cache filesystem and Docker’s image filesystem. Confirm the Spark can reach Docker Hub and Hugging Face and retry `./bin/recipe.sh fetch`. With a custom `INKLING_CACHE`, set the same absolute path before fetching and serving. The pinned checkpoint is public. An unexplained 401 or 403 can warrant checking the network or its HTTP proxy. [Pinned public checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7)

## API connection refused or the port is busy

First use the serving terminal to check whether weights are loading or kernels are compiling. After startup, check the listener from another Spark terminal.

```bash
ss -ltn 'sport = :18905'
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:18905/v1/models
```

The model list should include `thinkingmachines/Inkling-2-Mini`. If the port or container name is occupied, identify the owner before stopping anything. `./bin/recipe.sh stop` operates on a container started by this version of the launcher. For an older run, use Ctrl-C in its serving terminal. For access from another machine, use the [loopback SSH forwarding example](reproduce.md#connect-from-your-workstation).

## CUDA out of memory or benchmark slower than the card

Check `nvidia-smi`, `free -h`, and the serving logs. On the Spark, CPU and GPU workloads share unified memory. Pause your other inference services and retry with the published launch settings. Let the startup compile complete. Check that your prompt length, reasoning effort, sampling, and concurrency match the [baseline](baseline.md). A cold startup or first-use compilation can affect TTFT. Aggregate tokens/s across five users and tokens/s for one user measure different things.

## Chat or retrieval runs out of output tokens

Check the response’s `finish_reason` and `usage`. Reasoning can exhaust the completion budget before an answer appears, including on a request that asks for reasoning effort `none`. Increase `max_tokens` for your application when needed. For benchmark comparisons, keep the published token budgets and record any missed retrieval result. [Baseline retrieval results](baseline.md#cold-prefill)
