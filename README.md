# Inkling 2 Mini on one DGX Spark with vLLM

Run Thinking Machines Lab’s 26.4B open-weight model on one DGX Spark. This recipe uses the official mixed NVFP4/MXFP8 checkpoint, native DSpark ×7, and an OpenAI-compatible API bound to `127.0.0.1:18905`.

[Reproduce the benchmark](docs/reproduce.md) · [How it works](docs/how-it-works.md) · [Full results and raw data](docs/results.md) · [Troubleshooting](docs/troubleshooting.md)

## Quick start

Run these commands **on the Spark**, either in a terminal or over SSH. You need Git, Python 3, curl, and working Docker access with NVIDIA GPU support. The first run downloads an ARM64 image and an 18.8 GiB checkpoint. Check free space for both, plus runtime caches. The downloads are public and need no API key. [NVIDIA’s Spark Docker guide](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)

### 1. Download and start

```bash
git clone https://github.com/outsmartchad/inkling-2-mini-single-dgx-spark-vllm.git &&
cd inkling-2-mini-single-dgx-spark-vllm &&
./bin/recipe.sh doctor &&
./bin/recipe.sh fetch &&
./bin/recipe.sh serve
```

Leave the serving terminal open. On first launch, vLLM loads weights and compiles kernels before the API is ready. Wait until the logs show that the server has started.

### 2. Check the API and send a prompt

In a second terminal **on the Spark**, enter the same checkout.

```bash
./bin/recipe.sh smoke
```

A successful check prints `Chat smoke passed`. It checks the model ID and runs a short generation. To send your own prompt:

```bash
curl --noproxy '*' -fsS --max-time 120 http://127.0.0.1:18905/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingmachines/Inkling-2-Mini",
    "messages": [{"role": "user", "content": "Explain how a hash map works in one sentence."}],
    "max_tokens": 256,
    "chat_template_kwargs": {"reasoning_effort": "none"}
  }'
```

Use `http://127.0.0.1:18905/v1` as the base URL in an OpenAI-compatible client. See [reproduction](docs/reproduce.md) for an SSH forwarding example.

### 3. Stop

Press **Ctrl-C** in the serving terminal, or run `./bin/recipe.sh stop` from the second terminal. `./bin/recipe.sh logs` shows recent output while the container exists. Docker removes the container when it exits, and the downloaded weights and runtime cache stay on disk.

## Performance on one DGX Spark

| Workload | Throughput | TTFT |
|---|---:|---:|
| Prose decode, 1 user | 42.7 tok/s | 219 ms |
| Prose decode, 5 users | 107.8 tok/s aggregate | 1.73 s |
| 32K cold prefill | 3,149 tok/s | 10.40 s |
| 128K cold prefill | 2,464 tok/s | 53.19 s |

Measured November 21, 2026. Decode used high reasoning effort with 400 output tokens per request and reports the fastest aggregate wave at each concurrency. Prefill reports medians from three fresh prompts at each length. Sampled peak GPU-process memory was **81.7 GiB**. One 32K request exhausted its output budget before returning the retrieval answer. [Method and raw results](docs/results.md)

![Inkling 2 Mini benchmark card showing decode, cold prefill, and memory measurements.](assets/inkling-2-mini-dgx-spark.png)

## Configuration

- Official `thinkingmachines/Inkling-2-Mini` checkpoint at `ed40e7da`.
- ARM64 vLLM `0.1.dev25855+g7cb5dae22.cu130`, pinned by image digest.
- Native DSpark ×7 with probabilistic drafting and automatic KV dtype.
- Up to five concurrent requests, up to 262,144 tokens per request, and about 599K shared KV tokens reported in the baseline.
- Prefix caching disabled in the measured configuration. [Exact launch arguments](results/baseline/launch.args.txt)

MIT for the recipe scripts and docs. The weights are distributed under Apache 2.0. [Upstream model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
