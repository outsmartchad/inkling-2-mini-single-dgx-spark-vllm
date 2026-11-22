# Reproduce the single-Spark measurements

Use one DGX Spark with DGX OS, the NVIDIA container runtime, and access to its local Docker daemon. Run `./bin/recipe.sh doctor` before downloading. The check reports free space on the model-cache filesystem. Docker stores image layers on its own filesystem, which can have a different amount of free space. [NVIDIA container setup](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)

```bash
df -h "$(docker info --format '{{.DockerRootDir}}')"
```

Checking Docker’s storage path may require admin access. Review the available memory shown by `doctor` before loading the model.

## Stage the pinned artifacts

The [quick start](../README.md#quick-start) installs the measured ARM64 image and downloads the pinned public checkpoint. The checkpoint alone takes about 18.8 GiB. The first fetch also needs disk space for the Docker image and for the Hub and image caches. Run `fetch` again if a download is interrupted. Completed Hub files can be reused.

By default, the launcher stores weights and engine caches in the checkout’s `.cache/`. To share them between checkouts or use another disk, export the same absolute directory in each Spark terminal before the recipe commands.

```bash
export INKLING_CACHE="$HOME/.cache/inkling-2-mini-spark"
./bin/recipe.sh doctor &&
./bin/recipe.sh fetch &&
./bin/recipe.sh serve
```

`fetch` verifies the file sizes against Hub metadata and records the checkpoint revision. At serve time the launcher checks the staged checkpoint’s required files and weight shards. Rerun `fetch` if an earlier staging attempt left an incomplete cache. vLLM receives the pinned snapshot path with offline Hub mode and a read-only weights mount. [Launch command and build pins](results.md#hardware-and-build-pins)

## Clean-install check

On November 22, 2026, a fresh clone with an empty recipe weight and engine cache passed `doctor`, `fetch`, `serve`, and `smoke` on one DGX Spark. The first startup took roughly four and a half minutes on that run, including weight loading, autotuning, and CUDA graph capture. The API bound to `127.0.0.1:18905` and served the smoke request and the README's hash-map example. The recipe's `stop` command then shut down the test container.

## Run the full benchmark

Pause other inference services you control to give vLLM the same memory headroom and reduce contention. With the server ready, run the following from a second Spark terminal inside the checkout.

```bash
./bin/recipe.sh smoke &&
./bin/recipe.sh benchmark
```

The run takes several minutes. It warms the model and runs three decode waves at each concurrency from 1 to 5. It then runs one warm-up and three measured samples per prefill target near 32K and 128K tokens. Watch the terminal for errors and progress. The launcher prints a new `runs/baseline-*` directory with `decode.json` and `prefill.json`. The published GPU-process memory measurements were collected separately. [Method and samples](baseline.md)

Use the benchmark client on the Spark for throughput and TTFT comparison. Each TTFT starts before the HTTP request and ends at the first streamed reasoning or answer chunk. Prefill throughput is actual prompt tokens divided by client TTFT. The [results index](results.md) links the original settings, samples, and memory trace.

## Connect from your workstation

For interactive use, replace `your-spark` with a working SSH host alias. Run this on your workstation and keep the terminal open.

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18905:127.0.0.1:18905 your-spark
```

Then point the client on your workstation to `http://127.0.0.1:18905/v1` with model ID `thinkingmachines/Inkling-2-Mini`. Both ends of the forwarding path use loopback. If your workstation port is already in use, choose a different port for the first `18905` and use that port in the client URL.

## Finish the run

Press Ctrl-C in the server terminal or run `./bin/recipe.sh stop`. If you paused another inference service before the benchmark, start it again using its usual launcher. The pinned image and cache remain available for the next run.
