# Results and raw data

## Baseline

[Baseline report](baseline.md) has the full 1–5 user decode table, the cold-prefill measurements, retrieval counts, and methods. The published card reports **42.7 tok/s** for one user, **107.8 tok/s** aggregate for five users, and **3,149** and **2,464 tok/s** cold prefill near 32K and 128K input tokens.

- [Decode request and wave samples](../results/baseline/inkling.decode.json)
- [Cold-prefill samples and retrieval answers](../results/baseline/inkling.prefill.json)
- [Sampled GPU-process memory trace](../results/baseline/inkling.memory.csv)
- [Machine-readable baseline summary](../results/baseline/summary.json)

## Optimization screens

[Controlled tuning](optimization.md) records greedy drafting, classic CUDA graphs, DSpark off, and FP8 KV. The decode screens use a 256-output-token budget and report medians. The baseline uses 400 output tokens and reports best-wave decode throughput. Use the matched controls in each screen to calculate changes.

- [Drafting and CUDA graph samples, logs, and launch arguments](../results/tuning/)
- [FP8 KV comparison, including 128K retrieval probes](../results/tuning/fp8-kv/)
- [Hardware microbenchmark results](../results/tuning/hardware.json)

## Hardware and build pins

- One NVIDIA DGX Spark with GB10 and 128 GB unified memory.
- [Pinned ARM64 image manifest reference](../results/baseline/inkling.image.txt)
- [Pinned upstream checkpoint revision](../results/baseline/checkpoint.json)
- [Exact server launch arguments](../results/baseline/launch.args.txt)
