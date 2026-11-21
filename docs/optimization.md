# Inkling 2 Mini tuning on one DGX Spark

We used the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) to screen three recorded one-variable changes on the pinned vLLM image and official checkpoint. Controls bracket each candidate. The runner warmed one C1 and one C5 wave per launch, then alternated the concurrency order across six measured trials for each size. All 252 measured requests streamed reasoning and completed their 256-token output budget. The input was the same SparkDash hash-map prose at high reasoning effort, T=0.7, top-p=1 and top-k=20.

## Controlled medians

| Profile | C1 tok/s | C1 TTFT | C5 aggregate tok/s | C5 TTFT | GPU peak |
|---|---:|---:|---:|---:|---:|
| [`control-1`](../results/tuning/control-1.decode.json) · Baseline | 40.06 | 221 ms | 108.52 | 358 ms | 78.5 GiB |
| [`greedy-drafter`](../results/tuning/greedy-drafter.decode.json) · Greedy DSpark drafter | 38.78 | 176 ms | 102.52 | 357 ms | 78.2 GiB |
| [`control-2`](../results/tuning/control-2.decode.json) · Baseline | 40.11 | 179 ms | 110.03 | 360 ms | 78.5 GiB |
| [`classic-graphs`](../results/tuning/classic-graphs.decode.json) · Classic CUDA graphs | 40.08 | 200 ms | 111.30 | 357 ms | 78.3 GiB |
| [`control-3`](../results/tuning/control-3.decode.json) · Baseline | 40.09 | 221 ms | 110.78 | 317 ms | 78.1 GiB |
| [`no-draft`](../results/tuning/no-draft.decode.json) · DSpark off | 37.84 | 169 ms | 108.78 | 326 ms | 79.4 GiB |
| [`control-4`](../results/tuning/control-4.decode.json) · Baseline | 40.05 | 198 ms | 110.99 | 358 ms | 78.6 GiB |

C1 and C5 decode values are medians of six aggregate rates at each concurrency. TTFT is the median of per-wave median times to the first reasoning or answer chunk. The measured process peak includes startup and the short decode screen. These medians use a 256-token budget. The [original 400-token benchmark card](../assets/inkling-2-mini-dgx-spark.png) shows the original best-wave results.

- **Greedy DSpark drafter:** Only the proposal sampler switched from probabilistic to greedy. The target sampler stayed fixed. Both throughput medians fell below the neighbouring controls.
- **Classic CUDA graphs:** Set `VLLM_USE_BREAKABLE_CUDAGRAPH=0` in place of `1`. Single-stream performance stayed flat. The five-stream change was small and did not clear a useful gain at both controls.
- **DSpark off:** Omitted `--speculative-config`, retaining the rest of the launch arguments. The single-stream median fell below both controls. Five-stream results stayed in the control range.

The recipe remains on probabilistic DSpark ×7 and vLLM's breakable CUDA graphs. We kept the 32K and 128K cold-prefill results from the baseline. The tuning candidates were speed screens and did not advance to full-context qualification.

## Bandwidth and next experiments

The idle GB10 microbenchmark read and wrote a 512 MiB device buffer at **219.1 GB/s combined**. A 4096³ BF16 GEMM reached **90.1 TFLOP/s**. Counting the checkpoint's active expert bytes and shared weights across 48 effective layer visits gives a plain-decode proxy of about 5.50 GB/token. At the observed copy bandwidth that is **39.9 tok/s** before KV traffic and other overhead. This is a back-of-the-envelope estimate for the no-draft path. Drafting can share target weight reads over multiple accepted tokens.

The native draft block requires seven proposals. Research found FP8 KV on this pinned build and official GB10 SGLang support. Those are separate long-context and alternate-engine experiments and were outside this short-decode screen. [FP8 Spark configuration](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark/blob/main/launch_maxkv_fp8.sh) · [Official SGLang Spark configuration](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/configs/thinkingmachines/inkling-2-mini.jsx#L501)

## Run the screen

Start the recipe on the Spark, then run the same client with each candidate's recorded launch arguments. Run fresh controls around each candidate.

```bash
export SPARK_BENCHMARK_URL=http://127.0.0.1:18905
export SPARK_BENCHMARK_MODEL=thinkingmachines/Inkling-2-Mini
mkdir -p runs
python3 bin/tune-benchmark.py control-1 runs/control-1.decode.json
```

See [raw launch arguments, samples, and device measurements](../results/tuning/). The original Flash Next service was verified after both runs.
