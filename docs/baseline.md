# Inkling 2 Mini on one DGX Spark

Measured November 21, 2026 on the Spark's local API. The official 26.4B model uses mixed NVFP4 and MXFP8 weights, with 3.4B active parameters per token. [Model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

## Decode

Fixed 400-token outputs on the SparkDash hash-map prose workload, reasoning effort high, temperature 0.7, top-p 1, and top-k 20. The table uses the highest aggregate of three waves at each concurrency. Per-stream rate and TTFT are the medians within that wave. All 45 requests streamed reasoning and completed 400 tokens. Prefix caching was disabled.

| Streams | Aggregate decode tok/s | Per stream tok/s | Median TTFT |
|---:|---:|---:|---:|
| 1 | 42.7 | 42.7 | 219 ms |
| 2 | 63.0 | 32.2 | 328 ms |
| 3 | 76.7 | 26.2 | 245 ms |
| 4 | 94.0 | 23.7 | 268 ms |
| 5 | 107.8 | 22.4 | 1.73 s |

TTFT runs from the start of the request to the first reasoning or answer chunk. Some concurrency waves encountered first-use Triton compilation.

## Cold prefill

One warm-up and three measured fresh prompts per size. Both figures are medians from direct HTTP requests on the Spark. Effective prefill rate is prompt tokens divided by client TTFT. All six measured requests reported zero cached prompt tokens.

| Target | Actual prompt tokens | Median prefill tok/s | Median TTFT | Exact retrieval |
|---:|---:|---:|---:|---:|
| 32K | 32,758–32,764 | 3,149 | 10.40 s | 2/3 |
| 128K | 131,037–131,087 | 2,464 | 53.19 s | 3/3 |

The client requested reasoning effort `none`, and the model still emitted reasoning on the synthetic retrieval task. One 32K request reached its 512-token budget without returning an answer. The TTFT and prefill rate include this sample.

## Configuration

- **Checkpoint:** `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`
- **Runtime:** ARM64 vLLM `0.1.dev25855+g7cb5dae22.cu130`
- **Container:** `vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e`
- **Drafting:** Native DSpark, seven speculative tokens, probabilistic drafting. Single GPU, up to five concurrent requests, and 262,144-token configured request context.
- **GPU cache pool:** 599,083 tokens reported by the runtime. The model loaded with bfloat16 dtype and automatic KV dtype. vLLM reported 19.08 GiB at model load.
- **Peak sampled GPU memory:** 81.7 GiB for the vLLM GPU process, across startup, decode and prefill. The sampler requested a 200 ms interval and returned 3,449 usable samples.

Runtime and measurements are reproduced in the [recipe](../README.md). The Spark's prior Flash Next service was restored after testing.

- [Decode samples](../results/baseline/inkling.decode.json)
- [Prefill samples and audit results](../results/baseline/inkling.prefill.json)
- [Process GPU memory samples](../results/baseline/inkling.memory.csv)
- [Machine-readable summary](../results/baseline/summary.json)
- [Exact launch arguments](../results/baseline/launch.args.txt)
