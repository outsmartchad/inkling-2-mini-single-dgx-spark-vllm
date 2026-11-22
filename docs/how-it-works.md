# How the Inkling 2 Mini recipe works

## How the model works

Thinking Machines describes Inkling 2 Mini as a 26.4B mixture-of-experts model with 3.4B active parameters. The 24-layer decoder is arranged as a six-layer prelude, twelve looped layers, and a six-layer postlude. Layers 6–17 execute three times, giving `6 + 12 × 3 + 6 = 48` layer visits per token from 24 stored layers and about 5.4B effective active parameters. Each six-layer block has five sliding-window attention layers and one full-attention layer. In every layer, tokens route to 8 of 128 experts. These details come from the [release note](https://thinkingmachines.ai/news/inkling-2-mini/), [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/), and [NeMo architecture notes](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/vision-language-models/thinkingmachines/Inkling-2-Mini). The pinned checkpoint uses mostly NVFP4 weights with MXFP8 in the last four layers.

For each request, tokenization and the chat template create the target sequence. Prefill evaluates the prompt and writes attention state into the KV pool. Decode then advances one token at a time. The loop carries a message from the prior pass into the next pass. A learned rank-64 gate mixes that message with the prelude output and the current block output, separately affecting attention query, key and value inputs plus expert input. The first pass has no message. Later passes therefore reuse the same weights while retaining pass-specific state. The model card calls these attention residuals. Each pass has its own KV entries for the attention layers, which raises cache cost with loop depth. When reasoning is enabled, streamed reasoning tokens use this same decode loop.

DSpark is the speculative path. The draft block proposes up to seven tokens, then the target verifies them in one pass and accepts the longest valid prefix. The matched tuning run recorded 31,612 drafted tokens and 6,263 accepted tokens, about 19.8% accepted draft positions in that instrumentation window. The figure is workload-specific because prompt and sampling change draft agreement. [vLLM Inkling integration](https://vllm.ai/blog/2026-11-20-inkling2-mini) · [DSpark paper](https://arxiv.org/abs/2607.05147) · [Classic-graphs run counters](../results/tuning/classic-graphs.spec.metrics.txt)

## How this recipe runs it on one DGX Spark

```text
OpenAI client → 127.0.0.1:18905/v1 → vLLM scheduler and KV pool
             → Inkling target plus DSpark proposal/verification
             → streamed reasoning or answer
```

`./bin/recipe.sh fetch` pulls the pinned ARM64 image and checkpoint. `serve` mounts the immutable Hugging Face snapshot and a writable kernel cache, then launches vLLM with the exact arguments in [`results/baseline/launch.args.txt`](../results/baseline/launch.args.txt). The important serving limits are 262,144 tokens per request, five sequences, 8,192 maximum batched prefill tokens, and `gpu-memory-utilization 0.70`. Prefix caching is disabled. The parser handles Inkling reasoning and tools, and the multimodal limit allows one image, although the published measurements are text-only.

The API is loopback-only. A second terminal can call `./bin/recipe.sh smoke`, or send OpenAI-compatible streaming requests as shown in [`README.md`](../README.md). `stop` removes the container while the checkpoint and runtime cache remain on disk.

### Capacity and measured limits

The baseline reports a shared KV pool of 599,083 tokens. That is enough for five short requests, yet less than five simultaneous 262K contexts. The sampled GPU-process peak was **81.7 GiB**, and the server used a `0.70` allocation target. Three fresh 32K prompts reached a median **3,149 input tok/s** with **10.40 s TTFT**. Fresh 128K prompts reached **2,464 input tok/s** with **53.19 s TTFT**. Decode reached **42.7 tok/s** for one stream and **107.8 aggregate tok/s** for five streams. These measurements come from [`results/baseline/summary.json`](../results/baseline/summary.json).

The runtime pays prefill work at long input lengths, allocates pass-specific KV, and revisits the looped weights on each forward. Rapidly rejected drafts leave more target verification work per emitted token. The tuning artifacts show that graph mode and draft sampling move throughput only modestly on this workload. Exact kernel occupancy, router decisions, and per-layer latency require a trace.

The serving recipe keeps automatic KV dtype. In the matched screen, FP8 E4M3 increased reported KV capacity by 40–41% at a similar sampled memory peak, with slightly lower five-user throughput and a 2–3 second increase in 128K TTFT. Each profile recovered the audit code on its 128K probe. [FP8 KV comparison](optimization.md#fp8-kv-cache)
