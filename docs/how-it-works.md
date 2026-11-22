# How the Inkling 2 Mini recipe works

## Model and engine

Inkling 2 Mini is a 26.4B mixture-of-experts model with 3.4B active parameters. Its middle 12 layers run three times with shared weights, for 48 layer visits per token and 5.4B effective active parameters. Most of the checkpoint uses NVFP4, with MXFP8 in the last four layers. [Thinking Machines release](https://thinkingmachines.ai/news/inkling-2-mini/) · [Model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

The pinned vLLM image includes Inkling’s looped attention, reasoning and tool-call parsers, and native DSpark drafting. DSpark proposes seven tokens using its draft block. vLLM verifies the proposals with the target model. The acceptance rate depends on the prompt and sampling settings, which is why the recipe benchmarks an actual chat prompt. [vLLM integration](https://vllm.ai/blog/2026-11-20-inkling2-mini)

## Context and memory

The recipe configures a 262,144-token context limit per request and up to five concurrent requests. All active requests share the engine’s KV cache. The baseline reported a pool of 599,083 tokens, sufficient for five short benchmark prompts but below the total for five 262K requests. The `0.70` GPU-memory-utilization setting is a vLLM allocation target, while the published **81.7 GiB** is a sampled GPU-process peak across startup and both benchmarks. [Measured configuration](baseline.md#configuration)

Automatic KV dtype and probabilistic DSpark ×7 remain the default. In the matched screen, FP8 E4M3 increased reported KV capacity by 40–41%, with similar sampled GPU footprint and slightly lower five-user aggregate throughput. The recorded 128K retrieval probe passed on both KV profiles. [Tuning and FP8 KV results](optimization.md)

## Request flow

```text
OpenAI-compatible client on the Spark
  → HTTP on 127.0.0.1:18905/v1
  → vLLM chat template, continuous batching, and KV cache
  → Inkling 2 Mini target model and DSpark verification
  → streamed reasoning or answer
```

`fetch` downloads the artifacts before inference. `serve` loads the pinned local snapshot and uses a separate writable engine cache for compiled kernels. The API listens only on the Spark’s loopback interface. The recipe enables the text and image path with one image per request. The published benchmark measures text requests. [vLLM’s Inkling serving guide](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini)
