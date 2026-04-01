# Engram

> **Named, persistent, tiered KV-cache sessions for LLM inference.**

Instead of rebuilding the KV cache from scratch on every call, Engram *names* it, keeps it alive across requests, and resumes from exactly where it left off — only decoding the *delta* tokens (the new part of the prompt).

---

## Why it matters

Every time a conventional LLM server receives a prompt it runs **prefill**: hashing all previous tokens through every transformer layer to fill the KV cache. For long conversation histories or fixed system prompts this is enormously wasteful.

Engram treats the KV cache as a **named, persistent object**:

- First call: full prefill (unavoidable).
- Every subsequent call on the same session: **only new tokens are processed**, not the history.
- Sessions survive across API calls, HTTP connection resets, and (in future) server restarts.

This is especially valuable for:
- **Agentic pipelines** — long-running agents that keep appending to a conversation.
- **Document Q&A** — embed a 10k-token document once; ask many questions cheaply.
- **System-prompt caching** — share a single prefilled context across thousands of users.

---

## Benchmark results

Measured on **Apple M-series (CPU-only, Metal off)** with **Mistral 7B Q4_K_M**, 10-turn conversation, 128 tokens generated per turn.

### Prefill latency per turn

| Mode | Turn 1 | Turn 10 | Slowdown |
|------|--------|---------|----------|
| **Baseline** (Ollama, full history) | 4.86s | 164.68s | **33.9×** |
| **Engram chat** (delta-only prefill) | 1.03s | 3.07s | **3.0×** |

> **54× faster prefill at turn 10.** The baseline re-processes the entire conversation history on every call (O(n²)). Engram only processes the new delta tokens (O(delta × cached)), so prefill stays nearly flat.

### Engram chat — per-turn breakdown

```
 Turn  Cached tok  Delta tok  Prefill   Decode
 1     0           19         1.03s     29.13s   (cold start)
 2     147         19         1.12s     29.83s   ✓ cache hit
 3     294         18         1.27s     30.31s   ✓ cache hit
 4     440         19         1.27s     31.65s   ✓ cache hit
 5     587         22         1.45s     31.15s   ✓ cache hit
 6     737         24         1.49s     31.44s   ✓ cache hit
 7     889         19         1.32s     31.75s   ✓ cache hit
 8     1036        18         1.29s     32.84s   ✓ cache hit
 9     1182        20         1.39s     32.71s   ✓ cache hit
 10    1330        27         3.07s     39.28s   ✓ cache hit
```

**Delta tokens per turn stay at 18–27** regardless of how much history has accumulated — only the new user message and a couple of template framing tokens need prefilling. The rest is served from the persistent KV cache.

The slow prefill growth (1.03s → 3.07s) is unavoidable physics: each new token still attends to all cached keys/values. The formula is `prefill ∝ delta_tokens × cached_tokens`. At turn 10 that's `27 × 1330 ≈ 36k` ops vs the baseline's `1338 × 1338 ≈ 1.8M` ops — a **50× reduction in attention work**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       HTTP Clients                        │
└─────────────────────┬────────────────────────────────────┘
                      │  JSON / HTTP
┌─────────────────────▼────────────────────────────────────┐
│                   Engram HTTP Server                      │
│              (cpp-httplib, multi-threaded)                │
│                                                           │
│  POST /session/create   POST /session/infer               │
│  DELETE /session/evict  GET  /session/status              │
│  GET  /sessions                                           │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│                    SessionStore                           │
│              (mutex-protected, LRU eviction)              │
│                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐  │
│  │   HOT    │   │   WARM   │   │        COLD          │  │
│  │  tier    │   │  tier    │   │        tier          │  │
│  │          │   │          │   │                      │  │
│  │ live     │   │ KV bytes │   │ .kv blobs on NVMe    │  │
│  │ llama_   │   │ in RAM   │   │ (filesystem)         │  │
│  │ context  │   │          │   │                      │  │
│  │          │   │          │   │                      │  │
│  │ max_hot  │   │ max_warm │   │ unlimited            │  │
│  │ sessions │   │ sessions │   │                      │  │
│  └────┬─────┘   └────┬─────┘   └──────────────────────┘  │
│       │  LRU evict   │  LRU evict                         │
│       └──────────────┘──────────►  serialize to disk      │
│                                                           │
│  Model cache: shared llama_model* (ref-counted)           │
└──────────────────────────────────────────────────────────┘
```

### Tiered storage

| Tier | Storage    | Latency to resume | Capacity      |
|------|------------|-------------------|---------------|
| HOT  | VRAM / RAM | 0 ms (live)       | `--max-hot`   |
| WARM | RAM        | ~1–10 ms          | `--max-warm`  |
| COLD | NVMe disk  | ~10–100 ms        | Unlimited     |

Sessions are promoted *up* (COLD→WARM→HOT) on access and demoted *down* (HOT→WARM→COLD) by LRU eviction when a tier is full.

---

## Build

### Requirements

- CMake ≥ 3.20
- C++17 compiler (GCC 11+ or Clang 13+)
- Git (for FetchContent)
- Optional: CUDA/Metal for GPU acceleration

### Build steps

```bash
git clone https://github.com/yourname/engram
cd engram
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The binary will be at `build/engram`.

---

## Usage

### Start the server

```bash
./build/engram --host 127.0.0.1 --port 8080 \
               --model /models/llama-3-8b-q4.gguf \
               --max-hot 4 --max-warm 16 \
               --sessions-dir ./sessions \
               --threads 8 --gpu-layers 35
```

The `--model` flag sets a server-wide default so clients don't need to specify it on every `/session/create` call.

### Create a session

```bash
# Explicit model path:
curl -s -X POST http://localhost:8080/session/create \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "agent-001",
    "model": "/models/llama-3-8b-q4.gguf",
    "n_ctx": 4096,
    "n_gpu_layers": 35
  }' | jq

# Or omit "model" if --model was given to the server:
curl -s -X POST http://localhost:8080/session/create \
  -H "Content-Type: application/json" \
  -d '{"session_id": "agent-001"}' | jq
```

```json
{
  "session_id": "agent-001",
  "status": "created",
  "tier": "hot"
}
```

### Run inference (first call — cold prefill)

```bash
curl -s -X POST http://localhost:8080/session/infer \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "agent-001",
    "prompt": "You are a helpful assistant.\n\nUser: What is 2+2?\nAssistant:",
    "n_predict": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40
  }' | jq
```

```json
{
  "session_id": "agent-001",
  "text": " 4",
  "n_tokens_prompt": 23,
  "n_tokens_generated": 3,
  "ms_prefill": 312.4,
  "ms_decode": 42.1,
  "cache_hit": false,
  "n_tokens_in_cache": 0
}
```

### Continue conversation (cache hit — only new tokens prefilled)

```bash
curl -s -X POST http://localhost:8080/session/infer \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "agent-001",
    "prompt": "\n\nUser: And 3+3?\nAssistant:",
    "n_predict": 64,
    "temperature": 0.7
  }' | jq
```

```json
{
  "session_id": "agent-001",
  "text": " 6",
  "n_tokens_prompt": 10,
  "n_tokens_generated": 3,
  "ms_prefill": 18.2,
  "ms_decode": 14.8,
  "cache_hit": true,
  "n_tokens_in_cache": 26
}
```

> Notice the prefill dropped from 312ms → 18ms. That's the cache working.

### Chat with automatic template formatting (modern instruct models)

Instead of manually formatting prompts, use `/session/chat` with a `messages` array.
Engram automatically applies the model's built-in chat template (Llama 3, Mistral, Gemma,
Qwen, Phi, etc.) or falls back to ChatML if the model has none.

```bash
curl -s -X POST http://localhost:8080/session/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "agent-001",
    "messages": [
      {"role": "system",    "content": "You are a helpful assistant."},
      {"role": "user",      "content": "What is 2+2?"}
    ],
    "n_predict": 64,
    "temperature": 0.7
  }' | jq
```

```json
{
  "session_id": "agent-001",
  "text": "4",
  "n_tokens_prompt": 34,
  "n_tokens_generated": 3,
  "ms_prefill": 320.1,
  "ms_decode": 45.2,
  "cache_hit": false,
  "n_tokens_in_cache": 0,
  "formatted_prompt": "<|begin_of_text|>...<|start_header_id|>assistant<|end_header_id|>\n\n"
}
```

Continue the conversation by including prior turns:

```bash
curl -s -X POST http://localhost:8080/session/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "agent-001",
    "messages": [
      {"role": "system",    "content": "You are a helpful assistant."},
      {"role": "user",      "content": "What is 2+2?"},
      {"role": "assistant", "content": "4"},
      {"role": "user",      "content": "And 3+3?"}
    ],
    "n_predict": 64
  }' | jq
```

> **Tip:** Only the *delta* tokens (new turns) cost prefill time — the rest is served from cache.

### Inspect model info

```bash
curl -s "http://localhost:8080/model/info?session_id=agent-001" | jq
```

```json
{
  "arch":          "llama",
  "chat_template": "{% set loop_messages = messages %}...",
  "n_ctx_train":   "131072",
  "model_path":    "/models/llama-3-8b-q4.gguf",
  "tier":          "hot"
}
```

### Check session status

```bash
curl -s "http://localhost:8080/session/status?session_id=agent-001" | jq
```

```json
{
  "session_id": "agent-001",
  "tier": "hot",
  "n_tokens_used": 36,
  "last_accessed": "2025-06-01T10:23:14Z",
  "created_at": "2025-06-01T10:20:00Z",
  "model": "/models/llama-3-8b-q4.gguf"
}
```

### List all sessions

```bash
curl -s http://localhost:8080/sessions | jq
```

```json
{
  "sessions": [
    { "session_id": "agent-001", "tier": "hot",  "n_tokens_used": 36, ... },
    { "session_id": "agent-002", "tier": "warm", "n_tokens_used": 512, ... },
    { "session_id": "agent-003", "tier": "cold", "n_tokens_used": 0,  ... }
  ],
  "hot": 1,
  "warm": 1,
  "cold": 1
}
```

### Evict a session

```bash
curl -s -X DELETE http://localhost:8080/session/evict \
  -H "Content-Type: application/json" \
  -d '{"session_id": "agent-001"}' | jq
```

```json
{ "status": "evicted" }
```

---

## Configuration reference

| Flag             | Default        | Description                              |
|------------------|----------------|------------------------------------------|
| `--host`         | `127.0.0.1`    | Bind address                             |
| `--port`         | `8080`         | TCP port                                 |
| `--max-hot`      | `4`            | Max live `llama_context` objects in RAM  |
| `--max-warm`     | `16`           | Max serialised KV blobs kept in RAM      |
| `--sessions-dir` | `./sessions`   | Directory for COLD `.kv` blobs on disk   |
| `--model`        | *(none)*       | Default model path (used when `/session/create` omits `"model"`) |
| `--threads`      | `4`            | llama.cpp CPU threads                    |
| `--gpu-layers`   | `0`            | Layers offloaded to GPU (0 = CPU-only)   |

---

## Roadmap

- [ ] **Streaming** — chunked HTTP responses via SSE or chunked transfer encoding
- [ ] **Sequence slots** — multi-sequence batching within a single context (requires llama_seq API)
- [ ] **Quantized KV cache** — reduce VRAM footprint via Q8/Q4 KV quantisation
- [ ] **Eviction webhooks** — notify external systems when a session is demoted
- [ ] **Session metadata sidecars** — persist model path + token count for COLD sessions
- [ ] **Snapshot API** — fork a session at a given token offset (tree-of-thought branching)
- [ ] **gRPC transport** — lower latency for high-throughput agentic pipelines
- [ ] **Distributed tier** — WARM tier backed by Redis; COLD tier backed by object storage

---

## License

MIT
