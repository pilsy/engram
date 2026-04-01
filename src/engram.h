#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

// Forward-declare llama types so headers don't need to pull in all of llama.cpp
struct llama_model;
struct llama_context;

namespace engram {

// ──────────────────────────────────────────────
// Server configuration (populated from CLI args)
// ──────────────────────────────────────────────
struct Config {
    std::string host            = "127.0.0.1";
    int         port            = 8080;
    int         max_hot_sessions  = 4;   // live llama_context objects kept in memory
    int         max_warm_sessions = 16;  // serialised KV state kept in RAM
    std::string cold_storage_path = "./sessions"; // disk directory for cold KV blobs
    int         default_n_ctx  = 4096;
    int         n_threads      = 4;
    int         n_gpu_layers   = 0;
    std::string default_model_path; // optional default model for /session/create
};

// ──────────────────────────────────────────────
// Storage tier
// ──────────────────────────────────────────────
enum class Tier { HOT, WARM, COLD };

inline const char* tier_name(Tier t) {
    switch (t) {
        case Tier::HOT:  return "hot";
        case Tier::WARM: return "warm";
        case Tier::COLD: return "cold";
    }
    return "unknown";
}

// ──────────────────────────────────────────────
// Metadata that is common to every tier
// ──────────────────────────────────────────────
struct SessionMeta {
    std::string session_id;
    std::string model_path;
    int         n_ctx         = 0;
    int         n_gpu_layers  = 0;  // GPU layers the model was loaded with
    int         n_tokens_used = 0;  // tokens currently occupying the KV cache
    Tier        tier          = Tier::HOT;

    // Time-points stored as epoch seconds for easy JSON serialisation
    std::int64_t last_accessed = 0; // unix timestamp
    std::int64_t created_at    = 0; // unix timestamp
};

// ──────────────────────────────────────────────
// HOT session – live context ready to decode
// ──────────────────────────────────────────────
struct HotSession {
    SessionMeta    meta;
    llama_model*   model = nullptr; // NOT owned – managed by model cache
    llama_context* ctx   = nullptr; // OWNED – freed in SessionStore

    // The exact token IDs currently occupying the KV cache (in order).
    // Used by chat_infer() to compute the true delta via prefix matching
    // instead of relying on a raw token count, which breaks when the chat
    // template adds/changes special tokens around generated replies.
    std::vector<int32_t> kv_tokens;
};

// ──────────────────────────────────────────────
// WARM session – serialised KV state in RAM
// ──────────────────────────────────────────────
struct WarmSession {
    SessionMeta          meta;
    std::vector<uint8_t> kv_state; // raw bytes from llama_state_get_data
};

// ──────────────────────────────────────────────
// A single chat message (role + content)
// Used by chat_infer() for template-aware inference
// ──────────────────────────────────────────────
struct ChatMessage {
    std::string role;    // "system", "user", "assistant"
    std::string content;
};

// ──────────────────────────────────────────────
// Result returned from SessionStore::infer()
// ──────────────────────────────────────────────
struct InferResult {
    std::string text;
    int         n_tokens_prompt    = 0;
    int         n_tokens_generated = 0;
    double      ms_prefill         = 0.0;
    double      ms_decode          = 0.0;
    bool        cache_hit          = false;
    int         n_tokens_in_cache  = 0;
    std::string formatted_prompt;  // the prompt after template application (for debugging)
    std::string error; // non-empty on failure
};

} // namespace engram
