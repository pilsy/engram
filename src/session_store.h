#pragma once

#include "engram.h"

#include <map>
#include <set>
#include <mutex>
#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace engram {

// ──────────────────────────────────────────────
// SessionStore
//
// Thread-safe store for named KV-cache sessions.
// Sessions live in one of three tiers:
//   HOT  – live llama_context in VRAM/RAM, zero-latency decode
//   WARM – serialised bytes in RAM, ~milliseconds to restore
//   COLD – serialised bytes on NVMe, ~tens of milliseconds to restore
//
// Promotion:  COLD → WARM → HOT  (on access)
// Demotion:   HOT  → WARM → COLD (LRU eviction when tier is full)
// ──────────────────────────────────────────────
class SessionStore {
public:
    explicit SessionStore(const Config& cfg);
    ~SessionStore();

    // Create a new session (initially HOT).
    // Returns an empty string on success, or an error message on failure.
    std::string create_session(const std::string& session_id,
                               const std::string& model_path,
                               int n_ctx,
                               int n_gpu_layers);

    // Return the server-wide default model path (may be empty).
    std::string default_model_path() const { return cfg_.default_model_path; }

    // Run inference on an existing session (raw prompt string).
    InferResult infer(const std::string& session_id,
                      const std::string& prompt,
                      int   n_predict,
                      float temperature,
                      float top_p,
                      int   top_k);

    // Run inference using a chat messages array.
    // Automatically applies the model's built-in chat template (if any),
    // falling back to a generic ChatML format for models without one.
    InferResult chat_infer(const std::string&              session_id,
                           const std::vector<ChatMessage>& messages,
                           int   n_predict,
                           float temperature,
                           float top_p,
                           int   top_k,
                           bool  add_generation_prompt = true);

    // Return metadata about the model loaded for a given session.
    // Keys: "arch", "chat_template", "n_params", "n_ctx_train"
    // Returns empty map if session not found or model not loaded.
    std::map<std::string, std::string> model_info(const std::string& session_id) const;

    // Permanently remove a session from all tiers.
    std::string evict_session(const std::string& session_id);

    // Return metadata for a single session.
    std::optional<SessionMeta> get_status(const std::string& session_id) const;

    // Return metadata for every known session.
    std::vector<SessionMeta> list_sessions() const;

private:
    // ── Model cache ──────────────────────────────
    // Key: (model_path, n_gpu_layers) — different GPU configurations are
    // distinct model instances with separate weight layouts.
    struct ModelKey {
        std::string path;
        int         n_gpu_layers = 0;
        bool operator<(const ModelKey& o) const {
            return path < o.path || (path == o.path && n_gpu_layers < o.n_gpu_layers);
        }
    };

    struct ModelEntry {
        llama_model* model    = nullptr;
        int          refcount = 0;
    };

    // Load (or reuse) a model; increments refcount.
    // Returns nullptr on failure and writes error into `out_error`.
    llama_model* load_model(const std::string& model_path,
                            int n_gpu_layers,
                            std::string& out_error);

    // Decrement refcount and free model if it reaches zero.
    void release_model(const std::string& model_path, int n_gpu_layers);

    // ── Tier management ──────────────────────────

    // Ensure the session is in the HOT tier (promotes if needed).
    // Assumes lock is already held.
    // Returns error string on failure.
    std::string ensure_hot_locked(const std::string& session_id);

    // Demote the least-recently-used HOT session to WARM.
    // Assumes lock is already held.
    void demote_lru_hot_locked();

    // Demote the least-recently-used WARM session to COLD.
    // Assumes lock is already held.
    void demote_lru_warm_locked();

    // Promote a WARM session to HOT.  Assumes lock held.
    std::string promote_warm_to_hot_locked(const std::string& session_id);

    // Promote a COLD session to WARM.  Assumes lock held.
    std::string promote_cold_to_warm_locked(const std::string& session_id);

    // ── Helpers ──────────────────────────────────
    // Path for a COLD session blob.
    std::string cold_path(const std::string& session_id) const;

    // Current unix timestamp.
    static std::int64_t now_epoch();

    // ── Data members ─────────────────────────────
    Config cfg_;
    mutable std::mutex mtx_;

    std::map<ModelKey, ModelEntry>     model_cache_; // (model_path, gpu_layers) → entry
    std::map<std::string, HotSession>  hot_;         // session_id → hot session
    std::map<std::string, WarmSession> warm_;        // session_id → warm session
    std::set<std::string>              cold_;        // session_ids with blobs on disk
};

} // namespace engram
