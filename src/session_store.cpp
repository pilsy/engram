#include "session_store.h"

#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace fs = std::filesystem;

namespace engram {

// ─────────────────────────────────────────────────────────────────────────────
// Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

SessionStore::SessionStore(const Config& cfg) : cfg_(cfg) {
    // Ensure cold storage directory exists.
    fs::create_directories(cfg_.cold_storage_path);

    // Scan for existing cold blobs so they show up in list_sessions().
    for (const auto& entry : fs::directory_iterator(cfg_.cold_storage_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".kv") {
            cold_.insert(entry.path().stem().string());
        }
    }

    if (!cold_.empty()) {
        std::cout << "[engram] Found " << cold_.size()
                  << " cold session(s) on disk.\n";
    }
}

SessionStore::~SessionStore() {
    std::lock_guard<std::mutex> lk(mtx_);

    // Free all live contexts.
    for (auto& [id, hs] : hot_) {
        if (hs.ctx)   llama_free(hs.ctx);
        release_model(hs.meta.model_path, hs.meta.n_gpu_layers);
    }
    hot_.clear();

    // WARM sessions only hold byte vectors — nothing to free explicitly.
    warm_.clear();

    // Release any models that slipped through (shouldn't happen if refcounts
    // are correct, but be defensive).
    for (auto& [key, entry] : model_cache_) {
        if (entry.model) {
            std::cout << "[engram] Freeing model: " << key.path
                      << " gpu_layers=" << key.n_gpu_layers << "\n";
            llama_model_free(entry.model);
        }
    }
    model_cache_.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

std::string SessionStore::create_session(const std::string& session_id,
                                         const std::string& model_path,
                                         int n_ctx,
                                         int n_gpu_layers) {
    std::lock_guard<std::mutex> lk(mtx_);

    // Reject duplicate session ids.
    if (hot_.count(session_id) || warm_.count(session_id) || cold_.count(session_id)) {
        return "session already exists: " + session_id;
    }

    // Resolve n_ctx default.
    if (n_ctx <= 0) n_ctx = cfg_.default_n_ctx;

    // Resolve n_gpu_layers: per-session value wins; fall back to server default.
    if (n_gpu_layers < 0) n_gpu_layers = cfg_.n_gpu_layers;

    // Load (or reuse) the model.
    std::string model_err;
    llama_model* model = load_model(model_path, n_gpu_layers, model_err);
    if (!model) {
        return "failed to load model '" + model_path + "': " + model_err;
    }

    // Create a fresh llama context.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx         = static_cast<uint32_t>(n_ctx);
    cparams.n_threads     = static_cast<uint32_t>(cfg_.n_threads);
    // Flash Attention is not fully supported on Metal — some tensors fall back
    // to CPU causing a device-scheduling conflict.  Disable it when any layers
    // are offloaded to GPU.
    if (n_gpu_layers > 0) cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        release_model(model_path, n_gpu_layers);
        return "failed to create llama context for session: " + session_id;
    }

    // Demote LRU sessions if the HOT tier is at capacity.
    while (static_cast<int>(hot_.size()) >= cfg_.max_hot_sessions) {
        demote_lru_hot_locked();
    }

    HotSession hs;
    hs.meta.session_id    = session_id;
    hs.meta.model_path    = model_path;
    hs.meta.n_ctx         = n_ctx;
    hs.meta.n_gpu_layers  = n_gpu_layers;
    hs.meta.n_tokens_used = 0;
    hs.meta.tier          = Tier::HOT;
    hs.meta.created_at    = now_epoch();
    hs.meta.last_accessed = hs.meta.created_at;
    hs.model              = model;
    hs.ctx                = ctx;

    hot_.emplace(session_id, std::move(hs));
    std::cout << "[engram] Session created: " << session_id << " (hot)\n";
    return {};
}

// ─────────────────────────────────────────────────────────────────────────────

InferResult SessionStore::infer(const std::string& session_id,
                                const std::string& prompt,
                                int   n_predict,
                                float temperature,
                                float top_p,
                                int   top_k) {
    InferResult result;
    std::unique_lock<std::mutex> lk(mtx_);

    // Promote session to HOT if needed.
    std::string err = ensure_hot_locked(session_id);
    if (!err.empty()) {
        result.error = err;
        return result;
    }

    HotSession& hs = hot_.at(session_id);
    llama_model*   model = hs.model;
    llama_context* ctx   = hs.ctx;

    // Remember how many tokens were already in the cache.
    int tokens_before      = hs.meta.n_tokens_used;
    result.n_tokens_in_cache = tokens_before;
    result.cache_hit        = (tokens_before > 0);

    // Tokenize the new prompt.
    // We allocate enough space for the full context.
    std::vector<llama_token> tokens(hs.meta.n_ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_prompt_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        /*add_special=*/true,
        /*parse_special=*/false
    );

    if (n_prompt_tokens < 0) {
        result.error = "tokenize failed (prompt too long?)";
        return result;
    }
    tokens.resize(n_prompt_tokens);
    result.n_tokens_prompt = n_prompt_tokens;

    // ── Prefill phase ────────────────────────────────────────────────────────
    auto t_prefill_start = std::chrono::high_resolution_clock::now();

    // Feed prompt tokens to the context, positioned AFTER the existing cache.
    // llama_batch_get_one() always uses pos=0,1,2,... which would overwrite the
    // existing KV cache entries — so we must set explicit positions here.
    {
        llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);
        batch.n_tokens = n_prompt_tokens;
        for (int j = 0; j < n_prompt_tokens; ++j) {
            batch.token[j]     = tokens[j];
            batch.pos[j]       = tokens_before + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            // Only need logits for the very last token (used for first sample).
            batch.logits[j]    = (j == n_prompt_tokens - 1) ? 1 : 0;
        }
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            result.error = "llama_decode failed during prefill (rc=" + std::to_string(rc) + ")";
            return result;
        }
    }

    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    result.ms_prefill = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();

    // ── Sampling / decode phase ──────────────────────────────────────────────
    //
    // Build a sampler chain.  We use a unique_ptr with a custom deleter so the
    // chain is freed even if we return early.
    struct SamplerDeleter { void operator()(llama_sampler* s) { llama_sampler_free(s); } };
    std::unique_ptr<llama_sampler, SamplerDeleter> smpl(
        llama_sampler_chain_init(llama_sampler_chain_default_params())
    );

    llama_sampler_chain_add(smpl.get(), llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const llama_token eos_token = llama_vocab_eos(vocab);

    std::string generated_text;
    generated_text.reserve(n_predict * 4); // rough estimate

    auto t_decode_start = std::chrono::high_resolution_clock::now();

    int n_generated = 0;
    int cur_pos     = tokens_before + n_prompt_tokens; // next KV cache position

    for (int i = 0; i < n_predict; ++i) {
        // Sample the next token from the last logits.
        llama_token tok = llama_sampler_sample(smpl.get(), ctx, -1);

        if (tok == eos_token) {
            break;
        }

        // Detokenize and append to output.
        char piece[256];
        int  piece_len = llama_token_to_piece(
            vocab, tok, piece, sizeof(piece), /*lstrip=*/0, /*special=*/false);
        if (piece_len > 0) {
            generated_text.append(piece, piece_len);
        }

        // Decode the sampled token with explicit position so it lands correctly
        // in the KV cache (llama_batch_get_one always starts at pos=0).
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens     = 1;
        batch.token[0]     = tok;
        batch.pos[0]       = cur_pos++;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            // Non-fatal: stop generation but don't error.
            std::cerr << "[engram] llama_decode returned " << rc
                      << " during token " << i << " – stopping.\n";
            break;
        }

        ++n_generated;
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    result.ms_decode = std::chrono::duration<double, std::milli>(
        t_decode_end - t_decode_start).count();

    // ── Update session state ─────────────────────────────────────────────────
    hs.meta.n_tokens_used  = tokens_before + n_prompt_tokens + n_generated;
    hs.meta.last_accessed  = now_epoch();
    hs.meta.tier           = Tier::HOT;

    result.text              = std::move(generated_text);
    result.n_tokens_generated = n_generated;

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────

InferResult SessionStore::chat_infer(const std::string&              session_id,
                                     const std::vector<ChatMessage>& messages,
                                     int   n_predict,
                                     float temperature,
                                     float top_p,
                                     int   top_k,
                                     bool  add_generation_prompt) {
    InferResult result;

    // We need the model pointer to apply the chat template.
    // Promote to HOT first (also validates session_id).
    {
        std::unique_lock<std::mutex> lk(mtx_);
        std::string err = ensure_hot_locked(session_id);
        if (!err.empty()) {
            result.error = err;
            return result;
        }
    }

    // Apply chat template — build a single formatted prompt string.
    std::string formatted;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto hot_it = hot_.find(session_id);
        if (hot_it == hot_.end()) {
            result.error = "session not found after promotion: " + session_id;
            return result;
        }
        llama_model* model = hot_it->second.model;

        // Build llama_chat_message array.
        std::vector<llama_chat_message> lmsgs;
        lmsgs.reserve(messages.size());
        for (const auto& m : messages) {
            lmsgs.push_back({m.role.c_str(), m.content.c_str()});
        }

        // Fetch the model's built-in chat template string (may be nullptr).
        const char* tmpl_str = llama_model_chat_template(model, /*name=*/nullptr);

        // First pass: measure required buffer size.
        int needed = llama_chat_apply_template(
            tmpl_str,
            lmsgs.data(), lmsgs.size(),
            add_generation_prompt,
            nullptr, 0);

        if (needed < 0) {
            // Model has no built-in template — fall back to ChatML.
            std::string fallback;
            for (const auto& m : messages) {
                fallback += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
            }
            if (add_generation_prompt) {
                fallback += "<|im_start|>assistant\n";
            }
            formatted = std::move(fallback);
            std::cout << "[engram] chat_infer: no built-in template — using ChatML fallback\n";
        } else {
            formatted.resize(static_cast<size_t>(needed));
            llama_chat_apply_template(
                tmpl_str,
                lmsgs.data(), lmsgs.size(),
                add_generation_prompt,
                &formatted[0], needed);
        }
    }

    // ── Delta-aware inference ────────────────────────────────────────────────
    //
    // Tokenize the FULL formatted conversation, then skip the tokens already
    // sitting in the KV cache.  Only the delta (new tokens since last turn)
    // gets submitted to llama_decode — that's where the O(n²) saving comes from.
    std::unique_lock<std::mutex> lk(mtx_);

    auto hot_it = hot_.find(session_id);
    if (hot_it == hot_.end()) {
        result.error = "session not found after template build: " + session_id;
        return result;
    }
    HotSession&    hs    = hot_it->second;
    llama_model*   model = hs.model;
    llama_context* ctx   = hs.ctx;

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Tokenize the whole formatted string.
    std::vector<llama_token> all_tokens(hs.meta.n_ctx);
    int n_all = llama_tokenize(
        vocab,
        formatted.c_str(),
        static_cast<int32_t>(formatted.size()),
        all_tokens.data(),
        static_cast<int32_t>(all_tokens.size()),
        /*add_special=*/true,
        /*parse_special=*/true   // template tags like <|im_start|> are special
    );
    if (n_all < 0) {
        result.error = "chat_infer: tokenize failed (prompt too long?)";
        return result;
    }
    all_tokens.resize(n_all);

    // ── Prefix match against the actual KV token IDs ────────────────────────
    //
    // We can't trust n_tokens_used as a simple offset because the chat template
    // may add/change special tokens around generated replies (e.g. </s>,
    // <|im_end|>) that were never in the raw generated token stream.
    // Instead, compare all_tokens against hs.kv_tokens element-by-element to
    // find the longest exact common prefix — that's what's already cached.
    int tokens_before = 0;
    {
        int match_limit = std::min((int)hs.kv_tokens.size(), n_all);
        while (tokens_before < match_limit &&
               hs.kv_tokens[tokens_before] == all_tokens[tokens_before]) {
            ++tokens_before;
        }
    }

    // If the prefix shrank (history was somehow mutated), clear the cache.
    if (tokens_before < (int)hs.kv_tokens.size()) {
        // Debug: show the token IDs at the mismatch point so we can see why.
        if (tokens_before < n_all && tokens_before < (int)hs.kv_tokens.size()) {
            std::cout << "[engram] chat_infer: mismatch detail:"
                      << " kv[" << tokens_before << "]=" << hs.kv_tokens[tokens_before]
                      << " new[" << tokens_before << "]=" << all_tokens[tokens_before]
                      << " (kv[-1]=" << (tokens_before > 0 ? hs.kv_tokens[tokens_before-1] : -1)
                      << " new[-1]=" << (tokens_before > 0 ? all_tokens[tokens_before-1] : -1)
                      << ")\n";
        }
        std::cout << "[engram] chat_infer: prefix mismatch at " << tokens_before
                  << " (had " << hs.kv_tokens.size() << " cached) — truncating KV cache\n";
        // Trim the KV cache to the matched prefix length.
        llama_memory_seq_rm(llama_get_memory(ctx), 0,
                            static_cast<llama_pos>(tokens_before),
                            -1);
        hs.kv_tokens.resize(tokens_before);
        hs.meta.n_tokens_used = tokens_before;
    }

    // Delta: only the tokens not yet in the cache.
    int n_delta = n_all - tokens_before;
    result.n_tokens_in_cache = tokens_before;
    result.cache_hit         = (tokens_before > 0);
    result.n_tokens_prompt   = n_delta;

    if (n_delta <= 0) {
        result.error = "chat_infer: no new tokens to process (delta=0)";
        return result;
    }

    std::cout << "[engram] chat_infer: total=" << n_all
              << " cached=" << tokens_before
              << " delta=" << n_delta << "\n";

    // ── Prefill phase (delta tokens only) ───────────────────────────────────
    auto t_prefill_start = std::chrono::high_resolution_clock::now();

    {
        llama_batch batch = llama_batch_init(n_delta, 0, 1);
        batch.n_tokens = n_delta;
        for (int j = 0; j < n_delta; ++j) {
            batch.token[j]     = all_tokens[tokens_before + j];
            batch.pos[j]       = tokens_before + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = (j == n_delta - 1) ? 1 : 0;
        }
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            result.error = "chat_infer: llama_decode failed during prefill (rc=" + std::to_string(rc) + ")";
            return result;
        }
    }

    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    result.ms_prefill = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();

    // ── Sampling / decode phase ──────────────────────────────────────────────
    struct SamplerDeleter { void operator()(llama_sampler* s) { llama_sampler_free(s); } };
    std::unique_ptr<llama_sampler, SamplerDeleter> smpl(
        llama_sampler_chain_init(llama_sampler_chain_default_params())
    );
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const llama_token eos_token = llama_vocab_eos(vocab);

    std::string generated_text;
    generated_text.reserve(n_predict * 4);

    auto t_decode_start = std::chrono::high_resolution_clock::now();

    int n_generated = 0;
    int cur_pos     = n_all; // next position in the KV cache

    // Collect generated token IDs so kv_tokens mirrors the cache exactly.
    std::vector<int32_t> generated_ids;
    generated_ids.reserve(n_predict);

    for (int i = 0; i < n_predict; ++i) {
        llama_token tok = llama_sampler_sample(smpl.get(), ctx, -1);
        if (tok == eos_token) break;

        char piece[256];
        int  piece_len = llama_token_to_piece(
            vocab, tok, piece, sizeof(piece), 0, false);
        if (piece_len > 0) generated_text.append(piece, piece_len);

        // Decode with explicit position so generated tokens land correctly.
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens     = 1;
        batch.token[0]     = tok;
        batch.pos[0]       = cur_pos++;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            std::cerr << "[engram] chat_infer: llama_decode returned " << rc
                      << " at token " << i << " – stopping.\n";
            break;
        }
        generated_ids.push_back(tok);
        ++n_generated;
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    result.ms_decode = std::chrono::duration<double, std::milli>(
        t_decode_end - t_decode_start).count();

    // ── Update session state ─────────────────────────────────────────────────
    //
    // The KV cache now contains:
    //   positions 0..n_all-1        → formatted prompt tokens (with generation prompt)
    //   positions n_all..cur_pos-1  → raw sampled tokens (no template closing tokens)
    //
    // On the next turn, chat_infer will call llama_chat_apply_template on the
    // full conversation including the assistant reply, which produces closing
    // special tokens (</s>, <|im_end|> etc.) that the raw sampler never saw.
    //
    // So we re-apply the template right now — with add_generation_prompt=false —
    // to get the canonical representation of everything we just processed, then
    // tokenize it. That gives us the exact token sequence the next call will
    // produce for the "already seen" portion, enabling a full prefix match.
    {
        // Build messages-so-far including the assistant reply.
        std::vector<ChatMessage> msgs_with_reply = messages;
        msgs_with_reply.push_back({"assistant", generated_text});

        std::vector<llama_chat_message> lmsgs;
        lmsgs.reserve(msgs_with_reply.size());
        for (const auto& m : msgs_with_reply)
            lmsgs.push_back({m.role.c_str(), m.content.c_str()});

        const char* tmpl_str = llama_model_chat_template(model, nullptr);

        // Apply template WITHOUT generation prompt — this gives the canonical
        // closed form of the full conversation up to and including this reply.
        std::string canonical;
        int needed = llama_chat_apply_template(tmpl_str, lmsgs.data(), lmsgs.size(),
                                               /*add_gen_prompt=*/false, nullptr, 0);
        if (needed > 0) {
            canonical.resize(static_cast<size_t>(needed));
            llama_chat_apply_template(tmpl_str, lmsgs.data(), lmsgs.size(),
                                      false, &canonical[0], needed);
        } else {
            // ChatML fallback (no built-in template).
            for (const auto& m : msgs_with_reply)
                canonical += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        }

        // Tokenize the canonical form — this is what the NEXT call's all_tokens
        // will look like for the already-processed portion.
        std::vector<llama_token> canonical_tokens(hs.meta.n_ctx);
        int n_canonical = llama_tokenize(
            vocab, canonical.c_str(), static_cast<int32_t>(canonical.size()),
            canonical_tokens.data(), static_cast<int32_t>(canonical_tokens.size()),
            /*add_special=*/true, /*parse_special=*/true);

        if (n_canonical > 0) {
            // kv_tokens must mirror the ACTUAL KV cache (cur_pos entries).
            // The canonical form may have extra closing special tokens beyond
            // what the sampler decoded — trim to cur_pos so the next call's
            // prefix match never advances past the last cached position.
            int use = std::min(n_canonical, (int)cur_pos);
            hs.kv_tokens.assign(canonical_tokens.begin(),
                                 canonical_tokens.begin() + use);
            std::cout << "[engram] chat_infer: canonical=" << n_canonical
                      << " cache_pos=" << cur_pos
                      << " kv_tokens=" << use << "\n";
        } else {
            // Fallback: raw prompt + generated ids (better than nothing).
            hs.kv_tokens.assign(all_tokens.begin(), all_tokens.end());
            hs.kv_tokens.insert(hs.kv_tokens.end(), generated_ids.begin(), generated_ids.end());
            std::cout << "[engram] chat_infer: canonical tokenize failed, using raw kv_tokens="
                      << hs.kv_tokens.size() << "\n";
        }
    }

    hs.meta.n_tokens_used  = static_cast<int>(cur_pos);
    hs.meta.last_accessed  = now_epoch();
    hs.meta.tier           = Tier::HOT;

    result.text               = std::move(generated_text);
    result.n_tokens_generated = n_generated;
    result.formatted_prompt   = formatted;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────

std::map<std::string, std::string> SessionStore::model_info(const std::string& session_id) const {
    std::lock_guard<std::mutex> lk(mtx_);

    llama_model* model = nullptr;

    auto hot_it = hot_.find(session_id);
    if (hot_it != hot_.end()) {
        model = hot_it->second.model;
    } else {
        auto warm_it = warm_.find(session_id);
        if (warm_it != warm_.end()) {
            // Warm sessions have a model_path but no live model pointer.
            // Return what we know without loading.
            return {{"model_path", warm_it->second.meta.model_path},
                    {"tier",       "warm"}};
        }
        if (cold_.count(session_id)) {
            return {{"tier", "cold"}};
        }
        return {}; // not found
    }

    std::map<std::string, std::string> info;

    // Architecture string.
    {
        char buf[128] = {};
        if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) >= 0) {
            info["arch"] = buf;
        }
    }

    // Built-in chat template (if any).
    {
        const char* tmpl = llama_model_chat_template(model, nullptr);
        if (tmpl) {
            info["chat_template"] = tmpl;
        }
    }

    // Parameter count.
    {
        char buf[64] = {};
        if (llama_model_meta_val_str(model, "general.parameter_count", buf, sizeof(buf)) >= 0) {
            info["n_params"] = buf;
        }
    }

    // Training context length.
    {
        char buf[64] = {};
        if (llama_model_meta_val_str(model, "llama.context_length", buf, sizeof(buf)) >= 0) {
            info["n_ctx_train"] = buf;
        }
    }

    info["model_path"] = hot_it->second.meta.model_path;
    info["tier"]       = "hot";

    return info;
}

// ─────────────────────────────────────────────────────────────────────────────

std::string SessionStore::evict_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lk(mtx_);

    bool found = false;

    // Remove from HOT tier.
    auto hot_it = hot_.find(session_id);
    if (hot_it != hot_.end()) {
        llama_free(hot_it->second.ctx);
        release_model(hot_it->second.meta.model_path, hot_it->second.meta.n_gpu_layers);
        hot_.erase(hot_it);
        found = true;
    }

    // Remove from WARM tier.
    auto warm_it = warm_.find(session_id);
    if (warm_it != warm_.end()) {
        warm_.erase(warm_it);
        found = true;
    }

    // Remove COLD blob.
    if (cold_.count(session_id)) {
        std::error_code ec;
        fs::remove(cold_path(session_id), ec);
        cold_.erase(session_id);
        found = true;
    }

    if (!found) {
        return "session not found: " + session_id;
    }

    std::cout << "[engram] Session evicted: " << session_id << "\n";
    return {};
}

// ─────────────────────────────────────────────────────────────────────────────

std::optional<SessionMeta> SessionStore::get_status(const std::string& session_id) const {
    std::lock_guard<std::mutex> lk(mtx_);

    auto hot_it = hot_.find(session_id);
    if (hot_it != hot_.end()) return hot_it->second.meta;

    auto warm_it = warm_.find(session_id);
    if (warm_it != warm_.end()) return warm_it->second.meta;

    if (cold_.count(session_id)) {
        // For COLD sessions we don't have rich metadata without reading the
        // blob, so return a minimal struct.
        SessionMeta m;
        m.session_id = session_id;
        m.tier       = Tier::COLD;
        return m;
    }

    return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────

std::vector<SessionMeta> SessionStore::list_sessions() const {
    std::lock_guard<std::mutex> lk(mtx_);

    std::vector<SessionMeta> out;
    out.reserve(hot_.size() + warm_.size() + cold_.size());

    for (const auto& [id, hs] : hot_)  out.push_back(hs.meta);
    for (const auto& [id, ws] : warm_) out.push_back(ws.meta);
    for (const auto& id      : cold_) {
        SessionMeta m;
        m.session_id = id;
        m.tier       = Tier::COLD;
        out.push_back(m);
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

llama_model* SessionStore::load_model(const std::string& model_path,
                                      int n_gpu_layers,
                                      std::string& out_error) {
    ModelKey key{model_path, n_gpu_layers};
    auto it = model_cache_.find(key);
    if (it != model_cache_.end()) {
        it->second.refcount++;
        std::cout << "[engram] Reusing model: " << model_path
                  << " gpu_layers=" << n_gpu_layers
                  << " (refcount=" << it->second.refcount << ")\n";
        return it->second.model;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    std::cout << "[engram] Loading model: " << model_path
              << " gpu_layers=" << n_gpu_layers << " ...\n";

    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        out_error = "llama_model_load_from_file returned null";
        return nullptr;
    }

    ModelEntry entry;
    entry.model    = model;
    entry.refcount = 1;
    model_cache_.emplace(key, entry);

    std::cout << "[engram] Model loaded: " << model_path
              << " gpu_layers=" << n_gpu_layers << "\n";
    return model;
}

// ─────────────────────────────────────────────────────────────────────────────

void SessionStore::release_model(const std::string& model_path, int n_gpu_layers) {
    ModelKey key{model_path, n_gpu_layers};
    auto it = model_cache_.find(key);
    if (it == model_cache_.end()) return;

    it->second.refcount--;
    if (it->second.refcount <= 0) {
        std::cout << "[engram] Freeing model: " << model_path
                  << " gpu_layers=" << n_gpu_layers << "\n";
        llama_model_free(it->second.model);
        model_cache_.erase(it);
    }
}

// ─────────────────────────────────────────────────────────────────────────────

std::string SessionStore::ensure_hot_locked(const std::string& session_id) {
    // Already HOT — nothing to do.
    if (hot_.count(session_id)) return {};

    // Promote from WARM.
    if (warm_.count(session_id)) {
        return promote_warm_to_hot_locked(session_id);
    }

    // Promote from COLD (COLD → WARM → HOT).
    if (cold_.count(session_id)) {
        std::string err = promote_cold_to_warm_locked(session_id);
        if (!err.empty()) return err;
        return promote_warm_to_hot_locked(session_id);
    }

    return "session not found: " + session_id;
}

// ─────────────────────────────────────────────────────────────────────────────

std::string SessionStore::promote_warm_to_hot_locked(const std::string& session_id) {
    // Make room in HOT tier if necessary.
    while (static_cast<int>(hot_.size()) >= cfg_.max_hot_sessions) {
        demote_lru_hot_locked();
    }

    WarmSession ws = std::move(warm_.at(session_id));
    warm_.erase(session_id);

    // Re-load the model using the session's own gpu_layers (increments refcount).
    std::string model_err;
    llama_model* model = load_model(ws.meta.model_path, ws.meta.n_gpu_layers, model_err);
    if (!model) {
        // Put it back in warm so it isn't lost.
        warm_.emplace(session_id, std::move(ws));
        return "failed to load model during warm→hot promotion: " + model_err;
    }

    // Create a fresh context.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx         = static_cast<uint32_t>(ws.meta.n_ctx);
    cparams.n_threads     = static_cast<uint32_t>(cfg_.n_threads);
    if (ws.meta.n_gpu_layers > 0) cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        release_model(ws.meta.model_path, ws.meta.n_gpu_layers);
        warm_.emplace(session_id, std::move(ws));
        return "failed to create llama context during warm→hot promotion";
    }

    // Restore the KV state.
    size_t restored = llama_state_set_data(
        ctx, ws.kv_state.data(), ws.kv_state.size());
    if (restored == 0) {
        llama_free(ctx);
        release_model(ws.meta.model_path, ws.meta.n_gpu_layers);
        warm_.emplace(session_id, std::move(ws));
        return "llama_state_set_data returned 0 (state restore failed)";
    }

    HotSession hs;
    hs.meta        = ws.meta;
    hs.meta.tier   = Tier::HOT;
    hs.model       = model;
    hs.ctx         = ctx;

    hot_.emplace(session_id, std::move(hs));
    std::cout << "[engram] Promoted warm→hot: " << session_id << "\n";
    return {};
}

// ─────────────────────────────────────────────────────────────────────────────

std::string SessionStore::promote_cold_to_warm_locked(const std::string& session_id) {
    // Make room in WARM tier if necessary.
    while (static_cast<int>(warm_.size()) >= cfg_.max_warm_sessions) {
        demote_lru_warm_locked();
    }

    std::string path = cold_path(session_id);
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        return "cannot open cold blob: " + path;
    }

    std::streamsize sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    WarmSession ws;
    ws.kv_state.resize(static_cast<size_t>(sz));
    if (!ifs.read(reinterpret_cast<char*>(ws.kv_state.data()), sz)) {
        return "failed to read cold blob: " + path;
    }

    // We store minimal metadata in the KV blob filename; rich metadata would
    // require a sidecar file. For now populate what we can.
    ws.meta.session_id   = session_id;
    ws.meta.model_path   = ""; // unknown without sidecar
    ws.meta.tier         = Tier::WARM;
    ws.meta.last_accessed = now_epoch();

    cold_.erase(session_id);
    std::error_code ec;
    fs::remove(path, ec); // blob is now in RAM; reclaim disk

    warm_.emplace(session_id, std::move(ws));
    std::cout << "[engram] Promoted cold→warm: " << session_id << "\n";
    return {};
}

// ─────────────────────────────────────────────────────────────────────────────

void SessionStore::demote_lru_hot_locked() {
    if (hot_.empty()) return;

    // Find the least-recently-used HOT session.
    auto lru_it = std::min_element(
        hot_.begin(), hot_.end(),
        [](const auto& a, const auto& b) {
            return a.second.meta.last_accessed < b.second.meta.last_accessed;
        });

    const std::string& id = lru_it->first;
    HotSession& hs = lru_it->second;

    // Serialise KV state.
    size_t state_size = llama_state_get_size(hs.ctx);
    WarmSession ws;
    ws.meta          = hs.meta;
    ws.meta.tier     = Tier::WARM;
    ws.kv_state.resize(state_size);
    llama_state_get_data(hs.ctx, ws.kv_state.data(), state_size);

    // Make room in WARM tier if necessary.
    while (static_cast<int>(warm_.size()) >= cfg_.max_warm_sessions) {
        demote_lru_warm_locked();
    }

    // Free context; release model reference.
    llama_free(hs.ctx);
    release_model(hs.meta.model_path, hs.meta.n_gpu_layers);

    std::string session_id_copy = id;
    warm_.emplace(session_id_copy, std::move(ws));
    hot_.erase(lru_it);

    std::cout << "[engram] Demoted hot→warm: " << session_id_copy << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────

void SessionStore::demote_lru_warm_locked() {
    if (warm_.empty()) return;

    // Find the least-recently-used WARM session.
    auto lru_it = std::min_element(
        warm_.begin(), warm_.end(),
        [](const auto& a, const auto& b) {
            return a.second.meta.last_accessed < b.second.meta.last_accessed;
        });

    const std::string& id  = lru_it->first;
    WarmSession&       ws  = lru_it->second;

    // Write KV blob to disk.
    std::string path = cold_path(id);
    std::ofstream ofs(path, std::ios::binary);
    if (ofs.is_open()) {
        ofs.write(reinterpret_cast<const char*>(ws.kv_state.data()),
                  static_cast<std::streamsize>(ws.kv_state.size()));
        cold_.insert(id);
        std::cout << "[engram] Demoted warm→cold: " << id
                  << " (" << ws.kv_state.size() << " bytes)\n";
    } else {
        std::cerr << "[engram] WARNING: could not write cold blob for "
                  << id << " – session dropped.\n";
    }

    warm_.erase(lru_it);
}

// ─────────────────────────────────────────────────────────────────────────────

std::string SessionStore::cold_path(const std::string& session_id) const {
    return (fs::path(cfg_.cold_storage_path) / (session_id + ".kv")).string();
}

// ─────────────────────────────────────────────────────────────────────────────

std::int64_t SessionStore::now_epoch() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

} // namespace engram
