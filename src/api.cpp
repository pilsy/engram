#include "api.h"
#include "session_store.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace engram {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// Send a JSON error response.
void send_error(httplib::Response& res, int status, const std::string& msg) {
    res.status = status;
    res.set_content(json{{"error", msg}}.dump(), "application/json");
}

/// Convert a SessionMeta to a JSON object.
json meta_to_json(const SessionMeta& m) {
    // Format epoch seconds as an ISO-8601-like string.
    auto fmt_epoch = [](std::int64_t epoch) -> std::string {
        if (epoch == 0) return "";
        std::time_t t = static_cast<std::time_t>(epoch);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
        return buf;
    };

    return {
        {"session_id",    m.session_id},
        {"tier",          tier_name(m.tier)},
        {"n_tokens_used", m.n_tokens_used},
        {"n_gpu_layers",  m.n_gpu_layers},
        {"last_accessed", fmt_epoch(m.last_accessed)},
        {"created_at",    fmt_epoch(m.created_at)},
        {"model",         m.model_path}
    };
}

/// Parse a JSON body; returns false and writes error on failure.
bool parse_body(const httplib::Request& req,
                httplib::Response& res,
                json& out) {
    if (req.body.empty()) {
        send_error(res, 400, "request body is empty");
        return false;
    }
    try {
        out = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_error(res, 400, std::string("JSON parse error: ") + e.what());
        return false;
    }
    return true;
}

/// Safely get a field from a JSON object, or return a default.
template<typename T>
T jget(const json& j, const std::string& key, T def) {
    return j.contains(key) ? j.at(key).get<T>() : def;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Route registration
// ─────────────────────────────────────────────────────────────────────────────

void register_routes(httplib::Server& srv, SessionStore& store) {

    // ── POST /session/create ─────────────────────────────────────────────────
    srv.Post("/session/create", [&store](const httplib::Request& req,
                                         httplib::Response& res) {
        auto t0 = std::chrono::high_resolution_clock::now();

        json body;
        if (!parse_body(req, res, body)) return;

        std::string session_id = jget<std::string>(body, "session_id", "");
        std::string model_path = jget<std::string>(body, "model",      "");
        int         n_ctx       = jget<int>(body, "n_ctx",       0);
        int         n_gpu       = jget<int>(body, "n_gpu_layers", -1); // -1 = use server default

        if (session_id.empty()) { send_error(res, 400, "session_id is required"); return; }

        // Fall back to server-wide default model if the caller didn't specify one.
        if (model_path.empty()) {
            model_path = store.default_model_path();
        }
        if (model_path.empty()) { send_error(res, 400, "model is required (no default model configured)"); return; }

        std::string err = store.create_session(session_id, model_path, n_ctx, n_gpu);
        if (!err.empty()) {
            send_error(res, 500, err);
            return;
        }

        auto ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        json resp = {
            {"session_id",   session_id},
            {"status",       "created"},
            {"tier",         "hot"},
            {"n_gpu_layers", n_gpu},
            {"n_ctx",        n_ctx > 0 ? n_ctx : 4096}
        };
        res.set_content(resp.dump(), "application/json");

        std::cout << "[api] POST /session/create session=" << session_id
                  << " " << ms << "ms\n";
    });

    // ── POST /session/infer ──────────────────────────────────────────────────
    srv.Post("/session/infer", [&store](const httplib::Request& req,
                                        httplib::Response& res) {
        auto t0 = std::chrono::high_resolution_clock::now();

        json body;
        if (!parse_body(req, res, body)) return;

        std::string session_id  = jget<std::string>(body, "session_id",  "");
        std::string prompt      = jget<std::string>(body, "prompt",      "");
        int         n_predict   = jget<int>   (body, "n_predict",   128);
        float       temperature = jget<float> (body, "temperature", 0.8f);
        float       top_p       = jget<float> (body, "top_p",       0.9f);
        int         top_k       = jget<int>   (body, "top_k",       40);

        if (session_id.empty()) { send_error(res, 400, "session_id is required"); return; }
        if (prompt.empty())     { send_error(res, 400, "prompt is required");     return; }

        InferResult result = store.infer(session_id, prompt, n_predict,
                                         temperature, top_p, top_k);

        if (!result.error.empty()) {
            // Session not found → 404, other errors → 500
            int code = (result.error.find("not found") != std::string::npos) ? 404 : 500;
            send_error(res, code, result.error);
            return;
        }

        auto ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        json resp = {
            {"session_id",         session_id},
            {"text",               result.text},
            {"n_tokens_prompt",    result.n_tokens_prompt},
            {"n_tokens_generated", result.n_tokens_generated},
            {"ms_prefill",         result.ms_prefill},
            {"ms_decode",          result.ms_decode},
            {"cache_hit",          result.cache_hit},
            {"n_tokens_in_cache",  result.n_tokens_in_cache}
        };
        res.set_content(resp.dump(), "application/json");

        std::cout << "[api] POST /session/infer session=" << session_id
                  << " prompt_tokens=" << result.n_tokens_prompt
                  << " generated=" << result.n_tokens_generated
                  << " prefill=" << result.ms_prefill << "ms"
                  << " decode=" << result.ms_decode << "ms"
                  << " total=" << ms << "ms\n";
    });

    // ── DELETE /session/evict ────────────────────────────────────────────────
    srv.Delete("/session/evict", [&store](const httplib::Request& req,
                                          httplib::Response& res) {
        auto t0 = std::chrono::high_resolution_clock::now();

        json body;
        if (!parse_body(req, res, body)) return;

        std::string session_id = jget<std::string>(body, "session_id", "");
        if (session_id.empty()) { send_error(res, 400, "session_id is required"); return; }

        std::string err = store.evict_session(session_id);
        if (!err.empty()) {
            send_error(res, 404, err);
            return;
        }

        auto ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        res.set_content(json{{"status", "evicted"}}.dump(), "application/json");

        std::cout << "[api] DELETE /session/evict session=" << session_id
                  << " " << ms << "ms\n";
    });

    // ── GET /session/status ──────────────────────────────────────────────────
    srv.Get("/session/status", [&store](const httplib::Request& req,
                                        httplib::Response& res) {
        std::string session_id = req.has_param("session_id")
                               ? req.get_param_value("session_id")
                               : "";

        if (session_id.empty()) { send_error(res, 400, "session_id query param required"); return; }

        auto meta = store.get_status(session_id);
        if (!meta) {
            send_error(res, 404, "session not found: " + session_id);
            return;
        }

        res.set_content(meta_to_json(*meta).dump(), "application/json");

        std::cout << "[api] GET /session/status session=" << session_id
                  << " tier=" << tier_name(meta->tier) << "\n";
    });

    // ── POST /session/chat ───────────────────────────────────────────────────
    //
    // Template-aware chat completion.  Accepts a messages array and applies the
    // model's built-in chat template (or ChatML fallback) before inference.
    //
    // Body:
    //   {
    //     "session_id": "...",
    //     "messages": [
    //       {"role": "system",    "content": "You are helpful."},
    //       {"role": "user",      "content": "Hello!"},
    //       {"role": "assistant", "content": "Hi there!"},   // optional prior turns
    //       {"role": "user",      "content": "What is 2+2?"}
    //     ],
    //     "n_predict":   128,   // optional
    //     "temperature": 0.7,   // optional
    //     "top_p":       0.9,   // optional
    //     "top_k":       40,    // optional
    //     "add_generation_prompt": true  // optional, default true
    //   }
    srv.Post("/session/chat", [&store](const httplib::Request& req,
                                       httplib::Response& res) {
        auto t0 = std::chrono::high_resolution_clock::now();

        json body;
        if (!parse_body(req, res, body)) return;

        std::string session_id = jget<std::string>(body, "session_id", "");
        int         n_predict  = jget<int>   (body, "n_predict",   128);
        float       temperature= jget<float> (body, "temperature", 0.8f);
        float       top_p      = jget<float> (body, "top_p",       0.9f);
        int         top_k      = jget<int>   (body, "top_k",       40);
        bool        add_gen    = jget<bool>  (body, "add_generation_prompt", true);

        if (session_id.empty()) { send_error(res, 400, "session_id is required"); return; }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            send_error(res, 400, "messages array is required");
            return;
        }

        std::vector<ChatMessage> messages;
        for (const auto& m : body["messages"]) {
            if (!m.contains("role") || !m.contains("content")) {
                send_error(res, 400, "each message must have 'role' and 'content'");
                return;
            }
            ChatMessage cm;
            cm.role    = m["role"].get<std::string>();
            cm.content = m["content"].get<std::string>();
            messages.push_back(std::move(cm));
        }

        if (messages.empty()) { send_error(res, 400, "messages must not be empty"); return; }

        InferResult result = store.chat_infer(session_id, messages, n_predict,
                                              temperature, top_p, top_k, add_gen);
        if (!result.error.empty()) {
            int code = (result.error.find("not found") != std::string::npos) ? 404 : 500;
            send_error(res, code, result.error);
            return;
        }

        auto ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        json resp = {
            {"session_id",           session_id},
            {"text",                 result.text},
            {"n_tokens_prompt",      result.n_tokens_prompt},
            {"n_tokens_generated",   result.n_tokens_generated},
            {"ms_prefill",           result.ms_prefill},
            {"ms_decode",            result.ms_decode},
            {"cache_hit",            result.cache_hit},
            {"n_tokens_in_cache",    result.n_tokens_in_cache},
            {"formatted_prompt",     result.formatted_prompt}
        };
        res.set_content(resp.dump(), "application/json");

        std::cout << "[api] POST /session/chat session=" << session_id
                  << " messages=" << messages.size()
                  << " prompt_tokens=" << result.n_tokens_prompt
                  << " generated=" << result.n_tokens_generated
                  << " prefill=" << result.ms_prefill << "ms"
                  << " decode=" << result.ms_decode << "ms"
                  << " total=" << ms << "ms\n";
    });

    // ── GET /model/info ──────────────────────────────────────────────────────
    //
    // Returns metadata about the model loaded for a session.
    // Query param: session_id
    // Response keys: arch, chat_template, n_params, n_ctx_train, model_path, tier
    srv.Get("/model/info", [&store](const httplib::Request& req,
                                    httplib::Response& res) {
        std::string session_id = req.has_param("session_id")
                               ? req.get_param_value("session_id")
                               : "";
        if (session_id.empty()) { send_error(res, 400, "session_id query param required"); return; }

        auto info = store.model_info(session_id);
        if (info.empty()) {
            send_error(res, 404, "session not found: " + session_id);
            return;
        }

        json resp = json::object();
        for (const auto& [k, v] : info) resp[k] = v;
        res.set_content(resp.dump(), "application/json");

        std::cout << "[api] GET /model/info session=" << session_id << "\n";
    });

    // ── GET /sessions ────────────────────────────────────────────────────────
    srv.Get("/sessions", [&store](const httplib::Request& /*req*/,
                                   httplib::Response& res) {
        auto sessions = store.list_sessions();

        int hot_count  = 0;
        int warm_count = 0;
        int cold_count = 0;
        json arr = json::array();

        for (const auto& m : sessions) {
            arr.push_back(meta_to_json(m));
            switch (m.tier) {
                case Tier::HOT:  ++hot_count;  break;
                case Tier::WARM: ++warm_count; break;
                case Tier::COLD: ++cold_count; break;
            }
        }

        json resp = {
            {"sessions", arr},
            {"hot",      hot_count},
            {"warm",     warm_count},
            {"cold",     cold_count}
        };
        res.set_content(resp.dump(), "application/json");

        std::cout << "[api] GET /sessions hot=" << hot_count
                  << " warm=" << warm_count
                  << " cold=" << cold_count << "\n";
    });
}

} // namespace engram
