#include "engram.h"
#include "session_store.h"
#include "api.h"

#include <httplib.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --host        <str>   Listen address           (default: 127.0.0.1)\n"
        << "  --port        <int>   Listen port              (default: 8080)\n"
        << "  --model       <str>   Default model path       (optional; used when /session/create omits 'model')\n"
        << "  --max-hot     <int>   Max HOT sessions         (default: 4)\n"
        << "  --max-warm    <int>   Max WARM sessions        (default: 16)\n"
        << "  --sessions-dir <str> Cold storage directory   (default: ./sessions)\n"
        << "  --threads     <int>   llama.cpp threads        (default: 4)\n"
        << "  --gpu-layers  <int>   GPU layers               (default: 0)\n"
        << "  --help                Print this message\n";
}

static engram::Config parse_args(int argc, char* argv[]) {
    engram::Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto next_val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--host") {
            cfg.host = next_val();
        } else if (arg == "--port") {
            cfg.port = std::stoi(next_val());
        } else if (arg == "--model") {
            cfg.default_model_path = next_val();
        } else if (arg == "--max-hot") {
            cfg.max_hot_sessions = std::stoi(next_val());
        } else if (arg == "--max-warm") {
            cfg.max_warm_sessions = std::stoi(next_val());
        } else if (arg == "--sessions-dir") {
            cfg.cold_storage_path = next_val();
        } else if (arg == "--threads") {
            cfg.n_threads = std::stoi(next_val());
        } else if (arg == "--gpu-layers") {
            cfg.n_gpu_layers = std::stoi(next_val());
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Banner
// ─────────────────────────────────────────────────────────────────────────────

static void print_banner(const engram::Config& cfg) {
    std::cout << R"(
  ______                                  
 |  ____|                                 
 | |__   _ __   __ _ _ __ __ _ _ __ ___  
 |  __| | '_ \ / _` | '__/ _` | '_ ` _ \ 
 | |____| | | | (_| | | | (_| | | | | | |
 |______|_| |_|\__, |_|  \__,_|_| |_| |_|
                __/ |                     
               |___/   KV-cache sessions  
)" << "\n";

    std::cout << "  Tiered KV-cache inference server\n\n";
    std::cout << "  Config:\n"
              << "    listen         : " << cfg.host << ":" << cfg.port << "\n"
              << "    default_model  : " << (cfg.default_model_path.empty() ? "(none)" : cfg.default_model_path) << "\n"
              << "    max_hot        : " << cfg.max_hot_sessions  << " sessions\n"
              << "    max_warm       : " << cfg.max_warm_sessions << " sessions\n"
              << "    cold_storage   : " << cfg.cold_storage_path << "\n"
              << "    threads        : " << cfg.n_threads << "\n"
              << "    gpu_layers     : " << cfg.n_gpu_layers << "\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    engram::Config cfg;
    try {
        cfg = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    print_banner(cfg);

    // Build session store.
    engram::SessionStore store(cfg);

    // Build HTTP server.
    httplib::Server srv;

    // Register all API routes.
    engram::register_routes(srv, store);

    std::cout << "[engram] Listening on " << cfg.host << ":" << cfg.port << "\n\n";

    // Blocking call — returns when srv.stop() is called or SIGINT.
    if (!srv.listen(cfg.host.c_str(), cfg.port)) {
        std::cerr << "[engram] Failed to start HTTP server on "
                  << cfg.host << ":" << cfg.port << "\n";
        return 1;
    }

    return 0;
}
