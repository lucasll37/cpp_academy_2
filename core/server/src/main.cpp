// =============================================================================
// main.cpp — Miia Server entrypoint
// =============================================================================

#include "server/worker_server.hpp"

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;
using mlinference::server::WorkerServer;

static std::unique_ptr<WorkerServer> g_server;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    if (g_server)
        g_server->stop();
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "\n"
        << "Options gerais:\n"
        << "  --address    <addr>   Listen address              (default: 0.0.0.0:50052)\n"
        << "  --worker-id  <id>     Worker identifier           (default: worker-1)\n"
        << "  --threads    <n>      Inference threads           (default: 4)\n"
        << "  --models-dir <path>   Diretório local de modelos  (default: ./models)\n"
        << "  --gpu                 Habilitar inferência GPU\n"
        << "  --help                Exibir esta ajuda\n";
}

int main(int argc, char* argv[]) {
    // ---- Defaults ----
    std::string address    = "0.0.0.0:50052";
    std::string worker_id  = "worker-1";
    std::string models_dir = "../models";
    uint32_t    num_threads = 4;
    bool        enable_gpu  = false;

    // ---- Parse CLI ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // --- Gerais ---
        if (arg == "--address" && i + 1 < argc) {
            address = argv[++i];
        } else if (arg == "--worker-id" && i + 1 < argc) {
            worker_id = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--models-dir" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--gpu") {
            enable_gpu = true;

        // --- Ajuda / erro ---
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Argumento desconhecido: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // ---- Diretório de modelos locais ----
    if (!fs::exists(models_dir)) {
        std::cerr << "WARNING: models directory does not exist: "
                  << models_dir << std::endl;
        std::cerr << "         Creating it..." << std::endl;
        fs::create_directories(models_dir);
    }

    std::string models_abs = fs::canonical(models_dir).string();

    // Injeta o caminho absoluto do .venv dentro de models_dir para o PythonBackend.
    // overwrite=0: respeita MIIA_VENV se já estiver setado pelo usuário.
    {
        std::string venv_path = models_abs + "/.venv";
        setenv("MIIA_VENV", venv_path.c_str(), /*overwrite=*/0); // debug?
    }

    int model_count = 0;
    for (const auto& entry : fs::directory_iterator(models_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".onnx" || ext == ".py") model_count++;
        }
    }

    // ---- Signal handlers ----
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ---- Banner ----
    std::cout
        << "========================================\n"
        << "  Miia Server\n"
        << "========================================\n"
        << "  Worker ID  : " << worker_id  << "\n"
        << "  Address    : " << address    << "\n"
        << "  Threads    : " << num_threads << "\n"
        << "  GPU        : " << (enable_gpu ? "enabled" : "disabled") << "\n"
        << "  Models dir : " << models_abs  << "\n"
        << "  Local models : " << model_count << " arquivo(s)\n";


    std::cout << "========================================" << std::endl;

    // ---- Start ----
    g_server = std::make_unique<WorkerServer>(
        worker_id, address, enable_gpu, num_threads, models_dir);

    g_server->run();  // Blocking

    std::cout << "[Miia Server] Server stopped" << std::endl;
    return 0;
}