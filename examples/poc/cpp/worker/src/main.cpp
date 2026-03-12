#include "worker/worker_server.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <csignal>
#include <filesystem>

static std::unique_ptr<mlinference::worker::WorkerServer> g_server;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --address    <addr>   Listen address        (default: 0.0.0.0:50052)\n"
        << "  --worker-id  <id>     Worker identifier     (default: worker-1)\n"
        << "  --threads    <n>      Inference threads      (default: 4)\n"
        << "  --models-dir <path>   Models directory       (default: ./models)\n"
        << "  --gpu                 Enable GPU inference\n"
        << "  --help                Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string address    = "0.0.0.0:50052";
    std::string worker_id  = "worker-1";
    std::string models_dir = "./models";
    uint32_t num_threads   = 4;
    bool enable_gpu        = false;

    // ---- Parse CLI ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

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
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // ---- Validate models directory ----
    namespace fs = std::filesystem;
    if (!fs::exists(models_dir)) {
        std::cerr << "WARNING: models directory does not exist: " << models_dir << std::endl;
        std::cerr << "         Creating it..." << std::endl;
        fs::create_directories(models_dir);
    }

    // Resolve to absolute for unambiguous logging
    std::string models_abs = fs::canonical(models_dir).string();

    // Count model files
    int model_count = 0;
    for (const auto& entry : fs::directory_iterator(models_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".onnx" || ext == ".py") model_count++;
        }
    }

    // ---- Signal handlers ----
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ---- Banner ----
    std::cout << "========================================\n"
              << "  AsaMiia Worker\n"
              << "========================================\n"
              << "  Worker ID  : " << worker_id << "\n"
              << "  Address    : " << address << "\n"
              << "  Threads    : " << num_threads << "\n"
              << "  GPU        : " << (enable_gpu ? "enabled" : "disabled") << "\n"
              << "  Models dir : " << models_abs << "\n"
              << "  Models     : " << model_count << " file(s) found\n"
              << "========================================" << std::endl;

    // ---- Start ----
    g_server = std::make_unique<mlinference::worker::WorkerServer>(
        worker_id, address, enable_gpu, num_threads, models_dir);

    g_server->run();  // Blocking

    std::cout << "Worker stopped." << std::endl;
    return 0;
}