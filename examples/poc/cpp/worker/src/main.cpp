// =============================================================================
// main.cpp — AsaMiia Worker entrypoint
// =============================================================================

#include "worker/worker_server.hpp"
#include "worker/s3_model_source.hpp"

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static std::unique_ptr<mlinference::worker::WorkerServer> g_server;

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
        << "  --help                Exibir esta ajuda\n"
        << "\n"
        << "Opções S3 (todas opcionais — S3 desabilitado se --s3-bucket não for fornecido):\n"
        << "  --s3-bucket  <name>   Nome do bucket S3\n"
        << "  --s3-prefix  <pfx>    Prefixo / pasta dentro do bucket (default: \"\")\n"
        << "  --s3-region  <reg>    Região AWS                  (default: us-east-1)\n"
        << "  --s3-cache   <path>   Diretório local de cache    (default: /tmp/asasamiia_s3_cache)\n"
        << "  --s3-endpoint <url>   Endpoint customizado        (ex: http://localhost:9000)\n"
        << "  --s3-path-style       Usar path-style addressing  (MinIO/LocalStack)\n"
        << "\n"
        << "Credenciais AWS (lidas automaticamente pelo SDK):\n"
        << "  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN\n"
        << "  ou perfis em ~/.aws/credentials / IAM Instance Profile\n";
}

int main(int argc, char* argv[]) {
    // ---- Defaults ----
    std::string address    = "0.0.0.0:50052";
    std::string worker_id  = "worker-1";
    std::string models_dir = "./models";
    uint32_t    num_threads = 4;
    bool        enable_gpu  = false;

    // S3
    mlinference::worker::S3Config s3cfg;
    bool s3_enabled = false;

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

        // --- S3 ---
        } else if (arg == "--s3-bucket" && i + 1 < argc) {
            s3cfg.bucket = argv[++i];
            s3_enabled   = true;
        } else if (arg == "--s3-prefix" && i + 1 < argc) {
            s3cfg.prefix = argv[++i];
        } else if (arg == "--s3-region" && i + 1 < argc) {
            s3cfg.region = argv[++i];
        } else if (arg == "--s3-cache" && i + 1 < argc) {
            s3cfg.cache_dir = argv[++i];
        } else if (arg == "--s3-endpoint" && i + 1 < argc) {
            s3cfg.endpoint_url = argv[++i];
        } else if (arg == "--s3-path-style") {
            s3cfg.use_path_style = true;

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

    // ---- Construir S3ModelSource (opcional) ----
    std::shared_ptr<mlinference::worker::S3ModelSource> s3_source;
    int s3_model_count = 0;

    if (s3_enabled) {
        try {
            s3_source = std::make_shared<mlinference::worker::S3ModelSource>(s3cfg);
            // Faz uma listagem prévia apenas para mostrar no banner
            // (sem download — só metadados)
            auto entries = s3_source->list();
            s3_model_count = static_cast<int>(entries.size());
        } catch (const std::exception& ex) {
            std::cerr << "ERRO ao inicializar fonte S3: " << ex.what() << std::endl;
            std::cerr << "Continuando sem integração S3." << std::endl;
            s3_source.reset();
            s3_enabled = false;
        }
    }

    // ---- Banner ----
    std::cout
        << "========================================\n"
        << "  AsaMiia Worker\n"
        << "========================================\n"
        << "  Worker ID  : " << worker_id  << "\n"
        << "  Address    : " << address    << "\n"
        << "  Threads    : " << num_threads << "\n"
        << "  GPU        : " << (enable_gpu ? "enabled" : "disabled") << "\n"
        << "  Models dir : " << models_abs  << "\n"
        << "  Local models : " << model_count << " arquivo(s)\n";

    if (s3_enabled && s3_source) {
        std::cout
            << "  S3 bucket  : s3://" << s3cfg.bucket << "/" << s3cfg.prefix << "\n"
            << "  S3 region  : " << s3cfg.region << "\n"
            << "  S3 cache   : " << s3cfg.cache_dir << "\n";
        if (!s3cfg.endpoint_url.empty())
            std::cout << "  S3 endpoint: " << s3cfg.endpoint_url << "\n";
        std::cout
            << "  S3 models  : " << s3_model_count
            << " objeto(s) disponível(is) (download on-demand)\n";
    } else {
        std::cout << "  S3         : desabilitado\n";
    }

    std::cout << "========================================" << std::endl;

    // ---- Start ----
    g_server = std::make_unique<mlinference::worker::WorkerServer>(
        worker_id, address, enable_gpu, num_threads, models_dir,
        std::move(s3_source));

    g_server->run();  // Blocking

    std::cout << "Worker stopped." << std::endl;
    return 0;
}


// #include "worker/worker_server.hpp"
// #include <iostream>
// #include <string>
// #include <cstdlib>
// #include <csignal>
// #include <filesystem>

// static std::unique_ptr<mlinference::worker::WorkerServer> g_server;

// void signal_handler(int signal) {
//     std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
//     if (g_server) {
//         g_server->stop();
//     }
// }

// void print_usage(const char* program) {
//     std::cout
//         << "Usage: " << program << " [options]\n"
//         << "\n"
//         << "Options:\n"
//         << "  --address    <addr>   Listen address        (default: 0.0.0.0:50052)\n"
//         << "  --worker-id  <id>     Worker identifier     (default: worker-1)\n"
//         << "  --threads    <n>      Inference threads      (default: 4)\n"
//         << "  --models-dir <path>   Models directory       (default: ./models)\n"
//         << "  --gpu                 Enable GPU inference\n"
//         << "  --help                Show this help\n";
// }

// int main(int argc, char* argv[]) {
//     std::string address    = "0.0.0.0:50052";
//     std::string worker_id  = "worker-1";
//     std::string models_dir = "./models";
//     uint32_t num_threads   = 4;
//     bool enable_gpu        = false;

//     // ---- Parse CLI ----
//     for (int i = 1; i < argc; ++i) {
//         std::string arg = argv[i];

//         if (arg == "--address" && i + 1 < argc) {
//             address = argv[++i];
//         } else if (arg == "--worker-id" && i + 1 < argc) {
//             worker_id = argv[++i];
//         } else if (arg == "--threads" && i + 1 < argc) {
//             num_threads = static_cast<uint32_t>(std::atoi(argv[++i]));
//         } else if (arg == "--models-dir" && i + 1 < argc) {
//             models_dir = argv[++i];
//         } else if (arg == "--gpu") {
//             enable_gpu = true;
//         } else if (arg == "--help" || arg == "-h") {
//             print_usage(argv[0]);
//             return 0;
//         } else {
//             std::cerr << "Unknown argument: " << arg << std::endl;
//             print_usage(argv[0]);
//             return 1;
//         }
//     }

//     // ---- Validate models directory ----
//     namespace fs = std::filesystem;
//     if (!fs::exists(models_dir)) {
//         std::cerr << "WARNING: models directory does not exist: " << models_dir << std::endl;
//         std::cerr << "         Creating it..." << std::endl;
//         fs::create_directories(models_dir);
//     }

//     // Resolve to absolute for unambiguous logging
//     std::string models_abs = fs::canonical(models_dir).string();

//     // Count model files
//     int model_count = 0;
//     for (const auto& entry : fs::directory_iterator(models_dir)) {
//         if (entry.is_regular_file()) {
//             std::string ext = entry.path().extension().string();
//             if (ext == ".onnx" || ext == ".py") model_count++;
//         }
//     }

//     // ---- Signal handlers ----
//     std::signal(SIGINT, signal_handler);
//     std::signal(SIGTERM, signal_handler);

//     // ---- Banner ----
//     std::cout << "========================================\n"
//               << "  AsaMiia Worker\n"
//               << "========================================\n"
//               << "  Worker ID  : " << worker_id << "\n"
//               << "  Address    : " << address << "\n"
//               << "  Threads    : " << num_threads << "\n"
//               << "  GPU        : " << (enable_gpu ? "enabled" : "disabled") << "\n"
//               << "  Models dir : " << models_abs << "\n"
//               << "  Models     : " << model_count << " file(s) found\n"
//               << "========================================" << std::endl;

//     // ---- Start ----
//     g_server = std::make_unique<mlinference::worker::WorkerServer>(
//         worker_id, address, enable_gpu, num_threads, models_dir);

//     g_server->run();  // Blocking

//     std::cout << "Worker stopped." << std::endl;
//     return 0;
// }