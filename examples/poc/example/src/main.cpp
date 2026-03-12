// NOLINTBEGIN

#include "asa-poc-miia/client/inference_client.hpp"
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <string>
#include <iostream>
#include <map>
#include <algorithm>

// ============================================
// CONFIGURAÇÃO DE MODELOS - HARDCODED
// ============================================

struct ModelConfig {
    std::string id;
    std::string path;
    std::vector<int> input_shape;  // Formato: [batch, channels, height, width] ou [batch, features]
    std::string input_name;
};

const std::vector<ModelConfig> AVAILABLE_MODELS = {
    // Modelos simples de teste
    {"simple_linear",      "./models/simple_linear.onnx",      {1, 5},              "input"},
    {"simple_classifier",  "./models/simple_classifier.onnx",  {1, 4},              "input"},
    
    // Modelos de visão computacional (ImageNet) - Todos usam 224x224 RGB
    {"resnet18",           "./models/resnet18.onnx",           {1, 3, 224, 224},    "input"},
    {"mobilenet_v2",       "./models/mobilenet_v2.onnx",       {1, 3, 224, 224},    "input"},
    {"squeezenet",         "./models/squeezenet1_0.onnx",      {1, 3, 224, 224},    "input"},
};

// ============================================
// FUNÇÕES AUXILIARES
// ============================================

// Gerar dados dummy baseado no shape
std::vector<float> generate_dummy_input(const std::vector<int>& shape) {
    size_t total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }
    
    std::vector<float> data(total_size);
    
    // Gerar dados aleatórios normalizados (0.0 a 1.0)
    // Para modelos de visão, isso simula pixels de imagem normalizados
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    
    return data;
}

void print_model_info(const ModelConfig& config) {
    std::cout << "Model Configuration:" << std::endl;
    std::cout << "  ID: " << config.id << std::endl;
    std::cout << "  Path: " << config.path << std::endl;
    std::cout << "  Input name: " << config.input_name << std::endl;
    std::cout << "  Input shape: [";
    for (size_t i = 0; i < config.input_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << config.input_shape[i];
    }
    std::cout << "]" << std::endl;
    
    // Calcular tamanho total
    size_t total = 1;
    for (int dim : config.input_shape) {
        total *= dim;
    }
    std::cout << "  Total elements: " << total << std::endl;
}

void print_separator() {
    std::cout << "========================================" << std::endl;
}

void print_vector(const std::string& name, const std::vector<float>& data) {
    std::cout << name << ": [";
    for (size_t i = 0; i < data.size() && i < 10; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << data[i];
    }
    if (data.size() > 10) {
        std::cout << ", ... (" << data.size() << " total)";
    }
    std::cout << "]" << std::endl;
}

// ============================================
// MAIN
// ============================================

int main(int argc, char** argv) {

    std::cout << "==================================" << std::endl;
    std::cout << "ML Inference Client Example" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    std::string server_address = "localhost:50052";
    
    if (argc > 1) {
        server_address = argv[1];
    }
    
    // ============================================
    // ESCOLHER O MODELO AQUI - Mudar o índice
    // ============================================
    // 0 = simple_linear       (1x5)
    // 1 = simple_classifier   (1x4)
    // 2 = resnet18           (1x3x224x224)
    // 3 = mobilenet_v2       (1x3x224x224)
    // 4 = squeezenet         (1x3x224x224)
    
    const int SELECTED_MODEL_INDEX = 2;  // <-- MODIFICAR ESTE NÚMERO
    
    // ============================================
    
    if (SELECTED_MODEL_INDEX < 0 || SELECTED_MODEL_INDEX >= static_cast<int>(AVAILABLE_MODELS.size())) {
        std::cerr << "Invalid model index: " << SELECTED_MODEL_INDEX << std::endl;
        std::cerr << "Available models (0-" << (AVAILABLE_MODELS.size() - 1) << "):" << std::endl;
        for (size_t i = 0; i < AVAILABLE_MODELS.size(); ++i) {
            std::cerr << "  [" << i << "] " << AVAILABLE_MODELS[i].id << std::endl;
        }
        return 1;
    }
    
    const ModelConfig& selected_model = AVAILABLE_MODELS[SELECTED_MODEL_INDEX];
    
    std::cout << "Selected Model:" << std::endl;
    print_model_info(selected_model);
    std::cout << std::endl;
    
    std::cout << "Connecting to server: " << server_address << std::endl;
    
    // Create client
    mlinference::client::InferenceClient client(server_address);
    
    // Connect to server
    if (!client.connect()) {
        std::cerr << "Failed to connect to server" << std::endl;
        return 1;
    }
    
    print_separator();
    
    // Health check
    std::cout << "Performing health check..." << std::endl;
    if (client.health_check()) {
        std::cout << "✓ Server is healthy" << std::endl;
    } else {
        std::cout << "✗ Server health check failed" << std::endl;
    }
    
    print_separator();
    
    // Load the selected model
    std::cout << "Loading model: " << selected_model.id << std::endl;
    
    if (client.load_model(selected_model.id, selected_model.path)) {
        std::cout << "✓ Model loaded successfully" << std::endl;
    } else {
        std::cout << "✗ Failed to load model" << std::endl;
        return 1;
    }
    
    print_separator();
    
// List loaded models
    std::cout << "Listing loaded models..." << std::endl;
    auto models = client.list_models();  // era: list_loaded_models()
    std::cout << "Loaded models (" << models.size() << "):" << std::endl;
    for (const auto& model : models) {
        std::cout << "  - " << model.model_id  // era: model (string direto)
                  << " [" << model.backend << "]"
                  << (model.is_warmed_up ? " (warmed up)" : "")
                  << std::endl;
    }
    
    print_separator();
    
    // Single prediction with generated dummy input
    std::cout << "Testing single prediction..." << std::endl;
    std::cout << "Generating dummy input data..." << std::endl;
    
    // Gerar dados dummy automaticamente baseado no shape
    std::vector<float> input_data = generate_dummy_input(selected_model.input_shape);
    
    std::map<std::string, std::vector<float>> inputs;
    inputs[selected_model.input_name] = input_data;
    
    std::cout << "Input data generated: " << input_data.size() << " elements" << std::endl;
    print_vector("Input (first 10 elements)", input_data);
    
    auto result = client.predict(selected_model.id, inputs);
    
    if (result.success) {
        std::cout << "✓ Prediction successful" << std::endl;
        std::cout << "  Inference time: " << result.inference_time_ms << " ms" << std::endl;
        
        for (const auto& output_pair : result.outputs) {
            print_vector("Output (" + output_pair.first + ")", output_pair.second);
        }
    } else {
        std::cout << "✗ Prediction failed: " << result.error_message << std::endl;
    }
    
    print_separator();
    
    // Batch prediction
    std::cout << "Testing batch prediction (3 samples)..." << std::endl;
    
    std::vector<std::map<std::string, std::vector<float>>> batch_inputs;
    
    for (int i = 0; i < 3; ++i) {
        std::map<std::string, std::vector<float>> batch_input;
        std::vector<float> data = generate_dummy_input(selected_model.input_shape);
        batch_input[selected_model.input_name] = data;
        batch_inputs.push_back(batch_input);
    }
    
    auto batch_results = client.batch_predict(selected_model.id, batch_inputs);
    
    std::cout << "Batch results: " << batch_results.size() << " predictions" << std::endl;
    for (size_t i = 0; i < batch_results.size(); ++i) {
        if (batch_results[i].success) {
            std::cout << "  [" << i << "] ✓ " << batch_results[i].inference_time_ms << " ms";
            
            // Para modelos de classificação, mostrar classe predita
            if (!batch_results[i].outputs.empty()) {
                auto& first_output = batch_results[i].outputs.begin()->second;
                if (!first_output.empty()) {
                    auto max_it = std::max_element(first_output.begin(), first_output.end());
                    int predicted_class = std::distance(first_output.begin(), max_it);
                    float confidence = *max_it;
                    std::cout << " | Class: " << predicted_class << " (conf: " << confidence << ")";
                }
            }
            std::cout << std::endl;
        } else {
            std::cout << "  [" << i << "] ✗ " << batch_results[i].error_message << std::endl;
        }
    }
    
    print_separator();
    
    // Get worker status
    std::cout << "Getting worker status..." << std::endl;
    auto status = client.get_status();
    
    std::cout << "Worker ID: " << status.worker_id << std::endl;
    std::cout << "Uptime: " << status.uptime_seconds << " seconds" << std::endl;
    std::cout << "Total requests: " << status.total_requests << std::endl;
    std::cout << "Successful: " << status.successful_requests << std::endl;
    std::cout << "Failed: " << status.failed_requests << std::endl;
    std::cout << "Active: " << status.active_requests << std::endl;
    std::cout << "Loaded models: " << status.loaded_models.size() << std::endl;
    
    print_separator();
    
    // Stress test (optional)
    std::cout << "Running stress test (10 requests)..." << std::endl;
    
    auto stress_start = std::chrono::high_resolution_clock::now();
    
    int successful = 0;
    int failed = 0;
    
    for (int i = 0; i < 10; ++i) {
        auto test_result = client.predict(selected_model.id, inputs);
        if (test_result.success) {
            successful++;
        } else {
            failed++;
        }
    }
    
    auto stress_end = std::chrono::high_resolution_clock::now();
    auto stress_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        stress_end - stress_start
    );
    
    std::cout << "Stress test completed:" << std::endl;
    std::cout << "  Successful: " << successful << std::endl;
    std::cout << "  Failed: " << failed << std::endl;
    std::cout << "  Total time: " << stress_duration.count() << " ms" << std::endl;
    std::cout << "  Avg time: " << (stress_duration.count() / 10.0) << " ms per request" << std::endl;
    
    print_separator();
    
    // Unload model
    std::cout << "Unloading model: " << selected_model.id << std::endl;
    if (client.unload_model(selected_model.id)) {
        std::cout << "✓ Model unloaded successfully" << std::endl;
    } else {
        std::cout << "✗ Failed to unload model" << std::endl;
    }
    
    print_separator();
    
    std::cout << "Client example completed!" << std::endl;
    std::cout << std::endl;
    std::cout << "To test with a different model, change SELECTED_MODEL_INDEX:" << std::endl;
    std::cout << "  0 = simple_linear (1x5)" << std::endl;
    std::cout << "  1 = simple_classifier (1x4)" << std::endl;
    std::cout << "  2 = resnet18 (1x3x224x224)" << std::endl;
    std::cout << "  3 = mobilenet_v2 (1x3x224x224)" << std::endl;
    std::cout << "  4 = squeezenet (1x3x224x224)" << std::endl;
    
    return 0;
}

// NOLINTEND