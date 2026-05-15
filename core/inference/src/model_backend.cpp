// =============================================================================
// model_backend.cpp — RuntimeMetrics percentile helpers + default warmup
// =============================================================================

#include "inference/model_backend.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <iostream>

namespace miia {
namespace inference {

// ============================================
// RuntimeMetrics — percentile helpers
// ============================================

static double percentile(const std::vector<double>& samples, double pct) {
    if (samples.empty()) return 0.0;

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    size_t n   = sorted.size();
    size_t idx = static_cast<size_t>(std::ceil(pct * static_cast<double>(n)));
    if (idx > 0) idx--;
    if (idx >= n) idx = n - 1;

    return sorted[idx];
}

double RuntimeMetrics::p95_time_ms() const {
    return percentile(latency_samples, 0.95);
}

double RuntimeMetrics::p99_time_ms() const {
    return percentile(latency_samples, 0.99);
}

// ============================================
// ModelBackend default warmup
// ============================================

void ModelBackend::warmup(uint32_t n) {
    if (!loaded_) return;

    auto schema = get_schema();

    // Build dummy inputs as Object — each tensor key maps to an Array of floats.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    client::Object dummy_inputs;
    for (const auto& spec : schema.inputs) {
        int64_t total = 1;
        for (int64_t dim : spec.shape)
            total *= (dim == -1) ? 1 : dim;

        client::Array arr;
        arr.reserve(static_cast<size_t>(total));
        for (int64_t i = 0; i < total; ++i)
            arr.push_back(client::Value{static_cast<double>(dist(rng))});

        dummy_inputs[spec.name] = client::Value{std::move(arr)};
    }

    for (uint32_t i = 0; i < n; ++i)
        predict(dummy_inputs);
}

}  // namespace inference
}  // namespace miia
