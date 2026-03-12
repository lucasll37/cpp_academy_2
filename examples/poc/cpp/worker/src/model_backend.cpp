// =============================================================================
// model_backend.cpp — RuntimeMetrics percentile helpers + default warmup
// =============================================================================

#include "worker/model_backend.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <iostream>

namespace mlinference {
namespace worker {

// ============================================
// RuntimeMetrics — percentile helpers
// ============================================

/// Nearest-rank percentile on a sorted copy of `samples`.
/// Returns 0.0 for empty input.  The result is always a real observed
/// value, so it is never negative.
static double percentile(const std::vector<double>& samples, double pct) {
    if (samples.empty()) return 0.0;

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    // Nearest-rank formula: index = ceil(pct * n) - 1  (0-based)
    size_t n   = sorted.size();
    size_t idx = static_cast<size_t>(std::ceil(pct * static_cast<double>(n)));
    if (idx > 0) idx--;            // convert to 0-based
    if (idx >= n) idx = n - 1;    // clamp (handles pct == 1.0)

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

    // Build dummy inputs from schema
    std::map<std::string, std::vector<float>> dummy_inputs;
    std::mt19937 rng(42);  // Deterministic seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (const auto& spec : schema.inputs) {
        // Calculate total elements, treating -1 as 1
        int64_t total = 1;
        for (int64_t dim : spec.shape) {
            total *= (dim == -1) ? 1 : dim;
        }

        std::vector<float> data(static_cast<size_t>(total));
        for (auto& v : data) v = dist(rng);
        dummy_inputs[spec.name] = std::move(data);
    }

    // Run N warmup inferences (discard results)
    for (uint32_t i = 0; i < n; ++i) {
        predict(dummy_inputs);
    }
}

}  // namespace worker
}  // namespace mlinference