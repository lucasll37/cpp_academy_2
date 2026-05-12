// NOLINTBEGIN
// =============================================================================
// main.cpp — Miia Interactive Client (TUI)
//
// Uso:
//   MiiaClient [endereço]          ex: MiiaClient localhost:50052
//   MiiaClient inprocess           modo in-process (sem worker separado)
// =============================================================================

#include "client/inference_client.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>   // para poll_key com select()

using Client     = mlinference::client::InferenceClient;
using ModelInfo  = mlinference::client::ModelInfo;
using TensorSpec = mlinference::client::ModelInfo::TensorSpec;
using Available  = mlinference::client::AvailableModel;
using Object     = mlinference::client::Object;
using Array      = mlinference::client::Array;
using Value      = mlinference::client::Value;

// =============================================================================
// ANSI & Terminal helpers
// =============================================================================

namespace c {
    const char* RST         = "\033[0m";
    const char* B           = "\033[1m";
    const char* DIM         = "\033[2m";
    const char* R           = "\033[31m";
    const char* G           = "\033[32m";
    const char* Y           = "\033[33m";
    const char* BL          = "\033[34m";
    const char* MG          = "\033[35m";
    const char* CY          = "\033[36m";
    const char* CLR         = "\033[2J\033[H";
    const char* ALT_ON      = "\033[?1049h";
    const char* ALT_OFF     = "\033[?1049l";
    const char* HIDE_CURSOR = "\033[?25l";
    const char* SHOW_CURSOR = "\033[?25h";
}

enum class Key { UP, DOWN, LEFT, RIGHT, ENTER, BACK, ESC, CHAR, UNKNOWN };

struct KeyEvent {
    Key  key = Key::UNKNOWN;
    char ch  = 0;
};

KeyEvent read_key() {
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt        = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    newt.c_cc[VMIN]  = 1;
    newt.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    KeyEvent ev;
    int ch = getchar();

    if (ch == 27) {
        newt.c_cc[VMIN]  = 0;
        newt.c_cc[VTIME] = 1;
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        int c2 = getchar();
        if (c2 == '[') {
            int c3 = getchar();
            switch (c3) {
                case 'A': ev.key = Key::UP;    break;
                case 'B': ev.key = Key::DOWN;  break;
                case 'C': ev.key = Key::RIGHT; break;
                case 'D': ev.key = Key::LEFT;  break;
                default:  ev.key = Key::UNKNOWN; break;
            }
        } else {
            ev.key = Key::ESC;
        }
    } else if (ch == '\n' || ch == '\r') {
        ev.key = Key::ENTER;
    } else if (ch == 127 || ch == 8) {
        ev.key = Key::BACK;
    } else if (ch == 'q' || ch == 'Q') {
        ev.key = Key::ESC;
    } else {
        ev.key = Key::CHAR;
        ev.ch  = static_cast<char>(ch);
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ev;
}

// -----------------------------------------------------------------------------
// poll_key — aguarda tecla por até timeout_ms usando select().
// Retorna UNKNOWN se o timeout expirar sem entrada.
// -----------------------------------------------------------------------------
KeyEvent poll_key(int timeout_ms) {
    // Coloca stdin em modo raw/não-canônico para leitura imediata
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt           = oldt;
    newt.c_lflag  &= ~(ICANON | ECHO);
    newt.c_cc[VMIN]  = 0;
    newt.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    // Usa select() para aguardar dados com timeout preciso
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);

    struct timeval tv;
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int ready = select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv);

    KeyEvent ev;  // key = UNKNOWN por padrão

    if (ready > 0) {
        int ch = getchar();
        if (ch != EOF && ch != -1) {
            if (ch == 'q' || ch == 'Q' || ch == 27) ev.key = Key::ESC;
            else if (ch == '\n' || ch == '\r')       ev.key = Key::ENTER;
            else { ev.key = Key::CHAR; ev.ch = static_cast<char>(ch); }
        }
    }
    // Se ready == 0 → timeout; ev.key permanece UNKNOWN

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ev;
}

// =============================================================================
// Alternate-screen RAII guard
// =============================================================================

struct TerminalGuard {
    TerminalGuard() {
        std::cout << c::ALT_ON << c::HIDE_CURSOR << std::flush;
    }
    ~TerminalGuard() {
        std::cout << c::ALT_OFF << c::SHOW_CURSOR << std::flush;
    }
};

// =============================================================================
// Drawing primitives
// =============================================================================

void clear() { std::cout << c::CLR << std::flush; }
void wait_key() {
    std::cout << c::DIM << "  Press any key..." << c::RST << std::flush;
    read_key();
}

void draw_header(const std::string& title,
                 const std::string& subtitle = "") {
    std::cout << "\n"
              << " " << c::BL << "┌────────────────────────────────────────────────" << c::RST
              << "\n"
              << " " << c::BL << "│ " << c::RST << c::B << c::CY << title << c::RST << "\n";
    if (!subtitle.empty())
        std::cout << " " << c::BL << "│ " << c::RST << c::DIM << subtitle << c::RST << "\n";
    std::cout << " " << c::BL << "└────────────────────────────────────────────────" << c::RST << "\n\n";
}

void draw_footer() {
    std::cout << "\n " << c::DIM
              << "────────────────────────────────────────────────"
              << c::RST << "\n";
}

void ui_ok(const std::string& msg) {
    std::cout << "    " << c::G << "✓ " << c::RST << msg << "\n";
}
void ui_err(const std::string& msg) {
    std::cout << "    " << c::R << "✗ " << c::RST << msg << "\n";
}
void ui_wrn(const std::string& msg) {
    std::cout << "    " << c::Y << "⚠ " << c::RST << msg << "\n";
}
void ui_inf(const std::string& msg) {
    std::cout << "    " << c::BL << "· " << c::RST << msg << "\n";
}

std::string ask_text(const std::string& prompt) {
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt        = oldt;
    newt.c_lflag |= (ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    std::cout << "    " << c::Y << prompt << c::RST;
    std::string val;
    std::getline(std::cin, val);

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return val;
}

int ask_int(const std::string& prompt, int default_val) {
    std::string s = ask_text(prompt);
    if (s.empty()) return default_val;
    try { return std::stoi(s); } catch (...) { return default_val; }
}

// =============================================================================
// Arrow-key list selector
// =============================================================================

int list_selector(const std::string& title,
                  const std::vector<std::string>& items,
                  const std::string& footer_hint = "↑↓ navigate  ENTER select  Q quit") {
    int sel = 0;
    int n   = static_cast<int>(items.size());
    if (n == 0) return -1;

    while (true) {
        clear();
        draw_header(title);

        for (int i = 0; i < n; ++i) {
            if (i == sel)
                std::cout << "  " << c::B << c::CY << "▶ " << items[i] << c::RST << "\n";
            else
                std::cout << "    " << c::DIM << items[i] << c::RST << "\n";
        }

        std::cout << "\n " << c::DIM << footer_hint << c::RST << "\n";

        auto ev = read_key();
        switch (ev.key) {
            case Key::UP:    sel = (sel - 1 + n) % n; break;
            case Key::DOWN:  sel = (sel + 1) % n;     break;
            case Key::ENTER: return sel;
            case Key::BACK:
            case Key::ESC:   return -1;
            default: break;
        }
    }
}

// =============================================================================
// Model card / spec helpers
// =============================================================================

void print_spec(const TensorSpec& s, const std::string& prefix) {
    std::cout << prefix << c::B << s.name << c::RST
              << "  [" << s.dtype << "]";
    if (s.structured) {
        std::cout << "  (structured)";
    } else {
        std::cout << "  shape=(";
        for (size_t i = 0; i < s.shape.size(); ++i) {
            if (i) std::cout << ", ";
            if (s.shape[i] == -1) std::cout << "?"; else std::cout << s.shape[i];
        }
        std::cout << ")";
    }
    std::cout << "\n";
    if (!s.description.empty())
        std::cout << prefix << "  " << c::DIM << s.description << c::RST << "\n";
}

void print_card(const ModelInfo& m) {
    std::cout << "\n";
    printf("    %-16s %s\n", "ID:",      m.model_id.c_str());
    printf("    %-16s %s\n", "Backend:", m.backend.c_str());
    printf("    %-16s %s\n", "Version:", m.version.c_str());
    if (!m.description.empty())
        printf("    %-16s %s\n", "Description:", m.description.c_str());
    if (!m.author.empty())
        printf("    %-16s %s\n", "Author:", m.author.c_str());
    if (m.memory_usage_bytes > 0)
        printf("    %-16s %.1f MB\n", "Memory:", m.memory_usage_bytes / 1048576.0);
    printf("    %-16s %s\n", "Warmed up:", m.is_warmed_up ? "yes" : "no");

    if (!m.inputs.empty()) {
        std::cout << "\n    " << c::B << "Inputs:" << c::RST << "\n";
        for (const auto& s : m.inputs) print_spec(s, "      ");
    }
    if (!m.outputs.empty()) {
        std::cout << "\n    " << c::B << "Outputs:" << c::RST << "\n";
        for (const auto& s : m.outputs) print_spec(s, "      ");
    }
    std::cout << "\n";
}

// =============================================================================
// Dummy input generator
//
// Constrói um Object a partir do schema do modelo:
//   - inputs estruturados (structured=true) → Object vazio
//     (o modelo deve tratar inputs ausentes/incompletos graciosamente)
//   - demais tensores → Array de doubles aleatórios em [0, 1]
//     (dimensões dinâmicas -1 são resolvidas como 1)
// =============================================================================

Object make_inputs(const ModelInfo& info) {
    Object out;
    for (const auto& spec : info.inputs) {
        if (spec.structured) {
            // Input estruturado: envia Object vazio como placeholder.
            // O modelo (ex: tutorial_model.py) deve lidar com state={} retornando
            // um valor default, evitando o crash de chamar .get() em uma list.
            out[spec.name] = Value{Object{}};
        } else {
            int64_t total = 1;
            for (int64_t d : spec.shape) total *= (d <= 0) ? 1 : d;

            Array arr;
            arr.reserve(static_cast<size_t>(total));
            for (int64_t i = 0; i < total; ++i)
                arr.push_back(Value{static_cast<double>(rand()) /
                                    static_cast<double>(RAND_MAX)});
            out[spec.name] = Value{std::move(arr)};
        }
    }
    return out;
}

// =============================================================================
// Performance statistics (client-side)
// =============================================================================

struct PerfStats {
    std::vector<double> inference;
    std::vector<double> rtt;
    uint32_t failures = 0;

    void add(bool ok, double inf_ms, double rtt_ms = 0) {
        if (ok) {
            inference.push_back(inf_ms);
            if (rtt_ms > 0) rtt.push_back(rtt_ms);
        } else {
            failures++;
        }
    }

    bool     empty() const { return inference.empty(); }
    uint32_t total() const { return static_cast<uint32_t>(inference.size()) + failures; }
    uint32_t ok()    const { return static_cast<uint32_t>(inference.size()); }

    static double vec_sum(const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0);
    }
    static double vec_avg(const std::vector<double>& v) {
        return v.empty() ? 0.0 : vec_sum(v) / v.size();
    }
    static double vec_min(const std::vector<double>& v) {
        return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
    }
    static double vec_max(const std::vector<double>& v) {
        return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
    }
    static double vec_stddev(const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        double avg = vec_avg(v);
        double sq  = 0.0;
        for (double x : v) sq += (x - avg) * (x - avg);
        return std::sqrt(sq / v.size());
    }
    static double vec_pct(std::vector<double> v, double p) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t idx = static_cast<size_t>(std::ceil(p * v.size())) - 1;
        return v[std::min(idx, v.size() - 1)];
    }

    void print_table(double wall_ms = 0.0) const {
        if (empty() && failures == 0) {
            ui_wrn("No data.");
            return;
        }

        printf("    Requests  : %u total, %u ok, %u failed\n", total(), ok(), failures);
        std::cout << "\n";

        printf("    %-22s %8s %8s %8s %8s %8s %8s\n",
               "Metric (ms)", "avg", "min", "max", "σ", "p50", "p99");
        std::cout << "    ";
        for (int i = 0; i < 76; ++i) std::cout << "─";
        std::cout << "\n";

        printf("    %-22s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
               "Server inference",
               vec_avg(inference), vec_min(inference), vec_max(inference),
               vec_stddev(inference),
               vec_pct(inference, 0.50), vec_pct(inference, 0.99));

        if (!rtt.empty()) {
            printf("    %-22s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
                   "Network RTT",
                   vec_avg(rtt), vec_min(rtt), vec_max(rtt),
                   vec_stddev(rtt),
                   vec_pct(rtt, 0.50), vec_pct(rtt, 0.99));
        }

        std::cout << "\n";

        // Throughput correto: req/s baseado no wall time total (fim a fim),
        // não na soma das latências individuais de inferência.
        if (wall_ms > 0) {
            printf("    Wall time  : %.1f ms (end-to-end)\n", wall_ms);
            printf("    Throughput : %.1f req/s (wall time)\n",
                   static_cast<double>(ok()) * 1000.0 / wall_ms);
        } else if (!inference.empty()) {
            // Fallback: sem wall time, usa soma das latências como aproximação
            // (válido apenas para execução sequencial sem paralelismo)
            double total_inf = vec_sum(inference);
            if (total_inf > 0)
                printf("    Throughput : %.1f req/s (seq. approx.)\n",
                       static_cast<double>(ok()) * 1000.0 / total_inf);
        }
    }
};

// Global session stats keyed by model_id (client-side only)
std::map<std::string, PerfStats> session_stats;

void record(const std::string& id, bool ok, double inf_ms, double rtt_ms = 0) {
    session_stats[id].add(ok, inf_ms, rtt_ms);
}

// =============================================================================
// Model selector via arrow keys
// =============================================================================

std::string pick_loaded_model(Client& client) {
    auto models = client.list_models();
    if (models.empty()) { ui_wrn("No models loaded."); return ""; }
    if (models.size() == 1) return models[0].model_id;

    std::vector<std::string> opts;
    for (const auto& m : models) {
        std::string label = m.model_id + "  [" + m.backend + "]";
        if (!m.description.empty()) label += "  " + m.description;
        opts.push_back(label);
    }

    int sel = list_selector("Select Loaded Model", opts);
    return (sel < 0) ? "" : models[sel].model_id;
}

// =============================================================================
// FLOW: Browse Available Models (server-side)
// =============================================================================

void flow_browse(Client& client, const std::string& directory) {
    auto available = client.list_available_models(directory);
    if (available.empty()) {
        clear();
        draw_header("Available Models");
        ui_wrn("No model files found on server.");
        draw_footer();
        wait_key();
        return;
    }

    std::vector<std::string> opts;
    for (const auto& m : available) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%-28s %7.1f KB  [%s]%s",
                 m.filename.c_str(),
                 m.file_size_bytes / 1024.0,
                 m.backend.c_str(),
                 m.is_loaded ? "  ● loaded" : "");
        opts.push_back(std::string(buf));
    }

    int sel = list_selector("Available Models (server directory)", opts,
                            "↑↓ navigate  ENTER load  Q back");
    if (sel < 0) return;

    const auto& chosen = available[sel];
    if (chosen.is_loaded) {
        clear();
        draw_header("Already Loaded");
        ui_inf("\"" + chosen.filename + "\" is already loaded as \"" + chosen.loaded_as + "\"");
        draw_footer();
        wait_key();
        return;
    }

    // Auto-derive model ID from filename
    std::string default_id = chosen.filename;
    auto dot = default_id.rfind('.');
    if (dot != std::string::npos) default_id = default_id.substr(0, dot);

    clear();
    draw_header("Load: " + chosen.filename);
    std::string id = ask_text("Model ID (enter = \"" + default_id + "\"): ");
    if (id.empty()) id = default_id;

    std::cout << "\n";
    ui_inf("Validating...");
    auto v = client.validate_model(chosen.path);
    if (!v.valid) {
        ui_err("Validation failed: " + v.error_message);
        draw_footer();
        wait_key();
        return;
    }
    ui_ok("Valid — backend: " + v.backend);
    for (const auto& s : v.inputs)  print_spec(s, "      IN  ");
    for (const auto& s : v.outputs) print_spec(s, "      OUT ");
    std::cout << "\n";

    ui_inf("Loading...");
    if (client.load_model(id, chosen.path))
        ui_ok("Loaded as \"" + id + "\"");
    else
        ui_err("Load failed.");

    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: List Loaded Models (with detail view)
// =============================================================================

void flow_list(Client& client) {
    auto models = client.list_models();
    if (models.empty()) {
        clear();
        draw_header("Loaded Models");
        ui_wrn("None loaded.");
        draw_footer();
        wait_key();
        return;
    }

    std::vector<std::string> opts;
    for (const auto& m : models) {
        std::string label = m.model_id + "  [" + m.backend + "]";
        if (!m.description.empty()) label += "  — " + m.description;
        opts.push_back(label);
    }

    int sel = list_selector("Loaded Models — select for details", opts);
    if (sel < 0) return;

    auto info = client.get_model_info(models[sel].model_id);
    clear();
    draw_header("Model Details: " + info.model_id);
    print_card(info);
    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: Load Model (manual path)
// =============================================================================

void flow_load(Client& client) {
    clear();
    draw_header("Load Model");

    std::string path = ask_text("Path (.onnx / .py): ");
    if (path.empty()) return;

    std::string id = ask_text("Model ID (enter = auto): ");
    if (id.empty()) {
        auto slash = path.rfind('/');
        std::string fname = (slash != std::string::npos) ? path.substr(slash + 1) : path;
        auto dot2 = fname.rfind('.');
        id = (dot2 != std::string::npos) ? fname.substr(0, dot2) : fname;
    }

    std::cout << "\n";
    ui_inf("Validating...");
    auto v = client.validate_model(path);
    if (!v.valid) {
        ui_err("Validation failed: " + v.error_message);
        draw_footer();
        wait_key();
        return;
    }
    ui_ok("Valid — backend: " + v.backend);
    for (const auto& s : v.inputs)  print_spec(s, "      IN  ");
    for (const auto& s : v.outputs) print_spec(s, "      OUT ");
    std::cout << "\n";

    ui_inf("Loading...");
    if (client.load_model(id, path))
        ui_ok("Loaded as \"" + id + "\"");
    else
        ui_err("Load failed.");

    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: Unload Model
// =============================================================================

void flow_unload(Client& client) {
    std::string id = pick_loaded_model(client);
    if (id.empty()) return;

    clear();
    draw_header("Unload: " + id);
    if (client.unload_model(id))
        ui_ok("Unloaded \"" + id + "\"");
    else
        ui_err("Failed to unload.");

    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: Validate Model (dry-run, sem carregar)
// =============================================================================

void flow_validate(Client& client) {
    clear();
    draw_header("Validate Model (dry-run)");

    std::string path = ask_text("Path: ");
    if (path.empty()) return;

    std::cout << "\n";
    auto v = client.validate_model(path);
    if (v.valid) {
        ui_ok("Valid — backend: " + v.backend);
        for (const auto& s : v.inputs)  print_spec(s, "    IN  ");
        for (const auto& s : v.outputs) print_spec(s, "    OUT ");
    } else {
        ui_err("Invalid: " + v.error_message);
    }
    for (const auto& w : v.warnings) ui_wrn(w);

    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: Single Prediction
// =============================================================================

void flow_predict(Client& client) {
    std::string id = pick_loaded_model(client);
    if (id.empty()) return;

    clear();
    draw_header("Single Prediction: " + id);

    auto info   = client.get_model_info(id);
    auto inputs = make_inputs(info);
    if (inputs.empty()) { ui_err("Schema has no inputs."); draw_footer(); wait_key(); return; }

    ui_inf("Running...");
    auto t0 = std::chrono::high_resolution_clock::now();
    auto r  = client.predict(id, inputs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rtt = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (r.success) {
        ui_ok("OK");
        printf("    Inference : %.3f ms (server-side)\n", r.inference_time_ms);
        printf("    RTT       : %.3f ms (end-to-end)\n",  rtt);
        std::cout << "\n";
        for (const auto& [name, val] : r.outputs) {
            size_t count = 0;
            double first = 0.0;
            if (val.is_number()) {
                count = 1;
                first = val.as_number();
            } else if (val.is_array() && !val.as_array().empty()) {
                count = val.as_array().size();
                const auto& f = val.as_array().front();
                if (f.is_number()) first = f.as_number();
            }
            printf("    Out %-16s  %zu values  first=%.4f\n",
                   name.c_str(), count, static_cast<float>(first));
        }
        record(id, true, r.inference_time_ms, rtt);
    } else {
        ui_err("Failed: " + r.error_message);
        record(id, false, 0, 0);
    }

    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: Warmup
// =============================================================================

void flow_warmup(Client& client) {
    std::string id = pick_loaded_model(client);
    if (id.empty()) return;

    clear();
    draw_header("Warmup: " + id);
    int n = ask_int("Runs (default 10): ", 10);
    if (n <= 0) return;

    std::cout << "\n";
    ui_inf("Warming up...");
    auto w = client.warmup_model(id, static_cast<uint32_t>(n));

    if (w.success)
        printf("    Runs: %u  Avg: %.3f ms  Min: %.3f ms  Max: %.3f ms\n",
               w.runs_completed, w.avg_time_ms, w.min_time_ms, w.max_time_ms);
    else
        ui_err("Failed: " + w.error_message);

    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: Benchmark
// =============================================================================

void flow_benchmark(Client& client) {
    std::string id = pick_loaded_model(client);
    if (id.empty()) return;

    clear();
    draw_header("Benchmark: " + id);
    int n = ask_int("Requests (default 100): ", 100);
    if (n <= 0) return;

    auto info   = client.get_model_info(id);
    auto inputs = make_inputs(info);
    if (inputs.empty()) { ui_err("Schema has no inputs."); draw_footer(); wait_key(); return; }

    std::cout << "\n";
    ui_inf("Running " + std::to_string(n) + " requests...\n");

    PerfStats bench;
    auto wall_start = std::chrono::high_resolution_clock::now();

    int step = std::max(1, n / 20);
    for (int i = 0; i < n; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r  = client.predict(id, inputs);
        auto t1 = std::chrono::high_resolution_clock::now();
        double rtt = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bench.add(r.success, r.inference_time_ms, rtt);
        record(id, r.success, r.inference_time_ms, rtt);

        if ((i + 1) % step == 0 || i == n - 1) {
            int pct = static_cast<int>(100.0 * (i + 1) / n);
            int bar = pct / 5;
            std::cout << "    [";
            for (int b = 0; b < 20; ++b)
                std::cout << (b < bar ? "#" : "-");
            printf("] %3d%%\r", pct);
            std::cout.flush();
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(
        wall_end - wall_start).count();
    std::cout << "\n\n";

    bench.print_table(wall_ms);

    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: Worker Status
// =============================================================================

void flow_status(Client& client) {
    auto s = client.get_status();

    clear();
    draw_header("Worker Status");

    printf("    %-20s %s\n",     "Worker ID:", s.worker_id.c_str());
    printf("    %-20s %lld s\n", "Uptime:",    (long long)s.uptime_seconds);
    printf("    %-20s %llu total  %llu ok  %llu failed  %u active\n",
           "Requests:",
           (unsigned long long)s.total_requests,
           (unsigned long long)s.successful_requests,
           (unsigned long long)s.failed_requests,
           s.active_requests);

    if (!s.loaded_models.empty()) {
        std::cout << "\n    " << c::B << "Loaded models:" << c::RST << "\n";
        for (const auto& m : s.loaded_models)
            std::cout << "      • " << m << "\n";
    }

    if (!s.supported_backends.empty()) {
        std::cout << "\n    " << c::B << "Backends:" << c::RST << "\n";
        for (const auto& b : s.supported_backends)
            std::cout << "      • " << b << "\n";
    }

    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: Session Statistics (client-side)
// =============================================================================

void flow_session_stats() {
    clear();
    draw_header("Session Statistics", "client-side measurements only");

    if (session_stats.empty()) {
        ui_wrn("No requests recorded in this session.");
        ui_inf("Load a model and run at least one inference to see statistics.");
    } else {
        for (const auto& [id, stats] : session_stats) {
            std::cout << "  " << c::CY << c::B << id << c::RST << "\n";
            stats.print_table();
            std::cout << "\n";
        }

        if (session_stats.size() > 1) {
            PerfStats all;
            for (const auto& [_, s] : session_stats) {
                all.inference.insert(all.inference.end(),
                                     s.inference.begin(), s.inference.end());
                all.rtt.insert(all.rtt.end(), s.rtt.begin(), s.rtt.end());
                all.failures += s.failures;
            }
            std::cout << "  " << c::B << "TOTAL (all models)" << c::RST << "\n";
            all.print_table();
        }
    }

    draw_footer();
    wait_key();
}

// =============================================================================
// FLOW: Server Inference Metrics (live, auto-refresh a cada 500 ms)
// =============================================================================

static void render_server_metrics(const mlinference::client::ServerMetrics& m,
                                   int refresh_counter) {
    auto fmt_time = [](int64_t unix_ts) -> std::string {
        if (unix_ts <= 0) return "—";
        time_t t = static_cast<time_t>(unix_ts);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
        return std::string(buf);
    };

    auto clamp0 = [](double v) -> double { return v < 0.0 ? 0.0 : v; };

    clear();

    const char* spin[] = {"⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"};
    std::string indicator = std::string(spin[refresh_counter % 10])
                          + " live · refresh 500 ms";

    draw_header("Server Inference Metrics", indicator);

    std::cout << "    " << c::B << "Uptime      " << c::RST
              << m.uptime_seconds << " s\n"
              << "    " << c::B << "Requests    " << c::RST
              << m.total_requests       << " total, "
              << c::G << m.successful_requests << " ok" << c::RST << ", "
              << c::R << m.failed_requests     << " failed" << c::RST << ", "
              << m.active_requests << " active\n\n";

    if (m.per_model.empty()) {
        ui_wrn("No per-model metrics available.");
        ui_inf("Load a model and run at least one inference to see statistics.");
    } else {
        for (const auto& [id, mm] : m.per_model) {
            uint64_t ok_count = mm.total_inferences - mm.failed_inferences;

            std::cout << "  " << c::CY << c::B << id << c::RST << "\n";
            printf("    Inferences : %llu total, %llu ok, %llu failed\n",
                   (unsigned long long)mm.total_inferences,
                   (unsigned long long)ok_count,
                   (unsigned long long)mm.failed_inferences);

            if (ok_count > 0) {
                std::cout << "\n";
                printf("    %-22s %8s %8s %8s %8s %8s\n",
                       "Metric (ms)", "avg", "min", "max", "p95", "p99");
                std::cout << "    ";
                for (int i = 0; i < 68; ++i) std::cout << "─";
                std::cout << "\n";
                printf("    %-22s %8.3f %8.3f %8.3f %8.3f %8.3f\n",
                       "Server inference",
                       clamp0(mm.avg_ms),
                       clamp0(mm.min_ms),
                       clamp0(mm.max_ms),
                       clamp0(mm.p95_ms),
                       clamp0(mm.p99_ms));

                std::cout << "\n";

                if (mm.failed_inferences > 0) {
                    double err_pct = 100.0
                        * static_cast<double>(mm.failed_inferences)
                        / static_cast<double>(mm.total_inferences);
                    printf("    %sError rate  : %.1f %%%s\n", c::R, err_pct, c::RST);
                }

                // Throughput correto: req/s baseado no tempo decorrido desde o
                // carregamento do modelo (wall time real), não na soma das latências.
                if (mm.loaded_at_unix > 0) {
                    auto now_unix = static_cast<int64_t>(
                        std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count());
                    double elapsed_s = static_cast<double>(now_unix - mm.loaded_at_unix);
                    if (elapsed_s > 0.0)
                        printf("    Throughput  : %.1f req/s (since load)\n",
                               static_cast<double>(mm.total_inferences) / elapsed_s);
                }

                if (mm.avg_ms > 0.0 && mm.p99_ms > 0.0)
                    printf("    Tail ratio  : %.1fx  (p99 / avg)\n",
                           clamp0(mm.p99_ms) / mm.avg_ms);
            }

            printf("    Loaded at   : %s\n", fmt_time(mm.loaded_at_unix).c_str());
            printf("    Last used   : %s\n", fmt_time(mm.last_used_at_unix).c_str());
            std::cout << "\n";
        }
    }

    std::cout << " " << c::BL << "│" << c::RST << "\n"
              << " " << c::BL << "└────────────────────────────────────────────────"
              << c::RST << "  " << c::DIM << "Q / ESC  back to menu" << c::RST << "\n";
}

// -----------------------------------------------------------------------------
// flow_server_metrics — loop de refresh live.
//
// Ordem correta:
//   1. render
//   2. poll (bloqueia até 500 ms ou tecla)
//   3. se tecla de saída → break; senão → volta ao 1
//
// Desta forma cada ciclo dura exatamente ~500 ms e não há double-render.
// -----------------------------------------------------------------------------
void flow_server_metrics(Client& client) {
    int counter = 0;

    while (true) {
        render_server_metrics(client.get_metrics(), counter++);

        auto ev = poll_key(500);
        if (ev.key == Key::ESC || ev.key == Key::BACK) break;
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    // ---- Defaults ----
    std::string address    = "0.0.0.0:50052";
    std::string models_dir = "../models";

    // ---- Parse CLI ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--address" && i + 1 < argc) {
            address = argv[++i];
        } else if (arg == "--models-dir" && i + 1 < argc) {
            models_dir = argv[++i];
        } else {
            std::cerr << "Argumento desconhecido: " << arg << std::endl;
            return 1;
        }
    }

    // Splash + conexão (fora do alt screen para preservar o histórico)
    std::cout << c::CY << c::B
              << "\n"
              << "      _               __  __ _ _\n"
              << "     / \\   ___  __ _ |  \\/  (_|_) __ _\n"
              << "    / _ \\ / __|/ _` || |\\/| | | |/ _` |\n"
              << "   / ___ \\\\__ \\ (_| || |  | | | | (_| |\n"
              << "  /_/   \\_\\___/\\__,_||_|  |_|_|_|\\__,_|\n"
              << c::RST << c::DIM
              << "    ML Inference Client — Interactive\n"
              << c::RST << "\n";

    std::cout << "  Connecting to " << c::B << address << c::RST << " ...\n";
    Client client(address);

    if (!client.connect()) {
        std::cerr << c::R << "  ✗ Cannot connect to " << address << c::RST << "\n";
        return 1;
    }
    std::cout << c::G << "  ✓ Connected" << c::RST << "\n";

    if (!client.health_check()) {
        std::cerr << c::R << "  ✗ Server health check failed" << c::RST << "\n";
        return 1;
    }
    std::cout << c::G << "  ✓ Server healthy" << c::RST << "\n\n";
    std::cout << c::DIM << "  Entering interactive mode..." << c::RST << std::flush;
    usleep(1500000);

    // Entra no alt screen
    TerminalGuard guard;

    // ── Main menu loop ────────────────────────────────────────────────────────

    const std::vector<std::string> menu_items = {
        "Browse available models (server-side)",
        "List loaded models",
        "Load model (manual path)",
        "Unload model",
        "Validate model (dry-run)",
        "Single prediction",
        "Warmup",
        "Benchmark",
        "Worker status",
        "Session statistics",
        "Server inference metrics",
        "Quit",
    };

    while (true) {
        int ch = list_selector("MiiaClient — Main Menu", menu_items,
                               "↑↓ navigate  ENTER select  Q quit");

        switch (ch) {
            case  0: flow_browse(client, models_dir); break;
            case  1: flow_list(client);               break;
            case  2: flow_load(client);               break;
            case  3: flow_unload(client);             break;
            case  4: flow_validate(client);           break;
            case  5: flow_predict(client);            break;
            case  6: flow_warmup(client);             break;
            case  7: flow_benchmark(client);          break;
            case  8: flow_status(client);             break;
            case  9: flow_session_stats();            break;
            case 10: flow_server_metrics(client);     break;
            case 11: // Quit
            case -1: // ESC / Q
                return 0;
            default: break;
        }
    }
}
// NOLINTEND