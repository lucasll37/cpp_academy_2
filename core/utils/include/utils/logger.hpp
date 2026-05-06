// =============================================================================
/// @file   logger.hpp
/// @brief  Sistema de logging com múltiplos destinos, níveis e saída em arquivo.
///
/// @details
/// Implementa um logger por componente (*named logger*) com as seguintes
/// características:
///
/// - **Múltiplos destinos:** cada componente grava em seu próprio arquivo
///   (`logs/<nome>_<timestamp>.log`) e é espelhado automaticamente no
///   arquivo agregado `logs/all_<timestamp>.log`.
/// - **Saída condicional em stderr:** controlável globalmente e por componente.
/// - **Inicialização lazy:** arquivos são abertos apenas na primeira mensagem
///   do componente — componentes silenciosos não criam arquivos.
/// - **Nível mínimo global:** controlado pela variável de ambiente `LOG_LEVEL`.
///   Se não definida, o logger fica desligado (`OFF`) e nenhum arquivo é criado.
/// - **Thread-safe:** cada instância possui um `std::mutex`; o registro global
///   (`GlobalState`) é protegido por um mutex separado.
/// - **Zero-overhead quando desligado:** as macros verificam `min_level()`
///   antes de qualquer acesso ao registry — nenhum `Logger` é instanciado
///   se o nível da mensagem for insuficiente.
///
/// ### Configuração rápida
/// @code
/// // Antes de qualquer uso (geralmente em main()):
/// Logger::set_base_dir("logs");          // diretório base dos arquivos
/// Logger::set_default_stderr(true);      // espelhar no terminal por padrão
///
/// // Opcional: configurar por componente
/// Logger::configure("ml", LoggerConfig{
///     .dir         = "logs/ml",   // diretório próprio
///     .also_stderr = true,        // sempre no terminal
/// });
/// @endcode
///
/// ### Variável de ambiente
/// ```
/// export LOG_LEVEL=DEBUG   # DEBUG | INFO | WARN | ERROR
/// ```
/// Se `LOG_LEVEL` não estiver definida, o logger usa `Level::OFF` e
/// nenhuma mensagem é gravada.
///
/// ### Formato de saída
/// ```
/// HH:MM:SS.mmm LEVEL  component  mensagem  [arquivo:linha]
/// ```
/// Exemplo:
/// ```
/// 10:42:17.053 INFO   inference  [load_model] modelo carregado: nav  [inference_engine.cpp:142]
/// ```
/// No terminal, o nível é colorido com ANSI:
/// | Nível  | Cor      |
/// |--------|----------|
/// | DEBUG  | Ciano    |
/// | INFO   | Verde    |
/// | WARN   | Amarelo  |
/// | ERROR  | Vermelho |
///
/// ### Uso básico
/// @code
/// #include <utils/logger.hpp>
///
/// // Logger nomeado — arquivo próprio + espelhado em "all"
/// LOG_INFO("inference")  << "modelo carregado: " << model_id;
/// LOG_DEBUG("inference") << "latência=" << ms << " ms";
/// LOG_WARN("inference")  << "venv não encontrado, prosseguindo";
/// LOG_ERROR("inference") << "falha ao carregar: " << path;
///
/// // Sintaxe alternativa com level explícito
/// LOG("inference", INFO) << "modelo carregado: " << model_id;
///
/// // Logger "default" — só espelhado em "all", sem arquivo próprio
/// LOG_INFO() << "mensagem avulsa";
/// LOG(INFO)  << "mensagem avulsa";
/// @endcode
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

// =============================================================================
// Macros públicas
// =============================================================================

/// @defgroup logger_macros Macros de logging
/// @{

/// @brief Macro interna — cria um #Logger::LogEntry para o componente e nível dados.
///
/// @details
/// Verifica `Logger::min_level()` **antes** de acessar o registry.
/// Se o nível for insuficiente, passa `nullptr` como logger — o
/// `LogEntry` descarta a mensagem sem overhead de I/O.
#define _LOG_ENTRY(name, level) \
    Logger::LogEntry(Logger::Level::level, __FILE__, __LINE__, \
        Logger::Level::level >= Logger::min_level() \
            ? &Logger::get(name) : nullptr)

/// @brief Macro de logging com sintaxe de stream.
///
/// Suporta dois formatos:
/// - `LOG("component", LEVEL) << msg;` — logger nomeado, level explícito.
/// - `LOG(LEVEL) << msg;`              — logger `"default"` (só vai para `"all"`).
///
/// @code
/// LOG("inference", INFO)  << "modelo carregado: " << id;
/// LOG("inference", ERROR) << "falha: " << err;
/// LOG(WARN) << "aviso global";
/// @endcode
#define LOG(...) _LOG_SELECT(__VA_ARGS__, _LOG2, _LOG1)(__VA_ARGS__)
#define _LOG_SELECT(_1, _2, MACRO, ...) MACRO
#define _LOG2(name, level) _LOG_ENTRY(name,      level)
#define _LOG1(level)       _LOG_ENTRY("default", level)

/// @cond INTERNAL
#define _LOG_LEVEL(level, ...) \
    _LOG_LEVEL_SELECT(__VA_ARGS__, _LOG_LEVEL2, _LOG_LEVEL1)(level, ##__VA_ARGS__)
#define _LOG_LEVEL_SELECT(_1, MACRO, ...) MACRO
#define _LOG_LEVEL2(level, name) _LOG_ENTRY(name,      level)
#define _LOG_LEVEL1(level)       _LOG_ENTRY("default", level)
/// @endcond

/// @brief Emite mensagem de nível DEBUG.
///
/// Suporta componente opcional:
/// - `LOG_DEBUG("component") << msg;`
/// - `LOG_DEBUG() << msg;`  (logger `"default"`)
#define LOG_DEBUG(...) _LOG_LEVEL(DEBUG, ##__VA_ARGS__)

/// @brief Emite mensagem de nível INFO.
///
/// - `LOG_INFO("component") << msg;`
/// - `LOG_INFO() << msg;`
#define LOG_INFO(...)  _LOG_LEVEL(INFO,  ##__VA_ARGS__)

/// @brief Emite mensagem de nível WARN.
///
/// - `LOG_WARN("component") << msg;`
/// - `LOG_WARN() << msg;`
#define LOG_WARN(...)  _LOG_LEVEL(WARN,  ##__VA_ARGS__)

/// @brief Emite mensagem de nível ERROR.
///
/// - `LOG_ERROR("component") << msg;`
/// - `LOG_ERROR() << msg;`
#define LOG_ERROR(...) _LOG_LEVEL(ERROR, ##__VA_ARGS__)

/// @}

// =============================================================================
// LoggerConfig
// =============================================================================

/// @brief Configuração opcional por componente, aplicada antes do primeiro uso.
///
/// @details
/// Passada a `Logger::configure()`.  Campos não preenchidos herdam os valores
/// globais definidos em `Logger::set_base_dir()` e `Logger::set_default_stderr()`.
///
/// @note `configure()` é ignorado silenciosamente se chamado após o primeiro
///       uso do componente (primeira mensagem que efetivamente passa o filtro
///       de nível).
///
/// @code
/// Logger::configure("ml", LoggerConfig{
///     .dir         = "logs/ml",  // diretório próprio para este componente
///     .also_stderr = true,       // sempre exibir no terminal
/// });
/// @endcode
struct LoggerConfig {
    /// @brief Diretório de saída do arquivo de log deste componente.
    ///
    /// Sobrescreve o `base_dir` global.  Se vazio, usa o global.
    std::optional<std::string> dir;

    /// @brief Se @c true, espelha mensagens em `stderr` além do arquivo.
    ///
    /// Sobrescreve `default_stderr` global para este componente.
    std::optional<bool> also_stderr;
};

// =============================================================================
// Logger
// =============================================================================

/// @brief Logger por componente com múltiplos destinos e inicialização lazy.
///
/// @details
/// Não deve ser instanciado diretamente — use as macros #LOG_INFO,
/// #LOG_DEBUG, #LOG_WARN, #LOG_ERROR ou #LOG.
///
/// ### Ciclo de vida de um logger
/// 1. Primeira mensagem com nível suficiente → `get("component")` cria a
///    instância no registry e chama `lazy_init()`.
/// 2. `lazy_init()` abre o arquivo `<dir>/<name>_<timestamp>.log`.
/// 3. Mensagens subsequentes escrevem diretamente na instância (sem busca
///    no registry).
/// 4. O destrutor fecha o arquivo.
///
/// ### Logger especial "all"
/// Recebe uma cópia de todas as mensagens de todos os outros loggers.
/// Criado automaticamente na primeira mensagem de qualquer componente.
///
/// ### Logger especial "default"
/// Usado pelas macros sem nome de componente (`LOG(INFO) << ...`).
/// **Não cria arquivo próprio** — as mensagens são apenas espelhadas em `"all"`.
class Logger {
public:
    // -------------------------------------------------------------------------
    /// @name Enumeração de níveis
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Níveis de severidade em ordem crescente.
    ///
    /// | Valor  | Uso                                      |
    /// |--------|------------------------------------------|
    /// | DEBUG  | Detalhes internos, fluxo de execução     |
    /// | INFO   | Eventos normais do ciclo de vida         |
    /// | WARN   | Situações anômalas não-fatais            |
    /// | ERROR  | Falhas que impedem a operação pretendida |
    /// | OFF    | Desliga o logger (nenhuma saída)         |
    enum class Level { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, OFF = 4 };

    /// @}

    // -------------------------------------------------------------------------
    /// @name Constantes de nomes reservados
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Nome do logger agregado que recebe cópia de todos os outros.
    static constexpr const char* ALL     = "all";

    /// @brief Nome do logger sem arquivo próprio (só espelha em `"all"`).
    static constexpr const char* DEFAULT = "default";

    /// @}

    // -------------------------------------------------------------------------
    /// @name Nível mínimo global
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Retorna o nível mínimo lido da variável de ambiente `LOG_LEVEL`.
    ///
    /// @details
    /// Inicializado uma única vez via *magic static* no primeiro acesso.
    /// Valores aceitos para `LOG_LEVEL`: `DEBUG`, `INFO`, `WARN`, `ERROR`.
    /// Qualquer outro valor (ou ausência da variável) resulta em `Level::OFF`.
    ///
    /// As macros chamam este método **antes** de acessar o registry, garantindo
    /// zero-overhead quando o logger está desligado ou o nível é insuficiente.
    ///
    /// @return Nível mínimo configurado; `Level::OFF` se `LOG_LEVEL` não estiver
    ///         definida ou contiver valor inválido.
    static Level min_level() {
        static const Level lvl = []() -> Level {
            const char* env = std::getenv("LOG_LEVEL");
            if (!env) return Level::OFF;
            const std::string s(env);
            if (s == "DEBUG") return Level::DEBUG;
            if (s == "INFO")  return Level::INFO;
            if (s == "WARN")  return Level::WARN;
            if (s == "ERROR") return Level::ERROR;
            return Level::OFF;
        }();
        return lvl;
    }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Configuração global
    /// Deve ser chamada antes do primeiro uso de qualquer logger.
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Define o diretório base para todos os arquivos de log.
    ///
    /// @details
    /// Padrão: `"logs"` (relativo ao diretório de trabalho).
    /// Pode ser sobrescrito por componente via #LoggerConfig::dir.
    ///
    /// @param dir  Caminho do diretório (criado automaticamente se necessário).
    static void set_base_dir(const std::string& dir) {
        auto& g = globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        g.base_dir = dir;
    }

    /// @brief Define se os loggers espelham mensagens em `stderr` por padrão.
    ///
    /// @details
    /// Padrão: `false`.  Pode ser sobrescrito por componente via
    /// #LoggerConfig::also_stderr.
    ///
    /// @param enabled  @c true para habilitar saída em `stderr` por padrão.
    static void set_default_stderr(bool enabled) {
        auto& g = globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        g.default_stderr = enabled;
    }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Configuração por componente
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Define configuração específica para um componente.
    ///
    /// @details
    /// Deve ser chamada **antes** do primeiro uso do componente (antes da
    /// primeira mensagem que passe o filtro de nível).  Se o componente já
    /// tiver sido inicializado, a chamada é ignorada silenciosamente.
    ///
    /// @param name  Nome do componente (ex.: `"inference"`, `"ml"`).
    /// @param cfg   Configuração a aplicar (campos não preenchidos herdam o global).
    static void configure(const std::string& name, LoggerConfig cfg) {
        auto& g = globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        if (g.initialized.count(name)) return;
        g.configs[name] = std::move(cfg);
    }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Acesso ao registry
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Retorna (ou cria lazily) o logger para o componente indicado.
    ///
    /// @details
    /// Protegido por `GlobalState::mutex`.  Na primeira chamada para um dado
    /// `name`, instancia o logger e chama `lazy_init()`, que abre o arquivo
    /// de log.  Chamadas subsequentes retornam a instância já existente.
    ///
    /// @note Este método é chamado apenas pelas macros, após a verificação de
    ///       `min_level()`.  Não deve ser chamado diretamente.
    ///
    /// @param name  Nome do componente (padrão: `"default"`).
    ///
    /// @return Referência para o logger do componente.
    static Logger& get(const std::string& name = DEFAULT) {
        auto& g = globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        auto it = g.registry.find(name);
        if (it != g.registry.end()) return it->second;
        Logger& logger = g.registry[name];
        g.initialized[name] = true;
        logger.lazy_init(name, g);
        return logger;
    }

    /// @}

    // -------------------------------------------------------------------------
    // Ciclo de vida da instância
    // -------------------------------------------------------------------------

    /// @brief Fecha o arquivo de log ao destruir o logger.
    ~Logger() { if (file_.is_open()) file_.close(); }

    /// @cond INTERNAL
    Logger()                         = default;
    Logger(Logger&&)                 = default;
    Logger& operator=(Logger&&)      = default;
    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;
    /// @endcond

    // -------------------------------------------------------------------------
    /// @name LogEntry — objeto temporário criado pelas macros
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Objeto temporário que acumula a mensagem e a grava no destrutor.
    ///
    /// @details
    /// Criado pelas macros via `_LOG_ENTRY`.  Quando `logger == nullptr`
    /// (nível insuficiente), todos os `operator<<` são no-ops e o destrutor
    /// não chama `Logger::log()` — zero overhead.
    ///
    /// A gravação ocorre no **destrutor**, garantindo que a mensagem completa
    /// (incluindo todos os `<< val`) seja escrita atomicamente.
    class LogEntry {
    public:
        /// @brief Constrói o entry.
        ///
        /// @param level   Nível da mensagem.
        /// @param file    Caminho do arquivo fonte (`__FILE__`).
        /// @param line    Número da linha (`__LINE__`).
        /// @param logger  Ponteiro para o logger destino, ou @c nullptr para no-op.
        LogEntry(Level level, const char* file, int line, Logger* logger)
            : logger_(logger), level_(level), file_(file), line_(line) {}

        /// @brief Grava a mensagem acumulada no logger destino.
        ~LogEntry() {
            if (logger_)
                logger_->log(level_, oss_.str(), file_, line_);
        }

        /// @brief Acumula um valor na mensagem via `std::ostringstream`.
        ///
        /// @tparam T  Qualquer tipo suportado por `operator<<` de `ostream`.
        template <typename T>
        LogEntry& operator<<(const T& val) {
            if (logger_) oss_ << val;
            return *this;
        }

        /// @cond INTERNAL
        LogEntry(LogEntry&&)                 = default;
        LogEntry(const LogEntry&)            = delete;
        LogEntry& operator=(LogEntry&&)      = delete;
        LogEntry& operator=(const LogEntry&) = delete;
        /// @endcond

    private:
        Logger*            logger_;  ///< nullptr = no-op (nível insuficiente).
        Level              level_;
        const char*        file_;
        int                line_;
        std::ostringstream oss_;     ///< Buffer acumulador da mensagem.
    };

    /// @}

    // -------------------------------------------------------------------------
    // Método de gravação (chamado pelo destrutor de LogEntry)
    // -------------------------------------------------------------------------

    /// @brief Grava uma mensagem no arquivo e opcionalmente em stderr.
    ///
    /// @details
    /// Formata a entrada como:
    /// ```
    /// HH:MM:SS.mmm LEVEL  component  mensagem  [arquivo:linha]
    /// ```
    /// e grava no arquivo próprio do componente (se aberto) e, se não for o
    /// logger `"all"`, espelha automaticamente no logger `"all"`.
    ///
    /// @param level    Nível da mensagem.
    /// @param msg      Texto acumulado pelo `LogEntry`.
    /// @param filepath Caminho do arquivo fonte (`__FILE__`), ou @c nullptr.
    /// @param line     Número da linha no arquivo fonte.
    void log(Level              level,
             const std::string& msg,
             const char*        filepath = nullptr,
             int                line     = 0)
    {
        if (level < min_level()) return;

        const std::string timestamp = current_timestamp();
        const std::string location  = filepath
            ? short_filename(filepath) + ":" + std::to_string(line)
            : "";
        const std::string tag = level_str(level);

        write_entry(level, timestamp, tag, location, msg);

        // Espelha no "all", exceto se este já é o "all".
        if (!is_all_)
            Logger::get(ALL).write_entry(level, timestamp, tag, location, msg);
    }

    /// @brief Retorna @c true se o arquivo de log está aberto.
    bool is_open() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return file_.is_open();
    }

private:
    /// @cond INTERNAL

    // -------------------------------------------------------------------------
    // Estado global compartilhado entre todas as instâncias
    // -------------------------------------------------------------------------

    /// @brief Estado global do sistema de logging.
    struct GlobalState {
        std::mutex                                    mutex;        ///< Protege o registry.
        std::unordered_map<std::string, Logger>       registry;     ///< Instâncias por nome.
        std::unordered_map<std::string, LoggerConfig> configs;      ///< Configs por nome.
        std::unordered_map<std::string, bool>         initialized;  ///< Nomes já inicializados.
        std::string                                   base_dir       = "logs"; ///< Dir padrão.
        bool                                          default_stderr = false;  ///< stderr padrão.
        std::mutex                                    stderr_mutex;  ///< Protege stderr.
    };

    /// @brief Retorna o estado global singleton (*magic static*).
    static GlobalState& globals() {
        static GlobalState g;
        return g;
    }

    // -------------------------------------------------------------------------
    // Inicialização lazy
    // -------------------------------------------------------------------------

    /// @brief Inicializa o logger: define flags e abre o arquivo de log.
    ///
    /// @details
    /// Chamado dentro do lock do registry na primeira vez que o componente
    /// é acessado.  O logger `"default"` não abre arquivo próprio.
    /// O arquivo é nomeado `<dir>/<name>_<timestamp>.log`.
    void lazy_init(const std::string& name, GlobalState& g) {
        is_all_      = (name == ALL);
        also_stderr_ = resolve_stderr(name, g);

        // "default" não abre arquivo próprio — só espelha no "all".
        if (name == DEFAULT) return;

        const std::string dir  = resolve_dir(name, g);
        const std::string path = (std::filesystem::path(dir) /
                                  (name + "_" + file_timestamp() + ".log")).string();

        std::filesystem::create_directories(dir);
        file_.open(path, std::ios::app);

        if (!file_.is_open())
            std::cerr << "[Logger] Não foi possível abrir: " << path << "\n";
        else if (also_stderr_)
            std::cerr << "[Logger] Gravando em: " << path << "\n";
    }

    /// @brief Resolve o diretório de saída para o componente.
    std::string resolve_dir(const std::string& name, const GlobalState& g) const {
        auto it = g.configs.find(name);
        if (it != g.configs.end() && it->second.dir.has_value())
            return it->second.dir.value();
        return g.base_dir;
    }

    /// @brief Resolve se o componente deve espelhar em stderr.
    bool resolve_stderr(const std::string& name, const GlobalState& g) const {
        auto it = g.configs.find(name);
        if (it != g.configs.end() && it->second.also_stderr.has_value())
            return it->second.also_stderr.value();
        return g.default_stderr;
    }

    // -------------------------------------------------------------------------
    // Escrita formatada
    // -------------------------------------------------------------------------

    /// @brief Formata e grava uma entrada no arquivo e opcionalmente em stderr.
    void write_entry(Level              level,
                     const std::string& timestamp,
                     const std::string& tag,
                     const std::string& location,
                     const std::string& msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (file_.is_open()) {
            // Arquivo: sem cores ANSI.
            file_ << timestamp << " " << tag << "  " << msg;
            if (!location.empty()) file_ << "  [" << location << "]";
            file_ << "\n";
            file_.flush();
        }

        if (also_stderr_) {
            auto& g = globals();
            std::lock_guard<std::mutex> slock(g.stderr_mutex);
            std::cerr << level_color(level)
                      << timestamp << " " << tag << "  " << msg;
            if (!location.empty()) std::cerr << "  [" << location << "]";
            std::cerr << RESET << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // Formatação de timestamp
    // -------------------------------------------------------------------------

    /// @brief Retorna timestamp no formato `HH:MM:SS.mmm` para logs em arquivo.
    static std::string current_timestamp() {
        using namespace std::chrono;
        auto now  = system_clock::now();
        auto time = system_clock::to_time_t(now);
        auto ms   = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
        std::ostringstream oss;
        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &time);
#else
        localtime_r(&time, &tm_buf);
#endif
        oss << std::put_time(&tm_buf, "%H:%M:%S") << "."
            << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    /// @brief Retorna timestamp no formato `YYYYMMDD_HHMMSS` para nomes de arquivo.
    static std::string file_timestamp() {
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &time);
#else
        localtime_r(&time, &tm_buf);
#endif
        oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
        return oss.str();
    }

    /// @brief Extrai o nome do arquivo de um caminho completo (`__FILE__`).
    static std::string short_filename(const char* path) {
        std::string s(path);
        const auto pos = s.find_last_of("/\\");
        return (pos == std::string::npos) ? s : s.substr(pos + 1);
    }

    // -------------------------------------------------------------------------
    // Cores ANSI
    // -------------------------------------------------------------------------

    static constexpr const char* RESET  = "\033[0m";   ///< Reset de atributos.
    static constexpr const char* DIM    = "\033[2m";   ///< Texto esmaecido.
    static constexpr const char* CYAN   = "\033[36m";  ///< DEBUG.
    static constexpr const char* GREEN  = "\033[32m";  ///< INFO.
    static constexpr const char* YELLOW = "\033[33m";  ///< WARN.
    static constexpr const char* RED    = "\033[31m";  ///< ERROR.

    /// @brief Retorna o código ANSI correspondente ao nível.
    static const char* level_color(Level l) {
        switch (l) {
            case Level::DEBUG: return CYAN;
            case Level::INFO:  return GREEN;
            case Level::WARN:  return YELLOW;
            case Level::ERROR: return RED;
            default:           return RESET;
        }
    }

    /// @brief Retorna a string de 5 caracteres do nível (com espaço de alinhamento).
    static const char* level_str(Level l) {
        switch (l) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO:  return "INFO ";
            case Level::WARN:  return "WARN ";
            case Level::ERROR: return "ERROR";
            default:           return "?????";
        }
    }

    // -------------------------------------------------------------------------
    // Estado por instância
    // -------------------------------------------------------------------------

    std::ofstream      file_;         ///< Arquivo de log próprio do componente.
    mutable std::mutex mutex_;        ///< Protege file_ e also_stderr_.
    bool               also_stderr_ = false; ///< Espelhar em stderr.
    bool               is_all_      = false; ///< @c true se este é o logger "all".

    /// @endcond
};