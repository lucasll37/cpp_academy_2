// =============================================================================
// s3_model_source.cpp — Implementação via AWS SDK C++
// =============================================================================

#include "worker/s3_model_source.hpp"

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3ClientConfiguration.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/GetObjectRequest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mlinference {
namespace worker {

// ============================================================================
// SdkGuard — definição completa (struct público no .cpp, não aninhado)
// O header declara apenas "struct SdkGuard;" como friend para o shared_ptr.
// Aqui usamos uma abordagem mais simples: struct livre no namespace anônimo.
// ============================================================================
namespace {

struct SdkGuardImpl {
    Aws::SDKOptions options;

    SdkGuardImpl() {
        options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Trace;
        Aws::InitAPI(options);
    }

    ~SdkGuardImpl() {
        Aws::ShutdownAPI(options);
        std::cout << "[S3] AWS SDK finalizado." << std::endl;
    }
};

std::weak_ptr<SdkGuardImpl> g_sdk_guard_weak;
std::mutex                  g_sdk_mutex;

const std::vector<std::string> KNOWN_EXTENSIONS = {".onnx", ".py"};

bool is_known_ext(const std::string& ext) {
    return std::find(KNOWN_EXTENSIONS.begin(), KNOWN_EXTENSIONS.end(), ext) !=
           KNOWN_EXTENSIONS.end();
}

}  // namespace anon

// ============================================================================
// SdkGuard — tipo opaco exposto pelo header apenas para o shared_ptr
// Redirecionamos para SdkGuardImpl internamente.
// ============================================================================
struct S3ModelSource::SdkGuard {
    std::shared_ptr<SdkGuardImpl> impl;
    explicit SdkGuard(std::shared_ptr<SdkGuardImpl> p) : impl(std::move(p)) {}
};

// ============================================================================
// Construtor / Destrutor
// ============================================================================

S3ModelSource::S3ModelSource(S3Config config)
    : config_(std::move(config)) {

    // Garante InitAPI exatamente uma vez por processo
    {
        std::lock_guard<std::mutex> lock(g_sdk_mutex);
        auto impl = g_sdk_guard_weak.lock();
        if (!impl) {
            impl = std::make_shared<SdkGuardImpl>();
            g_sdk_guard_weak = impl;
        }
        sdk_guard_ = std::make_shared<SdkGuard>(impl);
    }

    fs::create_directories(config_.cache_dir);
    client_ = make_client();

    std::cout << "[S3] Fonte configurada: s3://"
              << config_.bucket << "/" << config_.prefix;
    if (!config_.endpoint_url.empty())
        std::cout << "  (endpoint: " << config_.endpoint_url << ")";
    std::cout << std::endl;
}

S3ModelSource::~S3ModelSource() = default;

// ============================================================================
// make_client() — usa a API correta do SDK 1.11
// Construtores disponíveis (ver S3Client.h linha 81 e 95):
//   S3Client(const S3ClientConfiguration&, shared_ptr<EndpointProviderBase>)
//   S3Client(const shared_ptr<AWSCredentialsProvider>&,
//            shared_ptr<EndpointProviderBase>,
//            const S3ClientConfiguration&)
// ============================================================================

std::shared_ptr<Aws::S3::S3Client> S3ModelSource::make_client() const {
    Aws::S3::S3ClientConfiguration cfg;
    cfg.region = config_.region;

    if (!config_.endpoint_url.empty()) {
        cfg.endpointOverride = config_.endpoint_url;
    }

    // Para MinIO/LocalStack: desabilita virtual hosting e força path-style
    if (config_.use_path_style) {
        cfg.useVirtualAddressing        = false;
        cfg.enableHostPrefixInjection   = false;  // era disableHostPrefixInjection
    }

    auto creds = std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();

    return std::make_shared<Aws::S3::S3Client>(
        creds,
        Aws::MakeShared<Aws::S3::S3EndpointProvider>("S3ModelSource"),
        cfg
    );
}

// ============================================================================
// Utilitários de URI / paths
// ============================================================================

// static
std::string S3ModelSource::make_uri(const std::string& bucket,
                                     const std::string& key) {
    return "s3://" + bucket + "/" + key;
}

// static
std::string S3ModelSource::key_from_uri(const std::string& s3_uri) {
    const std::string pfx = "s3://";
    if (s3_uri.rfind(pfx, 0) != 0)
        throw std::invalid_argument("URI não é s3://: " + s3_uri);
    auto slash = s3_uri.find('/', pfx.size());
    if (slash == std::string::npos) return {};
    return s3_uri.substr(slash + 1);
}

bool S3ModelSource::is_s3_uri(const std::string& uri) const noexcept {
    return uri.rfind("s3://", 0) == 0;
}

std::string S3ModelSource::cache_path(const S3ModelEntry& entry) const {
    return (fs::path(config_.cache_dir) / entry.filename).string();
}

std::string S3ModelSource::etag_path(const S3ModelEntry& entry) const {
    return cache_path(entry) + ".etag";
}

// ============================================================================
// list() — apenas metadados, sem download
// ============================================================================

std::vector<S3ModelEntry> S3ModelSource::list() const {
    std::vector<S3ModelEntry> result;

    Aws::S3::Model::ListObjectsV2Request req;
    req.SetBucket(config_.bucket);
    if (!config_.prefix.empty())
        req.SetPrefix(config_.prefix);

    bool truncated = true;
    while (truncated) {
        auto outcome = client_->ListObjectsV2(req);
        if (!outcome.IsSuccess()) {
            const auto& err = outcome.GetError();
            throw std::runtime_error(
                "[S3] ListObjectsV2 falhou: " +
                err.GetExceptionName() + " — " + err.GetMessage());
        }

        const auto& res = outcome.GetResult();
        for (const auto& obj : res.GetContents()) {
            std::string key = obj.GetKey();
            if (!key.empty() && key.back() == '/') continue;  // "diretórios"

            fs::path    kp(key);
            std::string ext = kp.extension().string();
            if (!is_known_ext(ext)) continue;

            S3ModelEntry entry;
            entry.key        = key;
            entry.s3_uri     = make_uri(config_.bucket, key);
            entry.filename   = kp.filename().string();
            entry.extension  = ext;
            entry.etag       = obj.GetETag();
            entry.size_bytes = static_cast<int64_t>(obj.GetSize());
            result.push_back(std::move(entry));
        }

        truncated = res.GetIsTruncated();
        if (truncated)
            req.SetContinuationToken(res.GetNextContinuationToken());
    }

    return result;
}

// ============================================================================
// download() — on-demand, com cache condicional por ETag
// ============================================================================

std::string S3ModelSource::download(const S3ModelEntry& entry) const {
    std::string local  = cache_path(entry);
    std::string etag_f = etag_path(entry);

    // Cache hit: arquivo existe e ETag coincide
    if (fs::exists(local) && fs::exists(etag_f)) {
        std::ifstream ef(etag_f);
        std::string cached_etag((std::istreambuf_iterator<char>(ef)),
                                 std::istreambuf_iterator<char>());
        if (cached_etag == entry.etag) {
            std::cout << "[S3] Cache hit: " << entry.filename << std::endl;
            return local;
        }
    }

    std::cout << "[S3] Baixando: " << entry.key
              << "  (" << entry.size_bytes / 1024 << " KB) ..." << std::endl;

    Aws::S3::Model::GetObjectRequest req;
    req.SetBucket(config_.bucket);
    req.SetKey(entry.key);

    auto outcome = client_->GetObject(req);
    if (!outcome.IsSuccess()) {
        const auto& err = outcome.GetError();
        throw std::runtime_error(
            "[S3] GetObject falhou para '" + entry.key + "': " +
            err.GetExceptionName() + " — " + err.GetMessage());
    }

    // Escreve para arquivo temporário → rename atômico
    std::string tmp = local + ".tmp";
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open())
            throw std::runtime_error("[S3] Não foi possível criar: " + tmp);
        ofs << outcome.GetResult().GetBody().rdbuf();
    }
    fs::rename(tmp, local);

    // Persiste ETag para próximo hit de cache
    {
        std::ofstream ef(etag_f, std::ios::trunc);
        ef << entry.etag;
    }

    std::cout << "[S3] Download concluído: " << local << std::endl;
    return local;
}

// ============================================================================
// evict() — remove arquivo de cache após UnloadModel
// ============================================================================

void S3ModelSource::evict(const S3ModelEntry& entry) const noexcept {
    std::error_code ec;
    std::string local  = cache_path(entry);
    std::string etag_f = etag_path(entry);

    if (fs::exists(local, ec)) {
        fs::remove(local, ec);
        if (!ec)
            std::cout << "[S3] Cache removido: " << local << std::endl;
        else
            std::cerr << "[S3] Aviso: falha ao remover " << local
                      << ": " << ec.message() << std::endl;
    }
    fs::remove(etag_f, ec);  // silencioso
}

}  // namespace worker
}  // namespace mlinference