// =============================================================================
// s3_model_source.hpp — Fonte de modelos em bucket S3
// =============================================================================
//
// Responsabilidades:
//   • Listar objetos com extensão reconhecida (.onnx, .py) no bucket —
//     apenas metadados, sem download.
//   • Fazer download on-demand de um objeto para cache local, somente
//     quando LoadModel for chamado.
//   • Remover o arquivo de cache quando o modelo for descarregado (evict).
//
// Credenciais AWS (cadeia padrão do SDK):
//   1. Variáveis de ambiente (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
//   2. Perfis em ~/.aws/credentials
//   3. IAM Instance Profile (EC2 / ECS)
//
// Compatibilidade com MinIO / LocalStack:
//   Definir S3Config::endpoint_url e S3Config::use_path_style = true
//
// Dependência Conan:
//   aws-sdk-cpp/[>=1.11.0]   (componente: s3)
// =============================================================================

#pragma once

#include <memory>
#include <string>
#include <vector>

// Forward-declarations — não vaza headers do AWS SDK para os includers.
namespace Aws { namespace S3 { class S3Client; } }

namespace mlinference {
namespace worker {

// ---------------------------------------------------------------------------
// Configuração — preenchida no main() a partir dos argumentos CLI
// ---------------------------------------------------------------------------
struct S3Config {
    std::string bucket;
    std::string prefix;
    std::string region         = "us-east-1";
    std::string cache_dir      = "/tmp/asasamiia_s3_cache";
    std::string endpoint_url;       // vazio = AWS padrão
    bool        use_path_style = false;  // true para MinIO/LocalStack
};

// ---------------------------------------------------------------------------
// Metadados de um objeto S3 disponível (ainda não baixado)
// ---------------------------------------------------------------------------
struct S3ModelEntry {
    std::string s3_uri;        // "s3://bucket/key"  — identificador único
    std::string key;           // chave S3 completa
    std::string filename;      // basename: "resnet18.onnx"
    std::string extension;     // ".onnx" | ".py"
    std::string etag;          // ETag S3 (cache condicional)
    int64_t     size_bytes{0};
};

// ---------------------------------------------------------------------------
// S3ModelSource
// ---------------------------------------------------------------------------
class S3ModelSource {
public:
    explicit S3ModelSource(S3Config config);
    ~S3ModelSource();

    // Não copiável
    S3ModelSource(const S3ModelSource&)            = delete;
    S3ModelSource& operator=(const S3ModelSource&) = delete;

    // -----------------------------------------------------------------------
    // list()  — apenas metadados (ListObjectsV2), sem transferência de dados
    // -----------------------------------------------------------------------
    std::vector<S3ModelEntry> list() const;

    // -----------------------------------------------------------------------
    // download()  — faz GET do objeto para o cache local on-demand.
    // Cache condicional por ETag: reutiliza se o arquivo já existir com
    // o mesmo ETag.  Lança std::runtime_error em caso de falha.
    // Retorna o path local pronto para uso pelo InferenceEngine.
    // -----------------------------------------------------------------------
    std::string download(const S3ModelEntry& entry) const;

    // -----------------------------------------------------------------------
    // evict()  — remove arquivo de cache após UnloadModel.
    // Silencioso: não lança exceção se o arquivo não existir.
    // -----------------------------------------------------------------------
    void evict(const S3ModelEntry& entry) const noexcept;

    // -----------------------------------------------------------------------
    // Utilitários
    // -----------------------------------------------------------------------
    bool            is_s3_uri(const std::string& uri) const noexcept;
    bool            enabled()                          const noexcept { return !config_.bucket.empty(); }
    const S3Config& config()                           const noexcept { return config_; }

    static std::string key_from_uri(const std::string& s3_uri);
    static std::string make_uri(const std::string& bucket, const std::string& key);

private:
    S3Config config_;

    // RAII do AWS SDK — detalhes de implementação completamente opacos.
    // O tipo concreto é definido apenas em s3_model_source.cpp.
    struct SdkGuard;
    std::shared_ptr<SdkGuard>          sdk_guard_;
    std::shared_ptr<Aws::S3::S3Client> client_;

    std::shared_ptr<Aws::S3::S3Client> make_client() const;
    std::string cache_path(const S3ModelEntry& entry) const;
    std::string etag_path(const S3ModelEntry& entry)  const;
};

}  // namespace worker
}  // namespace mlinference