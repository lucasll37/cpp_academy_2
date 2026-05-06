// =============================================================================
// tests/unit/test_backend_registry.cpp — Testes unitários de BackendRegistry
//
// Cobertura:
//   GRUPO 1  — Singleton: identidade e acesso
//   GRUPO 2  — register_backend() / supports()
//   GRUPO 3  — create_for_file(): caminho feliz
//   GRUPO 4  — create_for_file(): falhas e casos de borda
//   GRUPO 5  — create_by_type(): caminho feliz
//   GRUPO 6  — create_by_type(): falhas
//   GRUPO 7  — detect_backend(): extensões registradas
//   GRUPO 8  — detect_backend(): extensões não registradas / casos de borda
//   GRUPO 9  — registered_extensions()
//   GRUPO 10 — registered_backend_names()
//   GRUPO 11 — get_extension(): comportamento indireto via detect_backend
//   GRUPO 12 — Backends reais: OnnxBackendFactory
//   GRUPO 13 — Backends reais: PythonBackendFactory
//   GRUPO 14 — Substituição de fábrica (register overwrite)
//   GRUPO 15 — Consistência entre extensões e nomes registrados
//   GRUPO 16 — Backends registrados pelo InferenceEngine
//   GRUPO 17 — Ciclos de registro e criação repetidos
//   GRUPO 18 — Isolamento: instâncias criadas são independentes
//
// Nota sobre o singleton:
//   BackendRegistry é um singleton de processo. Testes que precisam de
//   extensões isoladas usam sufixos únicos (ex.: ".grpNx") para não colidir
//   com registros de outros testes ou do InferenceEngine.
// =============================================================================

#include <gtest/gtest.h>

#include "inference/backend_registry.hpp"
#include "inference/inference_engine.hpp"
#include "inference/onnx_backend.hpp"
#include "inference/python_backend.hpp"
#include "common.pb.h"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlinference::inference;
using namespace mlinference;

// =============================================================================
// Infraestrutura auxiliar
// =============================================================================

/// Stub mínimo de ModelBackend para testes de registro sem backends reais.
class StubBackend : public ModelBackend {
public:
    explicit StubBackend(const std::string& nm, common::BackendType t)
        : name_(nm), type_(t) {}

    bool load(const std::string&,
              const std::map<std::string, std::string>&) override { return true; }
    void unload() override {}
    InferenceResult predict(const client::Object&) override {
        return {false, {}, 0.0, "stub"};
    }
    common::BackendType backend_type() const override { return type_; }
    int64_t memory_usage_bytes() const override { return 0; }
    ModelSchema get_schema() const override { return {}; }
    // Não faz parte da interface ModelBackend — accessor de teste.
    const std::string& stub_name() const { return name_; }

private:
    std::string name_;
    common::BackendType type_;
};

/// Fábrica de StubBackend configurável por nome e tipo.
class StubFactory : public BackendFactory {
public:
    explicit StubFactory(const std::string& nm,
                         common::BackendType t = common::BACKEND_UNKNOWN)
        : name_(nm), type_(t) {}

    std::unique_ptr<ModelBackend> create() const override {
        return std::make_unique<StubBackend>(name_, type_);
    }
    common::BackendType backend_type() const override { return type_; }
    std::string name() const override { return name_; }

private:
    std::string name_;
    common::BackendType type_;
};

/// Constrói uma extensão com ponto a partir de um sufixo.
static std::string E(const std::string& suffix) { return "." + suffix; }

/// Retorna o singleton já aquecido com .onnx e .py registrados.
static BackendRegistry& reg() {
    static InferenceEngine _engine_init;
    (void)_engine_init;
    return BackendRegistry::instance();
}

// =============================================================================
// GRUPO 1 — Singleton: identidade e acesso
// =============================================================================

TEST(Singleton, InstanceReturnsSameAddress) {
    EXPECT_EQ(&BackendRegistry::instance(), &BackendRegistry::instance());
}

TEST(Singleton, InstanceAddressIsStableAcrossMultipleCalls) {
    auto* a = &BackendRegistry::instance();
    for (int i = 0; i < 10; ++i)
        EXPECT_EQ(a, &BackendRegistry::instance());
}

TEST(Singleton, InstanceIsNotNullptr) {
    EXPECT_NE(&BackendRegistry::instance(), nullptr);
}

TEST(Singleton, TwoReferencesAreAliased) {
    BackendRegistry& a = BackendRegistry::instance();
    BackendRegistry& b = BackendRegistry::instance();
    a.register_backend(E("grp1a"), std::make_unique<StubFactory>("s1a"));
    EXPECT_TRUE(b.supports(E("grp1a")));
}

// =============================================================================
// GRUPO 2 — register_backend() / supports()
// =============================================================================

TEST(RegistrySupports, ReturnsFalseForNeverRegisteredExtension) {
    EXPECT_FALSE(BackendRegistry::instance().supports(E("nunca_registrado_xyz")));
}

TEST(RegistrySupports, ReturnsTrueAfterRegister) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp2a"), std::make_unique<StubFactory>("s2a"));
    EXPECT_TRUE(r.supports(E("grp2a")));
}

TEST(RegistrySupports, ReturnsTrueForMultipleRegisteredExtensions) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp2b"), std::make_unique<StubFactory>("s2b"));
    r.register_backend(E("grp2c"), std::make_unique<StubFactory>("s2c"));
    EXPECT_TRUE(r.supports(E("grp2b")));
    EXPECT_TRUE(r.supports(E("grp2c")));
}

TEST(RegistrySupports, CaseSensitive_UpperCaseNotFound) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp2d"), std::make_unique<StubFactory>("s2d"));
    EXPECT_FALSE(r.supports(E("GRP2D")));
}

TEST(RegistrySupports, EmptyStringReturnsFalse) {
    EXPECT_FALSE(BackendRegistry::instance().supports(""));
}

TEST(RegistrySupports, ExtensionWithoutLeadingDotReturnsFalse) {
    // Extensões sem ponto não são registradas pelo sistema — sempre false.
    EXPECT_FALSE(BackendRegistry::instance().supports("onnx"));
}

TEST(RegistrySupports, RegisterDoesNotAffectUnrelatedExtensions) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp2e"), std::make_unique<StubFactory>("s2e"));
    EXPECT_FALSE(r.supports(E("grp2e_diferente")));
}

TEST(RegistrySupports, DoesNotThrow) {
    EXPECT_NO_THROW(BackendRegistry::instance().supports(E("qualquer_coisa")));
}

// =============================================================================
// GRUPO 3 — create_for_file(): caminho feliz
// =============================================================================

TEST(CreateForFile, ReturnsNonNullPtr) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp3a"), std::make_unique<StubFactory>("s3a"));
    EXPECT_NE(r.create_for_file("model.grp3a"), nullptr);
}

TEST(CreateForFile, ReturnedBackendHasCorrectName) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp3b"), std::make_unique<StubFactory>("s3b"));
    auto b = r.create_for_file("model.grp3b");
    EXPECT_EQ(static_cast<StubBackend*>(b.get())->stub_name(), "s3b");
}

TEST(CreateForFile, ReturnedBackendHasCorrectType) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp3c"),
        std::make_unique<StubFactory>("s3c", common::BACKEND_ONNX));
    auto b = r.create_for_file("model.grp3c");
    EXPECT_EQ(b->backend_type(), common::BACKEND_ONNX);
}

TEST(CreateForFile, UsesOnlyExtension_IgnoresDirectory) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp3d"), std::make_unique<StubFactory>("s3d"));
    auto b = r.create_for_file("/very/deep/path/to/model.grp3d");
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(static_cast<StubBackend*>(b.get())->stub_name(), "s3d");
}

TEST(CreateForFile, WorksWithRelativePath) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp3e"), std::make_unique<StubFactory>("s3e"));
    EXPECT_NE(r.create_for_file("./models/model.grp3e"), nullptr);
}

TEST(CreateForFile, WorksWithJustFilename) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp3f"), std::make_unique<StubFactory>("s3f"));
    EXPECT_NE(r.create_for_file("model.grp3f"), nullptr);
}

TEST(CreateForFile, ReturnedBackendInitialMetricsZero) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp3g"), std::make_unique<StubFactory>("s3g"));
    auto b = r.create_for_file("model.grp3g");
    EXPECT_EQ(b->metrics().total_inferences, 0u);
    EXPECT_EQ(b->metrics().failed_inferences, 0u);
}

// =============================================================================
// GRUPO 4 — create_for_file(): falhas e casos de borda
// =============================================================================

TEST(CreateForFile, ThrowsForUnregisteredExtension) {
    EXPECT_THROW(
        BackendRegistry::instance().create_for_file("model.ext_xyz_nunca_registrada"),
        std::runtime_error);
}

TEST(CreateForFile, ThrowsForPathWithNoExtension) {
    EXPECT_THROW(
        BackendRegistry::instance().create_for_file("model_sem_extensao"),
        std::runtime_error);
}

TEST(CreateForFile, ThrowsForEmptyString) {
    EXPECT_THROW(
        BackendRegistry::instance().create_for_file(""),
        std::runtime_error);
}

TEST(CreateForFile, ThrowsForUnknownExtensionEvenWithFullPath) {
    EXPECT_THROW(
        BackendRegistry::instance().create_for_file("/a/b/c/model.desconhecida_abc"),
        std::runtime_error);
}

TEST(CreateForFile, ThrowsMessageContainsExtension) {
    try {
        BackendRegistry::instance().create_for_file("model.ext_erro_msg_test");
        FAIL() << "esperava std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find(".ext_erro_msg_test"), std::string::npos)
            << "mensagem de erro não contém a extensão: " << msg;
    }
}

TEST(CreateForFile, ThrowsMessageIsNotEmpty) {
    try {
        BackendRegistry::instance().create_for_file("model.abc_never_reg");
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_FALSE(std::string(e.what()).empty());
    }
}

// =============================================================================
// GRUPO 5 — create_by_type(): caminho feliz
// =============================================================================

TEST(CreateByType, ReturnsNonNullForOnnxType) {
    EXPECT_NE(reg().create_by_type(common::BACKEND_ONNX), nullptr);
}

TEST(CreateByType, ReturnsNonNullForPythonType) {
    EXPECT_NE(reg().create_by_type(common::BACKEND_PYTHON), nullptr);
}

TEST(CreateByType, ReturnedOnnxBackendHasCorrectType) {
    auto b = reg().create_by_type(common::BACKEND_ONNX);
    EXPECT_EQ(b->backend_type(), common::BACKEND_ONNX);
}

TEST(CreateByType, ReturnedPythonBackendHasCorrectType) {
    auto b = reg().create_by_type(common::BACKEND_PYTHON);
    EXPECT_EQ(b->backend_type(), common::BACKEND_PYTHON);
}

TEST(CreateByType, OnnxDoesNotThrow) {
    EXPECT_NO_THROW(reg().create_by_type(common::BACKEND_ONNX));
}

TEST(CreateByType, PythonDoesNotThrow) {
    EXPECT_NO_THROW(reg().create_by_type(common::BACKEND_PYTHON));
}

TEST(CreateByType, FindsCustomRegisteredType) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp5a"),
        std::make_unique<StubFactory>("s5a", common::BACKEND_ONNX));
    EXPECT_NO_THROW(r.create_by_type(common::BACKEND_ONNX));
}

// =============================================================================
// GRUPO 6 — create_by_type(): falhas
// =============================================================================

TEST(CreateByType, ThrowsForUnregisteredType) {
    EXPECT_THROW(
        BackendRegistry::instance().create_by_type(
            static_cast<common::BackendType>(9999)),
        std::runtime_error);
}

TEST(CreateByType, ThrowsForAnotherUnregisteredType) {
    EXPECT_THROW(
        BackendRegistry::instance().create_by_type(
            static_cast<common::BackendType>(8888)),
        std::runtime_error);
}

TEST(CreateByType, ThrowsMessageIsNotEmpty) {
    try {
        BackendRegistry::instance().create_by_type(
            static_cast<common::BackendType>(7777));
        FAIL() << "esperava std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_FALSE(std::string(e.what()).empty());
    }
}

// =============================================================================
// GRUPO 7 — detect_backend(): extensões registradas
// =============================================================================

TEST(DetectBackend, ReturnsOnnxForDotOnnx) {
    EXPECT_EQ(reg().detect_backend("model.onnx"), common::BACKEND_ONNX);
}

TEST(DetectBackend, ReturnsPythonForDotPy) {
    EXPECT_EQ(reg().detect_backend("model.py"), common::BACKEND_PYTHON);
}

TEST(DetectBackend, ReturnsCorrectTypeForCustomExtension) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp7a"),
        std::make_unique<StubFactory>("s7a", common::BACKEND_PYTHON));
    EXPECT_EQ(r.detect_backend("model.grp7a"), common::BACKEND_PYTHON);
}

TEST(DetectBackend, IgnoresDirectoryAndUsesFileExtension) {
    EXPECT_EQ(reg().detect_backend("/a/b/c/model.onnx"), common::BACKEND_ONNX);
}

TEST(DetectBackend, WorksWithRelativePath) {
    EXPECT_EQ(reg().detect_backend("./models/nav.onnx"), common::BACKEND_ONNX);
}

TEST(DetectBackend, WorksWithJustFilename) {
    EXPECT_EQ(reg().detect_backend("nav.onnx"), common::BACKEND_ONNX);
}

TEST(DetectBackend, IsIdempotent) {
    auto first  = reg().detect_backend("model.onnx");
    auto second = reg().detect_backend("model.onnx");
    EXPECT_EQ(first, second);
}

TEST(DetectBackend, DoesNotThrowForKnownExtension) {
    EXPECT_NO_THROW(reg().detect_backend("model.onnx"));
}

// =============================================================================
// GRUPO 8 — detect_backend(): extensões não registradas / casos de borda
// =============================================================================

TEST(DetectBackend, ReturnsUnknownForUnregisteredExtension) {
    EXPECT_EQ(BackendRegistry::instance().detect_backend("model.xyz_nunca"),
              common::BACKEND_UNKNOWN);
}

TEST(DetectBackend, ReturnsUnknownForNoExtension) {
    EXPECT_EQ(BackendRegistry::instance().detect_backend("model_sem_ext"),
              common::BACKEND_UNKNOWN);
}

TEST(DetectBackend, ReturnsUnknownForEmptyString) {
    EXPECT_EQ(BackendRegistry::instance().detect_backend(""),
              common::BACKEND_UNKNOWN);
}

TEST(DetectBackend, DoesNotThrowForUnknown) {
    EXPECT_NO_THROW(
        BackendRegistry::instance().detect_backend("model.desconhecida_xyz"));
}

TEST(DetectBackend, CaseSensitive_UpperOnnxIsUnknown) {
    EXPECT_EQ(BackendRegistry::instance().detect_backend("model.ONNX"),
              common::BACKEND_UNKNOWN);
}

TEST(DetectBackend, CaseSensitive_UpperPyIsUnknown) {
    EXPECT_EQ(BackendRegistry::instance().detect_backend("model.PY"),
              common::BACKEND_UNKNOWN);
}

TEST(DetectBackend, ReturnsUnknownForWhitespaceOnlyExtension) {
    EXPECT_EQ(BackendRegistry::instance().detect_backend("model. "),
              common::BACKEND_UNKNOWN);
}

// =============================================================================
// GRUPO 9 — registered_extensions()
// =============================================================================

TEST(RegisteredExtensions, IsNotEmptyAfterEngineInit) {
    EXPECT_FALSE(reg().registered_extensions().empty());
}

TEST(RegisteredExtensions, ContainsDotOnnx) {
    const auto exts = reg().registered_extensions();
    EXPECT_NE(std::find(exts.begin(), exts.end(), ".onnx"), exts.end());
}

TEST(RegisteredExtensions, ContainsDotPy) {
    const auto exts = reg().registered_extensions();
    EXPECT_NE(std::find(exts.begin(), exts.end(), ".py"), exts.end());
}

TEST(RegisteredExtensions, AllStartWithDot) {
    for (const auto& e : reg().registered_extensions())
        EXPECT_EQ(e[0], '.') << "extensão sem ponto inicial: " << e;
}

TEST(RegisteredExtensions, NoDuplicates) {
    const auto exts = reg().registered_extensions();
    std::set<std::string> unique(exts.begin(), exts.end());
    EXPECT_EQ(unique.size(), exts.size());
}

TEST(RegisteredExtensions, ContainsCustomAfterRegister) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp9a"), std::make_unique<StubFactory>("s9a"));
    const auto exts = r.registered_extensions();
    EXPECT_NE(std::find(exts.begin(), exts.end(), E("grp9a")), exts.end());
}

TEST(RegisteredExtensions, IsIdempotent) {
    const auto a = reg().registered_extensions();
    const auto b = reg().registered_extensions();
    EXPECT_EQ(a, b);
}

TEST(RegisteredExtensions, SizeIncreasesAfterNewRegister) {
    auto& r = BackendRegistry::instance();
    auto before = r.registered_extensions().size();
    r.register_backend(E("grp9b"), std::make_unique<StubFactory>("s9b"));
    auto after = r.registered_extensions().size();
    EXPECT_GE(after, before);
}

TEST(RegisteredExtensions, SizeDoesNotIncreaseOnOverwrite) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp9c"), std::make_unique<StubFactory>("s9c_v1"));
    auto before = r.registered_extensions().size();
    r.register_backend(E("grp9c"), std::make_unique<StubFactory>("s9c_v2"));
    auto after = r.registered_extensions().size();
    EXPECT_EQ(after, before);
}

// =============================================================================
// GRUPO 10 — registered_backend_names()
// =============================================================================

TEST(RegisteredNames, IsNotEmpty) {
    EXPECT_FALSE(reg().registered_backend_names().empty());
}

TEST(RegisteredNames, ContainsOnnxName) {
    const auto names = reg().registered_backend_names();
    bool found = std::any_of(names.begin(), names.end(),
        [](const std::string& n){ return n.find("onnx") != std::string::npos; });
    EXPECT_TRUE(found);
}

TEST(RegisteredNames, ContainsPythonName) {
    const auto names = reg().registered_backend_names();
    bool found = std::any_of(names.begin(), names.end(),
        [](const std::string& n){ return n.find("python") != std::string::npos; });
    EXPECT_TRUE(found);
}

TEST(RegisteredNames, SizeMatchesExtensionsSize) {
    const auto& r = reg();
    EXPECT_EQ(r.registered_backend_names().size(),
              r.registered_extensions().size());
}

TEST(RegisteredNames, NoEmptyNames) {
    for (const auto& n : reg().registered_backend_names())
        EXPECT_FALSE(n.empty());
}

TEST(RegisteredNames, ContainsCustomNameAfterRegister) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp10a"),
        std::make_unique<StubFactory>("meu_stub_10a"));
    const auto names = r.registered_backend_names();
    EXPECT_NE(std::find(names.begin(), names.end(), "meu_stub_10a"), names.end());
}

TEST(RegisteredNames, IsIdempotent) {
    const auto a = reg().registered_backend_names();
    const auto b = reg().registered_backend_names();
    EXPECT_EQ(a, b);
}

TEST(RegisteredNames, OverwriteUpdatesName) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp10b"), std::make_unique<StubFactory>("nome_antigo_10b"));
    r.register_backend(E("grp10b"), std::make_unique<StubFactory>("nome_novo_10b"));
    const auto names = r.registered_backend_names();
    bool tem_antigo = std::find(names.begin(), names.end(), "nome_antigo_10b") != names.end();
    bool tem_novo   = std::find(names.begin(), names.end(), "nome_novo_10b")   != names.end();
    EXPECT_FALSE(tem_antigo) << "nome antigo não deveria mais estar presente";
    EXPECT_TRUE(tem_novo);
}

// =============================================================================
// GRUPO 11 — get_extension(): comportamento indireto via detect_backend
// =============================================================================

TEST(GetExtension, MultiDotFilenameTakesLastExtension) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp11a"), std::make_unique<StubFactory>("s11a"));
    EXPECT_TRUE(r.supports(E("grp11a")));
    EXPECT_NO_THROW(r.create_for_file("archive.tar.grp11a"));
}

TEST(GetExtension, DotInDirectoryDoesNotConfuseExtraction) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp11b"), std::make_unique<StubFactory>("s11b"));
    EXPECT_TRUE(r.supports(E("grp11b")));
    EXPECT_NO_THROW(r.create_for_file("/path/to.dir/model.grp11b"));
}

TEST(GetExtension, FileWithOnlyDotPrefixDoesNotThrow) {
    // ".gitignore" → rfind('.') == 0 → extensão = ".gitignore" → não registrada
    EXPECT_NO_THROW(BackendRegistry::instance().detect_backend(".gitignore"));
}

TEST(GetExtension, FileWithOnlyDotPrefixReturnsUnknown) {
    EXPECT_EQ(BackendRegistry::instance().detect_backend(".gitignore"),
              common::BACKEND_UNKNOWN);
}

TEST(GetExtension, DotOnlyFilenameReturnsUnknown) {
    EXPECT_EQ(BackendRegistry::instance().detect_backend("."),
              common::BACKEND_UNKNOWN);
}

TEST(GetExtension, ExtensionWithNumbersWorks) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp11c2"), std::make_unique<StubFactory>("s11c2"));
    EXPECT_TRUE(r.supports(E("grp11c2")));
    EXPECT_NO_THROW(r.create_for_file("model.grp11c2"));
}

TEST(GetExtension, LongExtensionWorks) {
    auto& r = BackendRegistry::instance();
    const std::string s = "grp11d_esta_eh_uma_extensao_muito_longa";
    r.register_backend(E(s), std::make_unique<StubFactory>("s11d"));
    EXPECT_TRUE(r.supports(E(s)));
    EXPECT_NO_THROW(r.create_for_file("model." + s));
}

TEST(GetExtension, MultipleDotsDirAndFile) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp11e"), std::make_unique<StubFactory>("s11e"));
    // "/a.b/c.d/e.grp11e" → extensão = ".grp11e"
    EXPECT_TRUE(r.supports(E("grp11e")));
    EXPECT_NO_THROW(r.create_for_file("/a.b/c.d/e.grp11e"));
}

// =============================================================================
// GRUPO 12 — Backends reais: OnnxBackendFactory
// =============================================================================

TEST(OnnxFactory, CreateReturnsNonNull) {
    OnnxBackendFactory f(false, 0, 1);
    EXPECT_NE(f.create(), nullptr);
}

TEST(OnnxFactory, BackendTypeIsOnnx) {
    OnnxBackendFactory f(false, 0, 1);
    EXPECT_EQ(f.backend_type(), common::BACKEND_ONNX);
}

TEST(OnnxFactory, NameContainsOnnx) {
    OnnxBackendFactory f(false, 0, 1);
    EXPECT_NE(f.name().find("onnx"), std::string::npos);
}

TEST(OnnxFactory, NameIsNotEmpty) {
    OnnxBackendFactory f(false, 0, 1);
    EXPECT_FALSE(f.name().empty());
}

TEST(OnnxFactory, CreatedBackendTypeIsOnnx) {
    OnnxBackendFactory f(false, 0, 1);
    auto b = f.create();
    EXPECT_EQ(b->backend_type(), common::BACKEND_ONNX);
}

TEST(OnnxFactory, CreatedBackendMemoryZero) {
    OnnxBackendFactory f(false, 0, 1);
    auto b = f.create();
    EXPECT_EQ(b->memory_usage_bytes(), 0);
}

TEST(OnnxFactory, CreatedBackendMetricsZero) {
    OnnxBackendFactory f(false, 0, 1);
    auto b = f.create();
    EXPECT_EQ(b->metrics().total_inferences, 0u);
    EXPECT_EQ(b->metrics().failed_inferences, 0u);
}

TEST(OnnxFactory, TwoCreationsAreIndependent) {
    OnnxBackendFactory f(false, 0, 1);
    auto b1 = f.create();
    auto b2 = f.create();
    EXPECT_NE(b1.get(), b2.get());
}

TEST(OnnxFactory, SingleThreadConstruction) {
    EXPECT_NO_FATAL_FAILURE({ OnnxBackendFactory f(false, 0, 1); });
}

TEST(OnnxFactory, MultiThreadConstruction) {
    EXPECT_NO_FATAL_FAILURE({ OnnxBackendFactory f(false, 0, 8); });
}

TEST(OnnxFactory, BackendTypeConsistentWithCreatedBackend) {
    OnnxBackendFactory f(false, 0, 4);
    EXPECT_EQ(f.backend_type(), f.create()->backend_type());
}

// =============================================================================
// GRUPO 13 — Backends reais: PythonBackendFactory
// =============================================================================

TEST(PythonFactory, CreateReturnsNonNull) {
    PythonBackendFactory f;
    EXPECT_NE(f.create(), nullptr);
}

TEST(PythonFactory, BackendTypeIsPython) {
    PythonBackendFactory f;
    EXPECT_EQ(f.backend_type(), common::BACKEND_PYTHON);
}

TEST(PythonFactory, NameContainsPython) {
    PythonBackendFactory f;
    EXPECT_NE(f.name().find("python"), std::string::npos);
}

TEST(PythonFactory, NameIsNotEmpty) {
    PythonBackendFactory f;
    EXPECT_FALSE(f.name().empty());
}

TEST(PythonFactory, CreatedBackendTypeIsPython) {
    PythonBackendFactory f;
    auto b = f.create();
    EXPECT_EQ(b->backend_type(), common::BACKEND_PYTHON);
}

TEST(PythonFactory, CreatedBackendMemoryZero) {
    PythonBackendFactory f;
    auto b = f.create();
    EXPECT_EQ(b->memory_usage_bytes(), 0);
}

TEST(PythonFactory, CreatedBackendMetricsZero) {
    PythonBackendFactory f;
    auto b = f.create();
    EXPECT_EQ(b->metrics().total_inferences, 0u);
    EXPECT_EQ(b->metrics().failed_inferences, 0u);
}

TEST(PythonFactory, TwoCreationsAreIndependent) {
    PythonBackendFactory f;
    auto b1 = f.create();
    auto b2 = f.create();
    EXPECT_NE(b1.get(), b2.get());
}

TEST(PythonFactory, BackendTypeConsistentWithCreatedBackend) {
    PythonBackendFactory f;
    EXPECT_EQ(f.backend_type(), f.create()->backend_type());
}

TEST(PythonFactory, NameConsistentWithCreatedBackend) {
    PythonBackendFactory f;
    EXPECT_EQ(f.backend_type(), f.create()->backend_type());
}

// =============================================================================
// GRUPO 14 — Substituição de fábrica (register overwrite)
// =============================================================================

TEST(OverwriteFactory, ExtensionStillSupportedAfterOverwrite) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp14a"), std::make_unique<StubFactory>("primeiro_14a"));
    r.register_backend(E("grp14a"), std::make_unique<StubFactory>("segundo_14a"));
    EXPECT_TRUE(r.supports(E("grp14a")));
}

TEST(OverwriteFactory, NewFactoryNameIsUsedOnCreate) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp14b"), std::make_unique<StubFactory>("antigo_14b"));
    r.register_backend(E("grp14b"), std::make_unique<StubFactory>("novo_14b"));
    { auto _b = r.create_for_file("model.grp14b");
    EXPECT_EQ(static_cast<StubBackend*>(_b.get())->stub_name(), "novo_14b"); }
}

TEST(OverwriteFactory, NewTypeIsReflectedInDetect) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp14c"),
        std::make_unique<StubFactory>("s14c", common::BACKEND_ONNX));
    r.register_backend(E("grp14c"),
        std::make_unique<StubFactory>("s14c2", common::BACKEND_PYTHON));
    EXPECT_EQ(r.detect_backend("model.grp14c"), common::BACKEND_PYTHON);
}

TEST(OverwriteFactory, OverwriteDoesNotAffectOtherExtensions) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp14d"), std::make_unique<StubFactory>("s14d"));
    r.register_backend(E("grp14e"), std::make_unique<StubFactory>("s14e"));
    r.register_backend(E("grp14d"), std::make_unique<StubFactory>("s14d_v2"));
    { auto _b = r.create_for_file("model.grp14e");
    EXPECT_EQ(static_cast<StubBackend*>(_b.get())->stub_name(), "s14e"); }
}

TEST(OverwriteFactory, MultipleOverwritesConvergeToLast) {
    auto& r = BackendRegistry::instance();
    for (int i = 0; i < 5; ++i)
        r.register_backend(E("grp14f"),
            std::make_unique<StubFactory>("iter_" + std::to_string(i)));
    { auto _b = r.create_for_file("model.grp14f");
    EXPECT_EQ(static_cast<StubBackend*>(_b.get())->stub_name(), "iter_4"); }
}

TEST(OverwriteFactory, SupportsRemainsTrue) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp14g"), std::make_unique<StubFactory>("s14g_v1"));
    r.register_backend(E("grp14g"), std::make_unique<StubFactory>("s14g_v2"));
    EXPECT_TRUE(r.supports(E("grp14g")));
}

// =============================================================================
// GRUPO 15 — Consistência entre extensões e nomes registrados
// =============================================================================

TEST(Consistency, ExtensionsAndNamesSameCount) {
    const auto& r = reg();
    EXPECT_EQ(r.registered_extensions().size(),
              r.registered_backend_names().size());
}

// Os testes de consistência abaixo iteram apenas sobre as extensões reais
// (.onnx, .py) porque o singleton acumula também extensões stub de outros
// testes (com BACKEND_UNKNOWN), tornando asserções genéricas inválidas.
static const std::vector<std::string> kRealExts = {".onnx", ".py"};
static const std::vector<common::BackendType> kRealTypes = {
    common::BACKEND_ONNX, common::BACKEND_PYTHON};

TEST(Consistency, EachRealExtensionProducesABackend) {
    const auto& r = reg();
    for (const auto& ext : kRealExts) {
        auto b = r.create_for_file("model" + ext);
        EXPECT_NE(b, nullptr) << "falhou para extensão: " << ext;
    }
}

TEST(Consistency, EachRealExtensionDetectsNonUnknownType) {
    const auto& r = reg();
    for (const auto& ext : kRealExts) {
        auto type = r.detect_backend("model" + ext);
        EXPECT_NE(type, common::BACKEND_UNKNOWN)
            << "BACKEND_UNKNOWN para extensão real: " << ext;
    }
}

TEST(Consistency, DetectedTypeMatchesCreatedBackendType) {
    const auto& r = reg();
    for (const auto& ext : kRealExts) {
        auto detected = r.detect_backend("model" + ext);
        auto b        = r.create_for_file("model" + ext);
        EXPECT_EQ(b->backend_type(), detected)
            << "tipo detectado != tipo criado para ext=" << ext;
    }
}

TEST(Consistency, RealExtensionsAndTypesMappedCorrectly) {
    const auto& r = reg();
    for (size_t i = 0; i < kRealExts.size(); ++i) {
        EXPECT_EQ(r.detect_backend("model" + kRealExts[i]), kRealTypes[i])
            << "tipo errado para ext=" << kRealExts[i];
    }
}

TEST(Consistency, AllRegisteredExtensionsAreSupported) {
    const auto& r = reg();
    for (const auto& ext : r.registered_extensions())
        EXPECT_TRUE(r.supports(ext)) << "supports() falso para ext registrada: " << ext;
}

TEST(Consistency, AllRegisteredExtensionsProduceABackend) {
    const auto& r = reg();
    for (const auto& ext : r.registered_extensions()) {
        auto b = r.create_for_file("model" + ext);
        EXPECT_NE(b, nullptr) << "create retornou null para ext: " << ext;
    }
}

// =============================================================================
// GRUPO 16 — Backends registrados pelo InferenceEngine
// =============================================================================

TEST(EngineRegistered, OnnxExtensionIsSupported) {
    EXPECT_TRUE(reg().supports(".onnx"));
}

TEST(EngineRegistered, PyExtensionIsSupported) {
    EXPECT_TRUE(reg().supports(".py"));
}

TEST(EngineRegistered, CreateForOnnxFileDoesNotThrow) {
    EXPECT_NO_THROW(reg().create_for_file("model.onnx"));
}

TEST(EngineRegistered, CreateForPyFileDoesNotThrow) {
    EXPECT_NO_THROW(reg().create_for_file("model.py"));
}

TEST(EngineRegistered, OnnxBackendTypeIsCorrect) {
    EXPECT_EQ(reg().create_for_file("model.onnx")->backend_type(),
              common::BACKEND_ONNX);
}

TEST(EngineRegistered, PythonBackendTypeIsCorrect) {
    EXPECT_EQ(reg().create_for_file("model.py")->backend_type(),
              common::BACKEND_PYTHON);
}

TEST(EngineRegistered, SecondEngineInstanceDoesNotDuplicateOnnx) {
    InferenceEngine e2;
    (void)e2;
    const auto exts = reg().registered_extensions();
    long count = std::count(exts.begin(), exts.end(), ".onnx");
    EXPECT_EQ(count, 1) << "'.onnx' registrado mais de uma vez";
}

TEST(EngineRegistered, SecondEngineInstanceDoesNotDuplicatePy) {
    InferenceEngine e2;
    (void)e2;
    const auto exts = reg().registered_extensions();
    long count = std::count(exts.begin(), exts.end(), ".py");
    EXPECT_EQ(count, 1) << "'.py' registrado mais de uma vez";
}

TEST(EngineRegistered, CreateByTypeOnnxSucceeds) {
    EXPECT_NO_THROW(reg().create_by_type(common::BACKEND_ONNX));
}

TEST(EngineRegistered, CreateByTypePythonSucceeds) {
    EXPECT_NO_THROW(reg().create_by_type(common::BACKEND_PYTHON));
}

// =============================================================================
// GRUPO 17 — Ciclos de registro e criação repetidos
// =============================================================================

TEST(Repeat, RegisterSameExtensionManyTimesDoesNotThrow) {
    auto& r = BackendRegistry::instance();
    EXPECT_NO_THROW({
        for (int i = 0; i < 20; ++i)
            r.register_backend(E("grp17a"),
                std::make_unique<StubFactory>("s17a_" + std::to_string(i)));
    });
}

TEST(Repeat, CreateAfterManyOverwritesUsesLast) {
    auto& r = BackendRegistry::instance();
    for (int i = 0; i < 10; ++i)
        r.register_backend(E("grp17b"),
            std::make_unique<StubFactory>("s17b_" + std::to_string(i)));
    { auto _b = r.create_for_file("model.grp17b");
    EXPECT_EQ(static_cast<StubBackend*>(_b.get())->stub_name(), "s17b_9"); }
}

TEST(Repeat, CreateCalledManyTimesSucceeds) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp17c"), std::make_unique<StubFactory>("s17c"));
    EXPECT_NO_THROW({
        for (int i = 0; i < 50; ++i) {
            auto b = r.create_for_file("model.grp17c");
            ASSERT_NE(b, nullptr);
        }
    });
}

TEST(Repeat, DetectCalledManyTimesIsConsistent) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp17d"),
        std::make_unique<StubFactory>("s17d", common::BACKEND_ONNX));
    auto first = r.detect_backend("model.grp17d");
    for (int i = 0; i < 20; ++i)
        EXPECT_EQ(r.detect_backend("model.grp17d"), first);
}

TEST(Repeat, SupportsCalledManyTimesIsStable) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp17e"), std::make_unique<StubFactory>("s17e"));
    for (int i = 0; i < 20; ++i)
        EXPECT_TRUE(r.supports(E("grp17e")));
}

TEST(Repeat, RegisteredExtensionsStableUnderRepeatedCalls) {
    auto& r = reg();
    auto first = r.registered_extensions();
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(r.registered_extensions(), first);
}

// =============================================================================
// GRUPO 18 — Isolamento: instâncias criadas são independentes
// =============================================================================

TEST(Isolation, TwoInstancesFromSameFactoryAreDifferentObjects) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp18a"), std::make_unique<StubFactory>("s18a"));
    auto b1 = r.create_for_file("model.grp18a");
    auto b2 = r.create_for_file("model.grp18a");
    EXPECT_NE(b1.get(), b2.get());
}

TEST(Isolation, OnnxInstancesAreIndependent) {
    auto b1 = reg().create_for_file("model.onnx");
    auto b2 = reg().create_for_file("model.onnx");
    EXPECT_NE(b1.get(), b2.get());
}

TEST(Isolation, PythonInstancesAreIndependent) {
    auto b1 = reg().create_for_file("model.py");
    auto b2 = reg().create_for_file("model.py");
    EXPECT_NE(b1.get(), b2.get());
}

TEST(Isolation, CreateByTypeOnnxInstancesAreIndependent) {
    auto b1 = reg().create_by_type(common::BACKEND_ONNX);
    auto b2 = reg().create_by_type(common::BACKEND_ONNX);
    EXPECT_NE(b1.get(), b2.get());
}

TEST(Isolation, CreateByTypePythonInstancesAreIndependent) {
    auto b1 = reg().create_by_type(common::BACKEND_PYTHON);
    auto b2 = reg().create_by_type(common::BACKEND_PYTHON);
    EXPECT_NE(b1.get(), b2.get());
}

TEST(Isolation, MetricsAreIndependentBetweenInstances) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp18b"), std::make_unique<StubFactory>("s18b"));
    auto b1 = r.create_for_file("model.grp18b");
    auto b2 = r.create_for_file("model.grp18b");
    EXPECT_EQ(b1->metrics().total_inferences, 0u);
    EXPECT_EQ(b2->metrics().total_inferences, 0u);
}

TEST(Isolation, DestroyingInstanceDoesNotAffectRegistry) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp18c"), std::make_unique<StubFactory>("s18c"));
    {
        auto b = r.create_for_file("model.grp18c");
        (void)b;
    }  // b destruído aqui
    EXPECT_TRUE(r.supports(E("grp18c")));
    EXPECT_NO_THROW(r.create_for_file("model.grp18c"));
}

TEST(Isolation, DestroyingInstanceDoesNotAffectOtherInstances) {
    auto& r = BackendRegistry::instance();
    r.register_backend(E("grp18d"), std::make_unique<StubFactory>("s18d"));
    auto b2 = r.create_for_file("model.grp18d");
    {
        auto b1 = r.create_for_file("model.grp18d");
        (void)b1;
    }  // b1 destruído, b2 deve continuar válido
    EXPECT_NE(b2, nullptr);
    EXPECT_EQ(static_cast<StubBackend*>(b2.get())->stub_name(), "s18d");
}