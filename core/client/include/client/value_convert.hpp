// =============================================================================
/// @file   value_convert.hpp
/// @brief  Conversão entre #client::Value / #client::Object e
///         `google::protobuf::Value` / `google::protobuf::Struct`.
///
/// @details
/// Este header define as quatro funções de conversão que formam a fronteira
/// *wire* do sistema AsaMiia:
///
/// | Função               | Direção                                       |
/// |----------------------|-----------------------------------------------|
/// | to_proto_value()     | `client::Value`  → `google::protobuf::Value`  |
/// | to_proto_struct()    | `client::Object` → `google::protobuf::Struct` |
/// | from_proto_value()   | `google::protobuf::Value`  → `client::Value`  |
/// | from_proto_struct()  | `google::protobuf::Struct` → `client::Object` |
///
/// As conversões são **recursivas** e preservam a estrutura arbitrariamente
/// aninhada do tipo #client::Value — escalares, arrays e objetos são
/// mapeados para os tipos equivalentes do `google.protobuf.Value`.
///
/// ### Onde é usado
/// - **GrpcClientBackend** — serializa inputs antes de enviar via gRPC e
///   desserializa os outputs recebidos.
/// - **WorkerServiceImpl** — desserializa inputs recebidos do stub e
///   serializa outputs antes de responder.
///
/// @note Este header **não faz parte da API pública** exposta via
///       `libasa_miia_client`.  Deve ser incluído apenas nas translation
///       units que precisam da conversão (backends gRPC e servidor).
///
/// ### Compatibilidade com protobuf v4 + GCC 13
/// `google::protobuf::Map` retorna iteradores de `MapPair<K,V>`, que **não**
/// é implicitamente convertível para `std::pair<const K, V>` nesta combinação
/// de compilador e biblioteca.  Por isso todos os loops sobre `protobuf::Map`
/// usam iteradores explícitos (`it->first` / `it->second`) em vez de
/// *structured bindings* ou construtores de `std::map` por range.
///
/// @see mlinference::client::Value
/// @see mlinference::client::Object
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_VALUE_CONVERT_HPP
#define ML_INFERENCE_VALUE_CONVERT_HPP

#include "client/inference_client.hpp"
#include <google/protobuf/struct.pb.h>

namespace mlinference {
namespace client {

// =============================================================================
// Declaração antecipada — necessária porque from_proto_value é recursivo
// =============================================================================

/// @cond INTERNAL
inline Value from_proto_value(const google::protobuf::Value& pv);
/// @endcond

// =============================================================================
// client::Value → google::protobuf::Value
// =============================================================================

/// @brief Converte um #Value para `google::protobuf::Value`.
///
/// @details
/// Usa `std::visit` sobre a variante interna de #Value para mapear cada
/// alternativa para o campo correspondente do protobuf:
///
/// | Tipo C++ (#Value)  | Campo protobuf           |
/// |--------------------|--------------------------|
/// | #Null              | `null_value`             |
/// | `double`           | `number_value`           |
/// | `bool`             | `bool_value`             |
/// | `std::string`      | `string_value`           |
/// | #Array             | `list_value`             |
/// | #Object            | `struct_value`           |
///
/// A conversão é recursiva para #Array e #Object: cada elemento/campo
/// é convertido chamando `to_proto_value()` novamente.
///
/// @param v  Valor a converter.
///
/// @return `google::protobuf::Value` equivalente.
inline google::protobuf::Value to_proto_value(const Value& v) {
    google::protobuf::Value pv;

    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, Null>) {
            pv.set_null_value(google::protobuf::NULL_VALUE);

        } else if constexpr (std::is_same_v<T, double>) {
            pv.set_number_value(arg);

        } else if constexpr (std::is_same_v<T, bool>) {
            pv.set_bool_value(arg);

        } else if constexpr (std::is_same_v<T, std::string>) {
            pv.set_string_value(arg);

        } else if constexpr (std::is_same_v<T, Array>) {
            auto* lv = pv.mutable_list_value();
            for (const auto& elem : arg)
                *lv->add_values() = to_proto_value(elem);

        } else if constexpr (std::is_same_v<T, Object>) {
            // Object = std::map<string, Value> — structured bindings são seguros
            // aqui pois é std::map, não protobuf::Map.
            auto* sv = pv.mutable_struct_value();
            for (const auto& kv : arg)
                (*sv->mutable_fields())[kv.first] = to_proto_value(kv.second);
        }
    }, v.data);

    return pv;
}

// =============================================================================
// client::Object → google::protobuf::Struct
// =============================================================================

/// @brief Converte um #Object para `google::protobuf::Struct`.
///
/// @details
/// Itera sobre o `std::map` e converte cada valor com to_proto_value().
/// É o ponto de entrada usado pelos backends gRPC ao serializar inputs
/// antes de enviar um `PredictRequest`.
///
/// @code
/// // GrpcClientBackend::predict():
/// auto proto_inputs = to_proto_struct(inputs);
/// *request.mutable_inputs() = std::move(proto_inputs);
/// @endcode
///
/// @param obj  Mapa de entradas a serializar.
///
/// @return `google::protobuf::Struct` com todos os campos convertidos.
inline google::protobuf::Struct to_proto_struct(const Object& obj) {
    google::protobuf::Struct s;
    for (const auto& kv : obj)
        (*s.mutable_fields())[kv.first] = to_proto_value(kv.second);
    return s;
}

// =============================================================================
// google::protobuf::Value → client::Value
// =============================================================================

/// @brief Converte um `google::protobuf::Value` para #Value.
///
/// @details
/// Despacha pelo `kind_case()` do protobuf e reconstrói o tipo C++
/// correspondente:
///
/// | Campo protobuf   | Tipo C++ resultante |
/// |------------------|---------------------|
/// | `kNullValue`     | #Value{}  (#Null)   |
/// | `kNumberValue`   | `Value{double}`     |
/// | `kBoolValue`     | `Value{bool}`       |
/// | `kStringValue`   | `Value{string}`     |
/// | `kListValue`     | `Value{`#Array`}`   |
/// | `kStructValue`   | `Value{`#Object`}`  |
/// | `default`        | #Value{}  (#Null)   |
///
/// A conversão de `kListValue` e `kStructValue` é recursiva.
///
/// @note Para `kStructValue`, os campos do `protobuf::Map` são iterados com
///       iteradores explícitos por compatibilidade com GCC 13 + protobuf v4
///       (ver nota de compatibilidade no `@file`).
///
/// @param pv  Valor protobuf a converter.
///
/// @return #Value equivalente.  Tipos desconhecidos resultam em #Null.
inline Value from_proto_value(const google::protobuf::Value& pv) {
    switch (pv.kind_case()) {

        case google::protobuf::Value::kNullValue:
            return Value{};

        case google::protobuf::Value::kNumberValue:
            return Value{pv.number_value()};

        case google::protobuf::Value::kBoolValue:
            return Value{pv.bool_value()};

        case google::protobuf::Value::kStringValue:
            return Value{pv.string_value()};

        case google::protobuf::Value::kListValue: {
            Array arr;
            arr.reserve(static_cast<size_t>(pv.list_value().values_size()));
            for (int i = 0; i < pv.list_value().values_size(); ++i)
                arr.push_back(from_proto_value(pv.list_value().values(i)));
            return Value{std::move(arr)};
        }

        case google::protobuf::Value::kStructValue: {
            // Iteradores explícitos — MapPair não é convertível para std::pair
            // em GCC 13 + protobuf v4; structured bindings falhariam aqui.
            Object obj;
            const auto& fields = pv.struct_value().fields();
            for (auto it = fields.begin(); it != fields.end(); ++it)
                obj[it->first] = from_proto_value(it->second);
            return Value{std::move(obj)};
        }

        default:
            return Value{};
    }
}

// =============================================================================
// google::protobuf::Struct → client::Object
// =============================================================================

/// @brief Converte um `google::protobuf::Struct` para #Object.
///
/// @details
/// Ponto de entrada usado pelos backends gRPC ao desserializar inputs
/// recebidos e pelo `WorkerServiceImpl` ao desserializar outputs antes
/// de retornar ao cliente:
///
/// @code
/// // WorkerServiceImpl::Predict():
/// client::Object inputs = from_proto_struct(request->inputs());
/// auto result = engine_->predict(model_id, inputs);
///
/// // GrpcClientBackend::predict():
/// client::Object outputs = from_proto_struct(resp.outputs());
/// @endcode
///
/// @note Usa iteradores explícitos sobre `protobuf::Map` pela mesma razão
///       documentada em from_proto_value() — compatibilidade GCC 13 + protobuf v4.
///
/// @param s  Struct protobuf a desserializar.
///
/// @return #Object com todos os campos convertidos recursivamente.
inline Object from_proto_struct(const google::protobuf::Struct& s) {
    Object obj;
    const auto& fields = s.fields();
    for (auto it = fields.begin(); it != fields.end(); ++it)
        obj[it->first] = from_proto_value(it->second);
    return obj;
}

}  // namespace client
}  // namespace mlinference

#endif  // ML_INFERENCE_VALUE_CONVERT_HPP