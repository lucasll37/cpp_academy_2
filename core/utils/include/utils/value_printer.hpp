// =============================================================================
/// @file   value_printer.hpp
/// @brief  Serialização de mlinference::client::Value para string legível.
///
/// @details
/// Fornece a função utilitária #value_to_str, que converte recursivamente
/// qualquer `Value` (null, bool, number, string, array, object) em uma
/// representação textual indentada — equivalente a um pretty-print JSON.
///
/// Tipos suportados:
/// | Tipo    | Exemplo de saída  |
/// |---------|-------------------|
/// | null    | `null`            |
/// | bool    | `true` / `false`  |
/// | number  | `6000.000000`     |
/// | string  | `"pilotBT"`       |
/// | array   | `[\n    ...\n]`   |
/// | object  | `{\n    "k": v}`  |
///
/// ### Uso básico
/// @code
/// #include <utils/logger.hpp>
/// #include <utils/value_printer.hpp>
///
/// // Inspecionar estado serializado antes de enviar ao modelo
/// auto state = prepareState(pilot_state);
/// LOG_DEBUG("asa_pilot_bt") << "INPUTS:\n"
///     << value_to_str(mlinference::client::Value{state});
///
/// // Inspecionar saídas após predição
/// auto result = client->predict(model_id, state);
/// LOG_DEBUG("asa_pilot_bt") << "OUTPUTS:\n"
///     << value_to_str(mlinference::client::Value{result.outputs});
/// @endcode
///
/// ### Formato de saída
/// ```
/// {
///     "ownship": {
///         "valid": true,
///         "altitude": 6000.000000,
///         "heading": 270.000000
///     },
///     "air_tracks": [
///         {
///             "valid": true,
///             "range": 45000.000000
///         }
///     ]
/// }
/// ```
///
/// @note Thread-safe (sem estado global mutável).
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#pragma once

#include <miia/client/inference_client.hpp>
#include <string>

// =============================================================================
// value_to_str
// =============================================================================

/// @brief Converte um #mlinference::client::Value em string indentada.
///
/// @details
/// A conversão é recursiva: arrays e objects são expandidos com indentação
/// proporcional ao nível de aninhamento.  Cada nível adiciona 4 espaços.
///
/// A função é definida no escopo global para manter a mesma ergonomia de
/// uso das macros de logging — sem qualificação adicional:
/// @code
/// LOG_DEBUG("asa_pilot_bt") << value_to_str(mlinference::client::Value{state});
/// @endcode
///
/// @param v       Valor a serializar.
/// @param indent  Nível de indentação inicial (default: 0).
///
/// @return String com a representação textual do valor.
inline std::string value_to_str(const mlinference::client::Value& v, int indent = 0)
{
    const std::string tab(indent * 4, ' ');
    const std::string inner_tab((indent + 1) * 4, ' ');

    if (v.is_null())
        return "null";

    if (v.is_bool())
        return v.as_bool() ? "true" : "false";

    if (v.is_number())
        return std::to_string(v.as_number());

    if (v.is_string())
        return "\"" + v.as_string() + "\"";

    if (v.is_array())
    {
        const auto& arr = v.as_array();
        if (arr.empty())
            return "[]";

        std::string s = "[\n";
        for (size_t i = 0; i < arr.size(); ++i)
        {
            s += inner_tab + value_to_str(arr[i], indent + 1);
            if (i + 1 < arr.size())
                s += ",";
            s += "\n";
        }
        return s + tab + "]";
    }

    if (v.is_object())
    {
        const auto& obj = v.as_object();
        if (obj.empty())
            return "{}";

        std::string s = "{\n";
        size_t i = 0;
        for (const auto& [k, val] : obj)
        {
            s += inner_tab + "\"" + k + "\": " + value_to_str(val, indent + 1);
            if (++i < obj.size())
                s += ",";
            s += "\n";
        }
        return s + tab + "}";
    }

    return "?";
}