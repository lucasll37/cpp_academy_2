// asa::models
#include "asa/tutorial/AsaExample.hpp"

#include <iostream>

namespace asa_template
{
IMPLEMENT_SUBCLASS(AsaExample, "AsaExample")
EMPTY_SLOTTABLE(AsaExample)
EMPTY_COPYDATA(AsaExample)
EMPTY_DELETEDATA(AsaExample)

// -------------------------------------------------------------------
// Função auxiliar: calcula o número total de elementos de um tensor
// a partir do seu shape (ex: [1, 4] → 4 elementos).
//
// Dimensões dinâmicas são representadas como -1 no shape.
// Nesse caso, assumimos batch_size = 1 para fins de exemplo.
// -------------------------------------------------------------------
static std::size_t calcularTotalElementos(const std::vector<int64_t>& shape)
{
    std::size_t total = 1;
    for (int64_t dim : shape)
    {
        // Dimensão dinâmica (ex: batch): assume 1
        total *= (dim > 0) ? static_cast<std::size_t>(dim) : 1;
    }
    return total;
}

AsaExample::AsaExample()
{
    STANDARD_CONSTRUCTOR()

    std::cerr << "HERE!" << std::endl;

    // ---------------------------------------------------------------
    // 1. Conectar ao servidor de inferência
    // ---------------------------------------------------------------
    const std::string servidor = "localhost:50052";
    const std::string modelo_id = "PPO_seed_1975_3600000_steps";
    const std::string modelo_path = "./models/PPO_seed_1975_3600000_steps.onnx";

    mlinference::client::InferenceClient client(servidor);

    if (!client.connect())
    {
        std::cerr << "[AsaExample] Falha ao conectar em " << servidor << std::endl;
        return;
    }

    // ---------------------------------------------------------------
    // 2. Validar o modelo para descobrir os shapes de entrada e saída
    //
    //    validate_model() inspeciona o arquivo .onnx sem carregá-lo
    //    permanentemente. Retorna um ValidationResult com:
    //      - result.inputs  → lista de TensorSpec (nome, dtype, shape)
    //      - result.outputs → lista de TensorSpec (nome, dtype, shape)
    // ---------------------------------------------------------------
    auto validacao = client.validate_model(modelo_path);

    if (!validacao.valid)
    {
        std::cerr << "[AsaExample] Modelo inválido: " << validacao.error_message << std::endl;
        return;
    }

    std::cout << "[AsaExample] Modelo validado. Backend: " << validacao.backend << std::endl;

    // Exibe os tensores de entrada descobertos
    std::cout << "[AsaExample] Tensores de entrada:" << std::endl;
    for (const auto& tensor : validacao.inputs)
    {
        std::cout << "  - " << tensor.name << " [" << tensor.dtype << "] shape: [";
        for (std::size_t i = 0; i < tensor.shape.size(); ++i)
        {
            std::cout << tensor.shape[i];
            if (i + 1 < tensor.shape.size())
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // ---------------------------------------------------------------
    // 3. Carregar o modelo no servidor
    // ---------------------------------------------------------------
    if (!client.load_model(modelo_id, modelo_path))
    {
        std::cerr << "[AsaExample] Falha ao carregar o modelo" << std::endl;
        return;
    }

    // ---------------------------------------------------------------
    // 4. Construir o mapa de inputs automaticamente a partir do shape
    //
    //    Para cada tensor de entrada descoberto na validação:
    //      a) Calcula quantos floats são necessários (produto das dims)
    //      b) Preenche com zeros (substitua por dados reais em produção)
    // ---------------------------------------------------------------
    std::map<std::string, std::vector<float>> inputs;

    for (const auto& tensor : validacao.inputs)
    {
        std::size_t n_elementos = calcularTotalElementos(tensor.shape);
        inputs[tensor.name] = std::vector<float>(n_elementos, 0.0f);

        std::cout << "[AsaExample] Input '" << tensor.name
                  << "' gerado com " << n_elementos << " elemento(s)" << std::endl;
    }

    // ---------------------------------------------------------------
    // 5. Executar inferência
    // ---------------------------------------------------------------
    auto resultado = client.predict(modelo_id, inputs);

    if (resultado.success)
    {
        std::cout << "[AsaExample] Inferência concluída em "
                  << resultado.inference_time_ms << " ms" << std::endl;

        for (const auto& [nome, dados] : resultado.outputs)
        {
            std::cout << "  Saída '" << nome << "': ";
            for (float v : dados)
                std::cout << v << " ";
            std::cout << std::endl;
        }
    }
    else
    {
        std::cerr << "[AsaExample] Erro na inferência: "
                  << resultado.error_message << std::endl;
    }

    // // ---------------------------------------------------------------
    // // 6. Descarregar o modelo ao finalizar
    // // ---------------------------------------------------------------
    // client.unload_model(modelo_id);
}

} // namespace asa_template

// // asa::models
// #include "asa/tutorial/AsaExample.hpp"

// namespace asa_tutorial
// {
// IMPLEMENT_SUBCLASS(AsaExample, "AsaExample")
// EMPTY_SLOTTABLE(AsaExample)
// EMPTY_COPYDATA(AsaExample)
// EMPTY_DELETEDATA(AsaExample)

// AsaExample::AsaExample()
// {
//     STANDARD_CONSTRUCTOR()
// }

// } // namespace asa_tutorial
