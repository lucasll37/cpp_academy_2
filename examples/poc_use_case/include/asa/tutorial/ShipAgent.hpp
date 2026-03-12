#ifndef __asa_tutorial_ShipAgent_H__
#define __asa_tutorial_ShipAgent_H__

#include "asa-poc-miia/client/inference_client.hpp"
#include "asa/models/agent/AsaAgent.hpp"

namespace asa_tutorial
{
//------------------------------------------------------------------------------
// Class: ShipAgent
// Factory name: ShipAgent
// Description:
//------------------------------------------------------------------------------
class ShipAgent : public asa::models::AsaAgent
{
    DECLARE_SUBCLASS(ShipAgent, asa::models::AsaAgent)

public:
    ShipAgent();
    ShipAgent(bool);

    void updateState(const double dt) override;
    void reasoning(const double dt) override;
    void setAction(const double dt) override;

private:
    mlinference::client::InferenceClient* client;

    const std::string servidor{"localhost:50052"};
    const std::string modelo_id{"PPO_seed_1975_3600000_steps"};
    const std::string modelo_path{"./models/PPO_seed_1975_3600000_steps.onnx"};
    std::map<std::string, std::vector<float>> inputs;
    mlinference::client::ValidationResult validacao;
    mlinference::client::PredictionResult resultado;
    std::size_t n_elementos;
};


} // namespace asa_tutorial

#endif
