// mixr
#include "mixr/base/util/nav_utils.hpp"
#include "mixr/models/navigation/Navigation.hpp"
#include "mixr/models/navigation/Route.hpp"
#include "mixr/models/navigation/Steerpoint.hpp"

// asa::tutorial
#include "asa/tutorial/HazardSource.hpp"
#include "asa/tutorial/Hydrodynamics.hpp"
#include "asa/tutorial/Scanner.hpp"
#include "asa/tutorial/ShipAction.hpp"
#include "asa/tutorial/ShipAgent.hpp"
#include "asa/tutorial/ShipState.hpp"

namespace asa_tutorial
{
// clang-format off
IMPLEMENT_SUBCLASS(ShipAgent, "ShipAgent")
EMPTY_SLOTTABLE(ShipAgent)
EMPTY_COPYDATA(ShipAgent)
EMPTY_DELETEDATA(ShipAgent)

static std::size_t calcularTotalElementos(const std::vector<int64_t>& shape)
{
    std::size_t total = 1;
    for (int64_t dim : shape)
    {
        total *= (dim > 0) ? static_cast<std::size_t>(dim) : 1;
    }
    return total;
}


ShipAgent::ShipAgent() : AsaAgent(false)
{
    STANDARD_CONSTRUCTOR()
    
    action = new ShipAction();
    state = new ShipState();

    client = new mlinference::client::InferenceClient(servidor);
    client->connect();
    client->load_model(modelo_id, modelo_path);
    validacao = client->validate_model(modelo_path);

    for (const auto& tensor : validacao.inputs)
    {
        n_elementos = calcularTotalElementos(tensor.shape);
        inputs[tensor.name] = std::vector<float>(n_elementos, 0.0f);
    }
}

ShipAgent::ShipAgent(bool) : AsaAgent(false)
{
    STANDARD_CONSTRUCTOR()

    client = new mlinference::client::InferenceClient(servidor);
    client->connect();
    client->load_model(modelo_id, modelo_path);
    validacao = client->validate_model(modelo_path);

    for (const auto& tensor : validacao.inputs)
    {
        n_elementos = calcularTotalElementos(tensor.shape);
        inputs[tensor.name] = std::vector<float>(n_elementos, 0.0f);
    }

}

// ShipAgent::~ShipAgent() {
//     client->unload_model(modelo_id);
//     delete client;
// }

// clang-format off

//------------------------------------------------------------------------------
// Override functions
//------------------------------------------------------------------------------

void ShipAgent::updateState(const double dt)
{
    BaseClass::updateState(dt);

    auto state{dynamic_cast<ShipState*>(getState())};
    if (state != nullptr)
    {
        // Scanner
        auto scanner{dynamic_cast<Scanner*>(getOwnship()->getGimbal())};
        if (scanner != nullptr)
        {
            state->setHazardSources(scanner->getHazardSources());
            state->setHasValidScannerData(true);    
        }
        
        // Nav System
        auto nav{getOwnship()->getNavigation()};
        if (nav != nullptr)
        {
            auto route{nav->getPriRoute()};
            if (route != nullptr)
            {
                auto stpt{route->getSteerpoint()};
                if (stpt != nullptr)
                {
                    state->setHasValidNavData(stpt->isNavDataValid());
                    if (stpt->isNavDataValid())
                    {
                        state->setToHeading(stpt->getTrueBrgDeg());
                    }
                }
            }
        }
    }
}
//------------------------------------------------------------------------------

void ShipAgent::reasoning(const double dt)
{
    BaseClass::reasoning(dt);

    auto state{dynamic_cast<ShipState*>(getState())};
    auto action{dynamic_cast<ShipAction*>(getAction())};

    if (state == nullptr || !state->getHasValidNavData() || !state->getHasValidScannerData() || action == nullptr)
        return;
    
    // Resulting direction vector
    double xres{}, yres{};
    double hdgRes{};

    // First, the force of attraction in the direction of the selected steerpoint
    const double weightAttraction{1.0};
    xres = cos(state->getToHeading() * mixr::base::angle::D2RCC) * weightAttraction;
    yres = sin(state->getToHeading() * mixr::base::angle::D2RCC) * weightAttraction;

    // Now, the forces of repulsion from the hazard sources
    auto sources{*state->getHazardSources()};
    for (auto player : sources)
    {
        // exponent used in weighting function
        const double expParam{2.0};
        
        auto hs{dynamic_cast<HazardSource*>(player)};
        double safeDist{hs->getMinSafeDist()}; 

        // calculate bearing and distance for the hazard source
        double brg{}, dist{};
        mixr::base::nav::gll2bd(hs->getLatitude(), hs->getLongitude(), state->getLatitude(), state->getLongitude(), &brg, &dist);
        dist *= mixr::base::distance::NM2M; // from nm to meters

        double weightRepulsion{};
        if (dist > safeDist * 2.0)
            weightRepulsion = 0.0;
        else if (dist < safeDist)
            weightRepulsion = 1.0;
        else
            weightRepulsion = pow((safeDist * 2.0 - dist), expParam) / pow((safeDist), expParam);

        xres += cos(brg * mixr::base::angle::D2RCC) * weightRepulsion;
        yres += sin(brg * mixr::base::angle::D2RCC) * weightRepulsion;
    }

    // Resulting direction
    hdgRes = atan2(yres, xres) * mixr::base::angle::R2DCC;

    //----------------------- POC MIIA BEGIN ----------------------------------
    resultado = client->predict(modelo_id, inputs);
    //----------------------- POC MIIA END ------------------------------------
    
    // Defining dynamic action
    action->setHasDynamicAction(true);
    action->setReqHeading(hdgRes);

}
//------------------------------------------------------------------------------

void ShipAgent::setAction(const double dt)
{
    BaseClass::setAction(dt);

    auto action{dynamic_cast<ShipAction*>(getAction())};
    if (action != nullptr)
    {
        // Dynamic action
        if (action->getHasDynamicAction())
        {
            auto hydro{dynamic_cast<Hydrodynamics*>(getOwnship()->getDynamicsModel())};
            if (hydro != nullptr)
            {
                hydro->setCommandedHeadingD(action->getReqHeading(), hydro->getTurnRate());
            }
        }
    }    
}

} // namespace asa_tutorial
