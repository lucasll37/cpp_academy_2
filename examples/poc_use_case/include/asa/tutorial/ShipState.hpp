#ifndef __asa_tutorial_ShipState_H__
#define __asa_tutorial_ShipState_H__

// std
#include <list>

// mixr
#include "mixr/models/player/Player.hpp"

// asa::models
#include "asa/models/agent/AsaAgentState.hpp"

namespace asa_tutorial
{
//------------------------------------------------------------------------------
// Class: ShipState
// Factory name: ShipState
// Description:
//------------------------------------------------------------------------------
class ShipState : public asa::models::AsaAgentState
{
    DECLARE_SUBCLASS(ShipState, asa::models::AsaAgentState)

public:
    ShipState();

    bool getHasValidNavData() { return hasValidNavData; }
    void setHasValidNavData(const bool b) { hasValidNavData = b; }

    double getToHeading() { return toHeading; }
    void setToHeading(const double hdg) { toHeading = hdg; }

    bool getHasValidScannerData() { return hasValidScannerData; }
    void setHasValidScannerData(const bool b) { hasValidScannerData = b; }

    std::list<mixr::models::Player*>* getHazardSources() { return hazardSources; }
    void setHazardSources(std::list<mixr::models::Player*>* lst) { hazardSources = lst; }

    void reset() override;

private:
    bool hasValidNavData{};
    double toHeading{};

    bool hasValidScannerData{};
    std::list<mixr::models::Player*>* hazardSources{};
};


} // namespace asa_tutorial

#endif
