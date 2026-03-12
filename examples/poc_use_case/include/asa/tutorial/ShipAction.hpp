#ifndef __asa_tutorial_ShipAction_H__
#define __asa_tutorial_ShipAction_H__

#include "asa/models/agent/AsaAgentAction.hpp"

namespace asa_tutorial
{
//------------------------------------------------------------------------------
// Class: ShipAction
// Factory name: ShipAction
// Description:
//------------------------------------------------------------------------------
class ShipAction : public asa::models::AsaAgentAction
{
    DECLARE_SUBCLASS(ShipAction, asa::models::AsaAgentAction)

public:
    ShipAction();

    bool getHasDynamicAction() { return hasDynamicAction; }
    void setHasDynamicAction(const bool b) { hasDynamicAction = b; }

    double getReqHeading() { return reqHeading; }
    void setReqHeading(const double hdg) { reqHeading = hdg; }

    void reset() override;

private:
    bool hasDynamicAction{};
    double reqHeading{};
};


} // namespace asa_tutorial

#endif
