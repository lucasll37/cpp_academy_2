#include "asa/tutorial/ShipAction.hpp"

namespace asa_tutorial
{
// clang-format off
IMPLEMENT_SUBCLASS(ShipAction, "ShipAction")
EMPTY_SLOTTABLE(ShipAction)
EMPTY_DELETEDATA(ShipAction)

ShipAction::ShipAction()
{
    STANDARD_CONSTRUCTOR()
}
// clang-format off

void ShipAction::copyData(const ShipAction& org, const bool)
{
    BaseClass::copyData(org);

    hasDynamicAction = org.hasDynamicAction;
    reqHeading = org.reqHeading; 
}
//------------------------------------------------------------------------------

void ShipAction::reset()
{
    BaseClass::reset();
    
    hasDynamicAction = false;
}

} // namespace asa_tutorial
