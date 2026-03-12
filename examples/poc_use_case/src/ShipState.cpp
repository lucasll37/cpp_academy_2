#include "asa/tutorial/ShipState.hpp"

namespace asa_tutorial
{
// clang-format off
IMPLEMENT_SUBCLASS(ShipState, "ShipState")
EMPTY_SLOTTABLE(ShipState)
EMPTY_DELETEDATA(ShipState)

ShipState::ShipState()
{
    STANDARD_CONSTRUCTOR()
}
// clang-format off

void ShipState::copyData(const ShipState& org, const bool)
{
    BaseClass::copyData(org);

    hasValidNavData = org.hasValidNavData;
    toHeading = org.toHeading;
    
    hasValidScannerData = org.hasValidScannerData;
    hazardSources = org.hazardSources;
}
//------------------------------------------------------------------------------

void ShipState::reset()
{
    BaseClass::reset();
    
    hasValidNavData = false;
    
    hasValidScannerData = false;
    hazardSources = nullptr;   
}

} // namespace asa_tutorial
