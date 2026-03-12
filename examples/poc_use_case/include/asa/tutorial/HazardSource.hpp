#ifndef __asa_tutorial_HazardSource_H__
#define __asa_tutorial_HazardSource_H__

// mixr
#include "mixr/base/numeric/Number.hpp"

// asa::models
#include "asa/models/player/mixr_inheritance/sea/AsaShip.hpp"

namespace asa_tutorial
{
//------------------------------------------------------------------------------
// Class: HazardSource
// Factory name: HazardSource
// Description:
//------------------------------------------------------------------------------
class HazardSource : public asa::models::AsaShip
{
    DECLARE_SUBCLASS(HazardSource, asa::models::AsaShip)

public:
    HazardSource();

    double getMinSafeDist() { return minSafeDist; }

private:
    double minSafeDist{}; // [m] minimum distance from the source considered safe

private:
    bool setSlotMinSafeDist(mixr::base::Number* const);
};

} // namespace asa_tutorial

#endif
