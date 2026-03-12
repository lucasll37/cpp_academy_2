// asa::core
#include "asa/core/base/Log.hpp"

// mixr
#include "mixr/base/units/Distances.hpp"

// asa::tutorial
#include "asa/tutorial/HazardSource.hpp"

namespace asa_tutorial
{
// clang-format off

IMPLEMENT_SUBCLASS(HazardSource, "HazardSource")
EMPTY_DELETEDATA(HazardSource)

BEGIN_SLOTTABLE(HazardSource)
    "minSafeDist",  // 1)
END_SLOTTABLE(HazardSource)

BEGIN_SLOT_MAP(HazardSource)
    ON_SLOT(1, setSlotMinSafeDist, mixr::base::Number)
END_SLOT_MAP()

HazardSource::HazardSource()
{
    STANDARD_CONSTRUCTOR()
}
// clang-format on

void HazardSource::copyData(const HazardSource& org, const bool)
{
    BaseClass::copyData(org);

    minSafeDist = org.minSafeDist;
}

//------------------------------------------------------------------------------
// Slot functions
//------------------------------------------------------------------------------

bool HazardSource::setSlotMinSafeDist(mixr::base::Number* const msg)
{
    bool ok{};
    if (msg != nullptr)
    {
        double val{};
        auto dist{dynamic_cast<mixr::base::Distance*>(msg)};
        if (dist != nullptr)
            val = mixr::base::Meters::convertStatic(*dist);
        else
            val = msg->getDouble(); // assuming meters

        if (val > 0.0)
        {
            minSafeDist = val;
            ok = true;
        }
        else
            ASA_LOG(error) << "HazardSource::setSlotMinSafeDist(): invalid value.";
    }

    return ok;
}

} // namespace asa_tutorial
