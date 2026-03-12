// mixr
#include "mixr/base/PairStream.hpp"
#include "mixr/models/Tdb.hpp"
#include "mixr/models/WorldModel.hpp"

// asa::tutorial
#include "asa/tutorial/HazardSource.hpp"
#include "asa/tutorial/Scanner.hpp"

namespace asa_tutorial
{
// clang-format off

IMPLEMENT_SUBCLASS(Scanner, "Scanner")
EMPTY_SLOTTABLE(Scanner)

Scanner::Scanner()
{
    STANDARD_CONSTRUCTOR()
}
// clang-format on

void Scanner::copyData(const Scanner& org, const bool)
{
    BaseClass::copyData(org);

    hazardSources.clear();
    hazardSources = org.hazardSources;
}
//------------------------------------------------------------------------------

void Scanner::deleteData()
{
    hazardSources.clear();
}
//------------------------------------------------------------------------------

void Scanner::reset()
{
    BaseClass::reset();

    hazardSources.clear();
}
//------------------------------------------------------------------------------

void Scanner::updateData(const double dt)
{
    BaseClass::updateData(dt);

    hazardSources.clear();

    auto own{getOwnship()};
    if (own != nullptr)
    {
        // Pass our players of interest (poi) to the gimbal for processing
        mixr::base::PairStream* poi{nullptr};
        mixr::models::WorldModel* sim{getWorldModel()};
        if (sim != nullptr)
        {
            poi = sim->getPlayers();
            processPlayersOfInterest(poi);
        }

        if (poi != nullptr)
            poi->unref();

        mixr::models::Player** sources{getCurrentTDB()->getTargets()};
        auto ntgt{getCurrentTDB()->getNumberOfTargets()};
        for (unsigned int i = 0; i < ntgt; i++)
        {
            if (sources[i]->isClassType(typeid(HazardSource)))
                hazardSources.push_back(sources[i]);
        }
    }
}

} // namespace asa_tutorial
