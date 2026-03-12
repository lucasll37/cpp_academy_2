#ifndef __asa_tutorial_Scanner_H__
#define __asa_tutorial_Scanner_H__

// std
#include <list>

// mixr
#include "mixr/models/player/Player.hpp"
#include "mixr/models/system/Gimbal.hpp"

namespace asa_tutorial
{
//------------------------------------------------------------------------------
// Class: Scanner
// Factory name: Scanner
// Description:
//------------------------------------------------------------------------------
class Scanner : public mixr::models::Gimbal
{
    DECLARE_SUBCLASS(Scanner, mixr::models::Gimbal)

public:
    Scanner();

    std::list<mixr::models::Player*>* getHazardSources() { return &hazardSources; }

    void reset() override;
    void updateData(const double dt) override;

private:
    std::list<mixr::models::Player*> hazardSources{};
};

} // namespace asa_tutorial

#endif
