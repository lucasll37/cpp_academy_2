#ifndef __asa_tutorial_Ship_H__
#define __asa_tutorial_Ship_H__

// asa::models
#include "asa/models/player/mixr_inheritance/sea/AsaShip.hpp"

namespace asa_tutorial
{
//------------------------------------------------------------------------------
// Class: Ship
// Factory name: Ship
// Description:
//------------------------------------------------------------------------------
class Ship : public asa::models::AsaShip
{
    DECLARE_SUBCLASS(Ship, asa::models::AsaShip)

public:
    Ship();

    void updateData(const double dt) override;

private:
    double startTime{};      // [s] exec time
    double navigationTime{}; // [s] exec time

private:
    bool setSlotStartTime(mixr::base::Number*);
    bool setSlotNavigationTime(mixr::base::Number*);
};

} // namespace asa_tutorial

#endif
