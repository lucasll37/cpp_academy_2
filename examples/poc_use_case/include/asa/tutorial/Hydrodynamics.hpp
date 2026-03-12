#ifndef __asa_tutorial_Hydrodynamics_H__
#define __asa_tutorial_Hydrodynamics_H__

// mixr
#include "mixr/base/numeric/Number.hpp"
#include "mixr/models/dynamics/DynamicsModel.hpp"
#include "mixr/models/player/Player.hpp"

namespace asa_tutorial
{
//------------------------------------------------------------------------------
// Class: Hydrodynamics
// Factory name: Hydrodynamics
// Description:
//------------------------------------------------------------------------------
class Hydrodynamics : public mixr::models::DynamicsModel
{
    DECLARE_SUBCLASS(Hydrodynamics, mixr::models::DynamicsModel)

public:
    Hydrodynamics();

    double getTurnRate() { return turnRate; }

    void reset() override;
    void dynamics(const double dt) override;
    bool setCommandedHeadingD(const double h, const double hDps = 0, const double maxBank = 0.0) override;

private:
    double turnRate{}; // [deg/s] rate of heading change

    // Body angular vel
    double psiDot{};  // [rad/s]
    double psiDot1{}; // [rad/s]

    // Euler rotation angle
    double psi{}; // [rad]

    // NED velocities
    double velN{}; // [m/s]
    double velE{}; // [m/s]

    // Ownship
    mixr::models::Player* ownship{};

private:
    bool setSlotTurnRate(mixr::base::Number* const);
};

} // namespace asa_tutorial

#endif
