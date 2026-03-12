// asa::core
#include "asa/core/base/Log.hpp"

// mixr
#include "mixr/base/util/math_utils.hpp"

// asa::tutorial
#include "asa/tutorial/Hydrodynamics.hpp"

namespace asa_tutorial
{
// clang-format off

IMPLEMENT_SUBCLASS(Hydrodynamics, "Hydrodynamics")
EMPTY_DELETEDATA(Hydrodynamics)

BEGIN_SLOTTABLE(Hydrodynamics)
    "turnRate",  // 1)
END_SLOTTABLE(Hydrodynamics)

BEGIN_SLOT_MAP(Hydrodynamics)
    ON_SLOT(1, setSlotTurnRate, mixr::base::Number)
END_SLOT_MAP()

Hydrodynamics::Hydrodynamics()
{
    STANDARD_CONSTRUCTOR()
}
// clang-format on

void Hydrodynamics::copyData(const Hydrodynamics& org, const bool)
{
    BaseClass::copyData(org);

    turnRate = org.turnRate;
}
//------------------------------------------------------------------------------

void Hydrodynamics::reset()
{
    BaseClass::reset();

    ownship = static_cast<mixr::models::Player*>(findContainerByType(typeid(mixr::models::Player)));
    if (ownship != nullptr)
    {
        psi = ownship->getHeading();
        psiDot = turnRate * mixr::base::angle::D2RCC;
        psiDot1 = psiDot;

        velN = ownship->getInitVelocity() * cos(psi);
        velE = ownship->getInitVelocity() * sin(psi);
    }
}
//------------------------------------------------------------------------------

void Hydrodynamics::dynamics(const double dt)
{
    //----------------------------------------------------
    // integrate Euler angles using Adams-Bashforth
    //----------------------------------------------------
    psi += 0.5 * (3.0 * psiDot - psiDot1) * dt;
    if (psi > mixr::base::PI)
        psi = -mixr::base::PI;
    if (psi < -mixr::base::PI)
        psi = mixr::base::PI;

    //----------------------------------------------------
    // update Euler angles
    //----------------------------------------------------
    ownship->setEulerAngles(0.0, 0.0, psi);

    //----------------------------------------------------
    // hold current rotational control values for next iteration
    //----------------------------------------------------
    psiDot1 = psiDot;

    //----------------------------------------------------
    // update angular velocities
    //----------------------------------------------------
    ownship->setAngularVelocities(0, 0, psiDot);

    //----------------------------------------------------
    // update velocity in NED system
    //----------------------------------------------------
    double spd{ownship->getTotalVelocity()};
    velN = spd * cos(psi);
    velE = spd * sin(psi);

    ownship->setVelocity(velN, velE, 0.0);
}
//------------------------------------------------------------------------------

bool Hydrodynamics::setCommandedHeadingD(const double h, const double hDps, const double maxBank)
{
    bool ok{(ownship != nullptr)};
    if (ok)
    {
        //----------------------------------------------------
        // define local constants
        //----------------------------------------------------
        const double TAU{1.0}; // time constant [sec]

        //-------------------------------------------------------
        // get current data
        //-------------------------------------------------------
        double hdgDeg{ownship->getHeadingD()};
        double hdgErrDeg{mixr::base::angle::aepcdDeg(h - hdgDeg)};
        double hdgErrAbsDeg{std::fabs(hdgErrDeg)};

        //-------------------------------------------------------
        // get absolute heading rate of change (hdgDotAbsDps)
        //-------------------------------------------------------
        double hdgDotAbsDps{fabs(hDps)};
        double hdgErrBrkAbsDeg{TAU * hdgDotAbsDps};
        if (hdgErrAbsDeg < hdgErrBrkAbsDeg)
            hdgDotAbsDps = hdgErrAbsDeg / TAU;

        //-------------------------------------------------------
        // define direction of heading rate of change (hdgDotDps)
        //-------------------------------------------------------
        double hdgDotDps{mixr::base::sign(hdgErrDeg) * hdgDotAbsDps};
        psiDot = hdgDotDps * mixr::base::angle::D2RCC;
    }

    return ok;
}

//------------------------------------------------------------------------------
// Slot functions
//------------------------------------------------------------------------------

bool Hydrodynamics::setSlotTurnRate(mixr::base::Number* const msg)
{
    bool ok{};
    if (msg != nullptr)
    {
        double val{msg->getDouble()};
        if (val > 0.0)
        {
            turnRate = val; // assuming deg/s
            ok = true;
        }
        else
            ASA_LOG(error) << "Hydrodynamics::setSlotTurnRate: invalid value.";
    }
    else
        ASA_LOG(error) << "Hydrodynamics::setSlotTurnRate: invalid input.";

    return ok;
}

} // namespace asa_tutorial
