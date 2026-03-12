// mixr
#include "mixr/base/units/Times.hpp"
#include "mixr/models/WorldModel.hpp"

// asa::core
#include "asa/core/base/Log.hpp"

// asa::extensions
#include "asa/extensions/recorder/Utils.hpp"
#include "asa/extensions/recorder/protobuf/Helper.hpp"

// asa::models
#include "asa/models/recorder/AsaModelsRecorder.hpp"

// asa::tutorial
#include "asa/tutorial/Ship.hpp"

namespace asa_tutorial
{
// clang-format off

IMPLEMENT_SUBCLASS(Ship, "Ship")
EMPTY_DELETEDATA(Ship)

BEGIN_SLOTTABLE(Ship)
    "startTime",      // 1)
    "navigationTime", // 2)
END_SLOTTABLE(Ship)

BEGIN_SLOT_MAP(Ship)
    ON_SLOT(1, setSlotStartTime,      mixr::base::Number)
    ON_SLOT(2, setSlotNavigationTime, mixr::base::Number)
END_SLOT_MAP()

Ship::Ship()
{
    STANDARD_CONSTRUCTOR()

    setInitMode(mixr::models::Player::INACTIVE);
    setMode(mixr::models::Player::INACTIVE);
}
// clang-format on

void Ship::copyData(const Ship& org, const bool)
{
    BaseClass::copyData(org);

    startTime = org.startTime;
    navigationTime = org.navigationTime;
}
//------------------------------------------------------------------------------

void Ship::updateData(const double dt)
{
    BaseClass::updateData(dt);

    double execTime{getWorldModel()->getExecTimeSec()};
    auto curMode{getMode()};

    if (execTime >= startTime && execTime < (startTime + navigationTime) &&
        curMode == mixr::simulation::AbstractPlayer::Mode::INACTIVE)
    {
        setMode(mixr::simulation::AbstractPlayer::Mode::ACTIVE);

        // registry of "departure"
        // it's just a way to trick Tacview into displaying the ship
        asamodels::recorder::pb::AsaAircraftTakeoffMsg msg{};
        auto id{msg.mutable_player_id()};
        id->set_fed_name("none");
        id->set_id(getID());
        id->set_name(*getName());
        id->set_side(asa::recorder::pb::Helper::getPlayerSide(getSide()));
        id->set_type(*getType());

        msg.set_airbase_id("Virtual Naval Base");
        msg.set_formation_id(getName()->getCopyString());
        msg.set_init_bmb(0);
        msg.set_init_fuel(0);
        msg.set_init_msl(0);
        msg.set_rank_id(1);
        msg.set_task_type("Virtual Navigation");

        RECORD_ASA_CUSTOM_DATA("asa::models::AsaAircraftTakeoffMsg", msg)
    }

    if (execTime >= startTime + navigationTime &&
        curMode == mixr::simulation::AbstractPlayer::Mode::ACTIVE)
    {
        // registy of "landing"
        // it's just a way to trick Tacview into not displaying the ship anymore
        asamodels::recorder::pb::AsaAircraftLandMsg msg{};
        auto id{msg.mutable_player_id()};
        id->set_fed_name("none");
        id->set_id(getID());
        id->set_name(*getName());
        id->set_side(asa::recorder::pb::Helper::getPlayerSide(getSide()));
        id->set_type(*getType());

        msg.set_airbase_id("Virtual Naval Base");
        msg.set_consumed_fuel(0);
        msg.set_formation_id(getName()->getCopyString());
        msg.set_rank_id(1);
        msg.set_remaining_bmb(0);
        msg.set_remaining_msl(0);
        msg.set_remaining_fuel(0);
        msg.set_remaining_msl(0);
        msg.set_time_of_flight(navigationTime);

        RECORD_ASA_CUSTOM_DATA("asa::models::AsaAircraftLandMsg", msg)

        setMode(mixr::simulation::AbstractPlayer::Mode::INACTIVE);
    }
}

//------------------------------------------------------------------------------
// Slot functions
//------------------------------------------------------------------------------
bool Ship::setSlotStartTime(mixr::base::Number* const msg)
{
    bool ok{};

    if (msg != nullptr)
    {
        double val{};
        auto time{dynamic_cast<mixr::base::Time*>(msg)};
        if (time != nullptr)
            val = mixr::base::Seconds::convertStatic(*time);
        else
            val = msg->getDouble(); // assuming seconds

        if (val >= 0)
        {
            startTime = val;
            ok = true;
        }

        if (!ok)
            ASA_LOG(error) << "Ship::setSlotStartTime(): invalid value.";
    }
    else
        ASA_LOG(error) << "Ship::setSlotStartTime(): invalid input.";

    return ok;
}
//------------------------------------------------------------------------------

bool Ship::setSlotNavigationTime(mixr::base::Number* const msg)
{
    bool ok{};

    if (msg != nullptr)
    {
        double val{};
        auto time{dynamic_cast<mixr::base::Time*>(msg)};
        if (time != nullptr)
            val = mixr::base::Seconds::convertStatic(*time);
        else
            val = msg->getDouble(); // assuming seconds

        if (val > 0)
        {
            navigationTime = val;
            ok = true;
        }

        if (!ok)
            ASA_LOG(error) << "Ship::setSlotNavigationTime(): invalid value.";
    }
    else
        ASA_LOG(error) << "Ship::setSlotNavigationTime(): invalid input.";

    return ok;
}

} // namespace asa_tutorial
