// asa::extensions
#include "asa/extensions/AsaExtension.hpp"

//
// ASA Reserved Models
//

#include "asa/tutorial/config.hpp"

// tutorial
#include "asa/tutorial/HazardSource.hpp"
#include "asa/tutorial/Hydrodynamics.hpp"
#include "asa/tutorial/Scanner.hpp"
#include "asa/tutorial/Ship.hpp"
#include "asa/tutorial/ShipAgent.hpp"

// Exportando todos os modelos implementados por essa Extension

namespace asa_tutorial
{

// clang-format off

ASA_EXTENSION_BEGIN(AsaTutorial, ASA_TUTORIAL_VERSION_MAJOR, ASA_TUTORIAL_VERSION_MINOR, ASA_TUTORIAL_VERSION_PATCH)

	// example
	ASA_EXTENSION_REGISTER(Scanner)
	ASA_EXTENSION_REGISTER(Ship)
	ASA_EXTENSION_REGISTER(ShipAgent)
	ASA_EXTENSION_REGISTER(HazardSource)
    ASA_EXTENSION_REGISTER(Hydrodynamics)

ASA_EXTENSION_END()

// clang-format on

} // namespace asa_tutorial
