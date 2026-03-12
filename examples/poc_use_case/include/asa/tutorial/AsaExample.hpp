#ifndef __asa_tutorial_AsaExample_H__
#define __asa_tutorial_AsaExample_H__

// mixr
#include "asa-poc-miia/client/inference_client.hpp"
#include "mixr/base/Object.hpp"

namespace asa_tutorial
{
//------------------------------------------------------------------------------
// Class: AsaExample
// Factory name: AsaExample
// Description:
//------------------------------------------------------------------------------
class AsaExample : public mixr::base::Object
{
    DECLARE_SUBCLASS(AsaExample, mixr::base::Object)

public:
    AsaExample();

protected:
private:
};

} // namespace asa_tutorial

#endif
