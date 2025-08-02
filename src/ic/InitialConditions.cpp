#include "InitialConditions.h"
#include "ShockTubeIC.h"          // 1‑D shock tubes
// #include "KelvinHelmholtzIC.h"  // (cuando lo implementes)

#include <stdexcept>              // std::runtime_error

/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
/*                        CREATE (“factory”)                         */
/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
namespace {

/* Devuelve el generador concreto adecuado …………………………………… */
std::unique_ptr<IInitialCondition>
makeGenerator(InitialConditionType type)
{
    switch (type)
    {
        /* Todos los shock tubes usan la misma clase ———————*/
        case InitialConditionType::TEST_RSOD :
        case InitialConditionType::TEST_RSOD2:
        case InitialConditionType::TEST_RSOD3:
        case InitialConditionType::TEST_SB   :
        case InitialConditionType::TEST_PERTUB_SIN:
        case InitialConditionType::TEST_TRANS_VEL :
        case InitialConditionType::NR_SOD          :
            return std::make_unique<ShockTubeIC>(type);

        /* Place‑holder para 2‑D/3‑D …———————————————*/
        case InitialConditionType::KELVIN_HELMHOLTZ:
            throw std::runtime_error(
                "Kelvin–Helmholtz IC aún no implementada");

        /* Tipo desconocido ———————————————*/
        default:
            throw std::runtime_error("Tipo de condición inicial no soportado");
    }
}

} // fin namespace anónimo
/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
/*                         MÉTODOS PÚBLICOS                          */
/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/

InitialConditions::InitialConditions()
    : icType_(InitialConditionType::TEST_SB)                // por defecto
{}

void InitialConditions::setInitialConditionType(InitialConditionType type)
{
    icType_ = type;
}

void InitialConditions::initializeParticles(std::vector<Particle>&      particles,
                                            std::shared_ptr<Kernel>     kernel,
                                            std::shared_ptr<EquationOfState> eos,
                                            int    N,
                                            double x_min,
                                            double x_max)
{
    /* 1. crear el generador adecuado */
    std::unique_ptr<IInitialCondition> gen = makeGenerator(icType_);

    /* 2. delegar la construcción de las partículas */
    gen->generate(particles, kernel, eos, N, x_min, x_max);
}
