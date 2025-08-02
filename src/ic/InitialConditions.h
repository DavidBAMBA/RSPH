#ifndef INITIALCONDITIONS_H
#define INITIALCONDITIONS_H

#include "InitialConditionTypes.h"
#include "IInitialCondition.h"
#include <memory>
#include <vector>

/**
 * @brief Fachada sencilla que esconde el “factory”.
 *
 * Tu aplicación solo crea un objeto `InitialConditions`, selecciona
 * el tipo con `setInitialConditionType( … )` y llama a
 * `initializeParticles( … )`. Internamente delega el trabajo al
 * generador adecuado (`ShockTubeIC`, etc.).
 */
class InitialConditions
{
public:
    InitialConditions();                                     ///< por defecto TEST_SB
    void setInitialConditionType(InitialConditionType type); ///< elige la IC
                                                             ///< crea las partículas
    void initializeParticles(std::vector<Particle>&          particles,
                             std::shared_ptr<Kernel>         kernel,
                             std::shared_ptr<EquationOfState> eos,
                             int    N,
                             double x_min,
                             double x_max);

private:
    InitialConditionType             icType_;                ///< selección actual
    std::unique_ptr<IInitialCondition> create(InitialConditionType);  ///< “factory”
};

#endif // INITIALCONDITIONS_H
