#ifndef SHOCKTUBEIC_H
#define SHOCKTUBEIC_H

#include "IInitialCondition.h"
enum class InitialConditionType;   // forward‑declaration

/**
 *  Generador de condiciones iniciales 1‑D tipo shock‑tube.
 *  El comportamiento exacto (valores de ρ, P, etc.) depende del
 *  valor de `InitialConditionType` pasado en el constructor.
 */
class ShockTubeIC : public IInitialCondition
{
public:
    explicit ShockTubeIC(InitialConditionType type) : type_(type) {}

    /** Implementa la interfaz para crear las partículas. */
    void generate(std::vector<Particle>&      particles,
                  std::shared_ptr<Kernel>     kernel,
                  std::shared_ptr<EquationOfState> eos,
                  int    N,
                  double x_min,
                  double x_max) override;

private:
    InitialConditionType type_;
};

#endif // SHOCKTUBEIC_H
