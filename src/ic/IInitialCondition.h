#ifndef IINITIALCONDITION_H
#define IINITIALCONDITION_H

#include <vector>
#include <memory>

// ――― Forward declarations (evita dependencias circulares) ―――
class Particle;
class Kernel;
class EquationOfState;

/**
 *  Interfaz común que deben implementar todas las clases
 *  generadoras de condiciones iniciales.
 */
class IInitialCondition
{
public:
    virtual ~IInitialCondition() = default;


    virtual void generate(std::vector<Particle>&             particles,
                          std::shared_ptr<Kernel>            kernel,
                          std::shared_ptr<EquationOfState>   eos,
                          int    N,
                          double x_min,
                          double x_max) = 0;
};

#endif // IINITIALCONDITION_H
