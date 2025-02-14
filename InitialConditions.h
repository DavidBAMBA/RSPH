#ifndef INITIALCONDITIONS_H
#define INITIALCONDITIONS_H

#include <vector>
#include <memory>
#include "Particle.h"
#include "Kernel.h"
#include "EquationOfState.h"
#include "VariableConverter.h"

#include <omp.h>

enum class InitialConditionType {
    TEST_RSOD,
    TEST_RSOD2,
    TEST_SB,
    TEST_PERTUB_SIN,
    TEST_TRANS_VEL,
    NR_SOD
};

class InitialConditions {
public:
    InitialConditions();
    void setInitialConditionType(InitialConditionType type);
    void initializeParticles(std::vector<Particle>& particles,
                             std::shared_ptr<Kernel> kernel,
                             std::shared_ptr<EquationOfState> eos,
                             int N, double x_min, double x_max);


private:
    InitialConditionType icType;
};

#endif // INITIALCONDITIONS_H
