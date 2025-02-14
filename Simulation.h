#ifndef SIMULATION_H
#define SIMULATION_H

#include <omp.h>
#include <iomanip>
#include <vector>
#include <memory>
#include <string>
#include "Particle.h"
#include "DensityUpdater.h"
#include "EquationOfState.h"
#include "DissipationTerms.h"
#include "InitialConditions.h"
#include "VariableConverter.h"
#include "Kernel.h"
#include "Boundaries.h"

#include <algorithm>    // Para std::find, std::min, std::remove_if
#include <iterator>

class Simulation {
public:
    Simulation(std::shared_ptr<Kernel> kern,
               std::shared_ptr<EquationOfState> eqs,
               std::shared_ptr<DissipationTerms> diss,
               std::shared_ptr<InitialConditions> initConds,
               double eta,
               double tol,
               bool use_fixed_h,
               double fixed_h,
               int N,                 // Número de partículas
               double x_min,
               double x_max,
               BoundaryType boundaryType);

    void run(double endTime);

private:
    double time;
    std::vector<Particle> particles;
    std::shared_ptr<Kernel> kernel;
    std::shared_ptr<EquationOfState> eos;
    std::shared_ptr<DissipationTerms> dissipation;
    std::shared_ptr<InitialConditions> initialConditions;
    DensityUpdater densityUpdater;
    Boundaries boundaries;

    int N;
    double x_min;
    double x_max;
    bool isNaN(double value);

    // Integradores:
    void rungeKuttaTVD3Step(double timeStep, VariableConverter& converter);
    void rungeKutta2Step(double timeStep, VariableConverter& converter);

    // Métodos para calcular el paso de tiempo:
    double calculateTimeStep() const;
    double calculateTimeStep2() const;

    // Funciones auxiliares:
    void writeOutputCSV(const std::string& filename) const;
    bool validateParticles() const;

    // Función helper para integrar una copia del estado usando RK2 o TVD3.
    // La bandera useRK3 = true indica que se utiliza el integrador de orden tres TVD (KVD3),
    // mientras que false se usa el método RK2.
    void integrateState(std::vector<Particle>& state, double dt, bool useRK3, VariableConverter& converter) const;
    //td::vector<Particle> getNeighbors1(const Particle& particle) const;{
    
    double computeTotalEnergy() const;
    double computeTotalMass() const;
    void appendInvariantsToFile(const std::string& filename, double currentTime, int step) const;

};

#endif // SIMULATION_H
