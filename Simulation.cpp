// Simulation.cpp
#include "Simulation.h"
#include "MathUtils.h"
#include "VariableConverter.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <omp.h>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sstream>
#include "Globals.h"

std::unique_ptr<Logger> g_logger;  // Definición única

Simulation::Simulation(std::shared_ptr<Kernel> kern,
                       std::shared_ptr<EquationOfState> eqs,
                       std::shared_ptr<DissipationTerms> diss,
                       std::shared_ptr<InitialConditions> initConds,
                       double eta,
                       double tol,
                       bool use_fixed_h,
                       double fixed_h,
                       int N,
                       double x_min,
                       double x_max,
                       BoundaryType boundaryType)
    : time(0.0),
      kernel(kern),
      eos(eqs),
      dissipation(diss),
      initialConditions(initConds),
      densityUpdater(eta, tol, use_fixed_h, fixed_h),
      boundaries(x_min, x_max, boundaryType),
      N(N),
      x_min(x_min),
      x_max(x_max)
{
}

bool Simulation::isNaN(double value) {
    return std::isnan(value);
}

//------------------------------------------------------------
// Método calculateTimeStep(): dt basado en condiciones de Courant y fuerza
//------------------------------------------------------------
double Simulation::calculateTimeStep() const {
    double C_cour = 0.3;   // Coeficiente Courant
    double C_force = 0.25; // Coeficiente fuerza

    double min_dt = std::numeric_limits<double>::max();

    // Bucle paralelo con reducción para encontrar el mínimo de dt
    #pragma omp parallel for reduction(min:min_dt)
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& particle = particles[i];
        double v_sig_max = 0.0;

        // Calculamos v_sig_max revisando a los vecinos (en este ejemplo, todos)
        for (const auto& neighbor : particles) {
            std::array<double, 3> v_ab = {
                particle.velocity[0] - neighbor.velocity[0],
                particle.velocity[1] - neighbor.velocity[1],
                particle.velocity[2] - neighbor.velocity[2]
            };

            std::array<double, 3> r_ab = {
                particle.position[0] - neighbor.position[0],
                particle.position[1] - neighbor.position[1],
                particle.position[2] - neighbor.position[2]
            };

            // Versor de la distancia r_ab
            std::array<double, 3> r_ab_hat = MathUtils::hatVector(r_ab);

            double cs_a = eos->calculateSoundSpeed(particle.density,
                                                   particle.specificInternalEnergy);
            double v_ab_star = -MathUtils::dotProduct(v_ab, r_ab_hat);
            double v_sig_a = (cs_a + std::fabs(v_ab_star)) 
                             / (1.0 + cs_a * std::fabs(v_ab_star));

            v_sig_max = std::max(v_sig_max, v_sig_a);
        }

        // Si v_sig_max es cero o negativo, no podemos calcular dt.
        if (v_sig_max <= 0.0) continue;
        double dt_cour = C_cour * (particle.h / v_sig_max);

        // Cálculo del dt_force
        double a_magnitude = 0.0;
        for (int d = 0; d < 3; ++d) {
            a_magnitude += std::pow(particle.acceleration[d], 2);
        }
        a_magnitude = std::sqrt(a_magnitude);

        double dt_force = (a_magnitude > 0.0) 
                          ? C_force * std::sqrt(particle.h / a_magnitude)
                          : std::numeric_limits<double>::max();

        double dt_particle = std::min(dt_cour, dt_force);

        // Actualizamos el mínimo de forma segura gracias a la reducción
        min_dt = std::min(min_dt, dt_particle);
    }

    // Control de un paso de tiempo mínimo para evitar dt demasiado pequeños
    double min_allowed_dt = 1e-8;
    if (min_dt < min_allowed_dt) {
        #pragma omp critical
        {
            std::cerr << "Advertencia: Paso de tiempo ajustado al mínimo permitido: "
                      << min_allowed_dt << std::endl;
        }
        min_dt = min_allowed_dt;
    }

    return min_dt;
}

//------------------------------------------------------------
// Método calculateTimeStep2(): dt adaptativo usando integración de prueba
//------------------------------------------------------------
double Simulation::calculateTimeStep2() const {
    double C_cour = 0.3;
    double C_force = 0.25;

    double min_dt = std::numeric_limits<double>::max();
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto &p = particles[i];
        double v_sig_max = 0.0;
        for (const auto &neighbor : particles) {
            std::array<double, 3> v_ab = {
                p.velocity[0] - neighbor.velocity[0],
                p.velocity[1] - neighbor.velocity[1],
                p.velocity[2] - neighbor.velocity[2]
            };
            std::array<double, 3> r_ab = {
                p.position[0] - neighbor.position[0],
                p.position[1] - neighbor.position[1],
                p.position[2] - neighbor.position[2]
            };
            std::array<double, 3> r_ab_hat = MathUtils::hatVector(r_ab);
            double cs_a = eos->calculateSoundSpeed(p.density, p.specificInternalEnergy);
            double v_ab_star = -MathUtils::dotProduct(v_ab, r_ab_hat);
            double v_sig_a = (cs_a + std::fabs(v_ab_star)) / (1.0 + cs_a * std::fabs(v_ab_star));
            v_sig_max = std::max(v_sig_max, v_sig_a);
        }
        if (v_sig_max <= 0.0) continue;
        double dt_cour = C_cour * (p.h / v_sig_max);
        double a_magnitude = std::sqrt(
            p.acceleration[0]*p.acceleration[0] +
            p.acceleration[1]*p.acceleration[1] +
            p.acceleration[2]*p.acceleration[2]
        );
        double dt_force = (a_magnitude > 0.0) ? C_force * std::sqrt(p.h / a_magnitude)
                                              : std::numeric_limits<double>::max();
        double dt_particle = std::min(dt_cour, dt_force);
        min_dt = std::min(min_dt, dt_particle);
    }
    double dt_candidate = (min_dt < 1e-8) ? 1e-8 : min_dt;

    std::vector<Particle> state_RK2 = particles;
    std::vector<Particle> state_RK3 = particles;

    VariableConverter converter(1e-10);
    integrateState(state_RK2, dt_candidate, false, converter);
    integrateState(state_RK3, dt_candidate, true, converter);

    double max_rel_error = 0.0;
    for (size_t i = 0; i < state_RK3.size(); ++i) {
        double N_RK3 = state_RK3[i].baryonDensity;
        double N_RK2 = state_RK2[i].baryonDensity;
        if (std::fabs(N_RK3) < 1e-14) continue;
        double rel_error = std::fabs(N_RK3 - N_RK2) / std::fabs(N_RK3);
        max_rel_error = std::max(max_rel_error, rel_error);
    }
    double epsilon = max_rel_error / dt_candidate;
    double s = 0.8;
    double epsilon_tol = 5e-4;
    double dt_new = s * std::sqrt(epsilon_tol / epsilon) * dt_candidate;
    
    g_logger->log("[TimeIntegration] dt_candidate = " + std::to_string(dt_candidate) +
                    ", max_rel_error = " + std::to_string(max_rel_error) +
                    ", epsilon = " + std::to_string(epsilon) +
                    ", dt_new = " + std::to_string(dt_new) + "\n");
                    
    return dt_new;
}

//------------------------------------------------------------
// Helper: integrar una copia del estado usando RK2 o RK3 (TVD3)
//------------------------------------------------------------
void Simulation::integrateState(std::vector<Particle>& state, double dt, bool useRK3, VariableConverter& converter) const {
    Simulation tempSim(*this);
    tempSim.particles = state;
    if (useRK3) {
        tempSim.rungeKuttaTVD3Step(dt, converter);
    } else {
        tempSim.rungeKutta2Step(dt, converter);
    }
    state = tempSim.particles;
}

//------------------------------------------------------------
// Integrador de orden dos (RK2) para el control de dt
//------------------------------------------------------------
void Simulation::rungeKutta2Step(double timeStep, VariableConverter& converter) {
    g_logger->log("[RK2] Iniciando paso RK2 con dt = " + std::to_string(timeStep) + "\n");

    size_t numParticles = particles.size();
    std::vector<std::array<double, 3>> S0(numParticles);
    std::vector<double> E0(numParticles);
    std::vector<std::array<double, 3>> pos0(numParticles);
    std::vector<std::array<double, 3>> vel0(numParticles);

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        S0[i]   = particles[i].specificMomentum;
        E0[i]   = particles[i].specificEnergy;
        pos0[i] = particles[i].position;
        vel0[i] = particles[i].velocity;
    }
    g_logger->log("[RK2] Estado inicial guardado para " + std::to_string(numParticles) + " partículas.\n");

    std::vector<std::array<double, 3>> k1S(numParticles);
    std::vector<double> k1E(numParticles);
    std::vector<std::array<double, 3>> k1Pos(numParticles);

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (particles[i].isGhost) continue;
        particles[i].calculateAccelerationAndEnergyChangeRate(particles, *kernel, *dissipation, *eos);
        for (int d = 0; d < 3; ++d) {
            k1S[i][d] = timeStep * particles[i].acceleration[d];
            k1Pos[i][d] = timeStep * particles[i].velocity[d];
        }
        k1E[i] = timeStep * particles[i].energyChangeRate;
    }
    g_logger->log("[RK2] k1 calculado.\n");

    std::vector<Particle> tempState = particles;

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (tempState[i].isGhost) continue;
        for (int d = 0; d < 3; ++d) {
            tempState[i].specificMomentum[d] = S0[i][d] + k1S[i][d];
            tempState[i].position[d] = pos0[i][d] + k1Pos[i][d];
        }
        tempState[i].specificEnergy = E0[i] + k1E[i];
    }
    boundaries.apply(tempState);

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (tempState[i].isGhost) continue;
        tempState[i].updateDensity(tempState, densityUpdater, *kernel);
        tempState[i].updatePressure(*eos);
        converter.conservedToPrimitives(tempState[i], *eos);
    }
    g_logger->log("[RK2] Estado temporal u^(temp) calculado.\n");
    //stabilizeHighVariables();

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        converter.primitivesToConserved(tempState[i], *eos);
    }

    std::vector<std::array<double, 3>> k2S(numParticles);
    std::vector<double> k2E(numParticles);
    std::vector<std::array<double, 3>> k2Pos(numParticles);

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (tempState[i].isGhost) continue;
        tempState[i].calculateAccelerationAndEnergyChangeRate(tempState, *kernel, *dissipation, *eos);
        for (int d = 0; d < 3; ++d) {
            k2S[i][d] = timeStep * tempState[i].acceleration[d];
            k2Pos[i][d] = timeStep * tempState[i].velocity[d];
        }
        k2E[i] = timeStep * tempState[i].energyChangeRate;
    }
    g_logger->log("[RK2] k2 calculado.\n");

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (particles[i].isGhost) continue;
        for (int d = 0; d < 3; ++d) {
            particles[i].specificMomentum[d] = S0[i][d] + 0.5 * (k1S[i][d] + k2S[i][d]);
            particles[i].position[d] = pos0[i][d] + 0.5 * (k1Pos[i][d] + k2Pos[i][d]);
        }
        particles[i].specificEnergy = E0[i] + 0.5 * (k1E[i] + k2E[i]);
    }
    g_logger->log("[RK2] Estado final calculado con RK2.\n");


    boundaries.apply(particles);
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        particles[i].updateDensity(particles, densityUpdater, *kernel);
        particles[i].updatePressure(*eos);
        converter.conservedToPrimitives(particles[i], *eos);
    }
    g_logger->log("[RK2] Conversión final y actualizaciones completadas.\n");
    //stabilizeHighVariables();
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        converter.primitivesToConserved(particles[i], *eos);
    }
    time += timeStep;
    g_logger->log("[RK2] Paso RK2 completado. Tiempo actualizado a " + std::to_string(time) + "\n");
}

//------------------------------------------------------------
// Integrador TVD3 (KVD3): método de integración de orden tres TVD
//------------------------------------------------------------
void Simulation::rungeKuttaTVD3Step(double timeStep, VariableConverter& converter) {
    //g_logger->log("[TVD] Iniciando paso TVD con dt = " + std::to_string(timeStep) + "\n");

    size_t numParticles = particles.size();

    std::vector<std::array<double, 3>> S0(numParticles);
    std::vector<double> E0(numParticles);
    std::vector<std::array<double, 3>> pos0(numParticles);
    std::vector<std::array<double, 3>> vel0(numParticles);

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        S0[i]   = particles[i].specificMomentum;
        E0[i]   = particles[i].specificEnergy;
        pos0[i] = particles[i].position;
        vel0[i] = particles[i].velocity;
    }

    std::vector<std::array<double, 3>> k1S(numParticles), k2S(numParticles), k3S(numParticles);
    std::vector<double> k1E(numParticles),  k2E(numParticles),  k3E(numParticles);

    // SUB-PASO 1
    //g_logger->log("[TVD] Iniciando sub‐paso 1\n");
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (particles[i].isGhost) continue;
        particles[i].calculateAccelerationAndEnergyChangeRate(particles, *kernel, *dissipation, *eos);
        for (int d = 0; d < 3; ++d) {
            k1S[i][d] = timeStep * particles[i].acceleration[d];
        }
        k1E[i] = timeStep * particles[i].energyChangeRate;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (particles[i].isGhost) continue;
        for (int d = 0; d < 3; ++d) {
            particles[i].specificMomentum[d] = S0[i][d] + k1S[i][d];
            particles[i].position[d] = pos0[i][d] + timeStep * vel0[i][d];
        }
        particles[i].specificEnergy = E0[i] + k1E[i];
    }
    boundaries.apply(particles);
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        particles[i].updateDensity(particles, densityUpdater, *kernel);
        particles[i].updatePressure(*eos);
        converter.conservedToPrimitives(particles[i], *eos);
    }
    //g_logger->log("[TVD] Sub‐paso 1 completado (u^(1) actualizado y convertido).\n");
    //stabilizeHighVariables();

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        converter.primitivesToConserved(particles[i], *eos);
    }
    // SUB-PASO 2
    //g_logger->log("[TVD] Iniciando sub‐paso 2\n");

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (particles[i].isGhost) continue;
        particles[i].calculateAccelerationAndEnergyChangeRate(particles, *kernel, *dissipation, *eos);
        for (int d = 0; d < 3; ++d) {
            k2S[i][d] = timeStep * particles[i].acceleration[d];
        }
        k2E[i] = timeStep * particles[i].energyChangeRate;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (particles[i].isGhost) continue;
        for (int d = 0; d < 3; ++d) {
            double u_1_d = S0[i][d] + k1S[i][d];
            particles[i].specificMomentum[d] = 0.75 * S0[i][d] + 0.25 * u_1_d + 0.25 * k2S[i][d];
        }
        double e_1 = E0[i] + k1E[i];
        particles[i].specificEnergy = 0.75 * E0[i] + 0.25 * e_1 + 0.25 * k2E[i];
        for (int d = 0; d < 3; ++d) {
            double r1_d = pos0[i][d] + timeStep * vel0[i][d];
            double v1_d = particles[i].velocity[d];
            particles[i].position[d] = 0.75 * pos0[i][d] + 0.25 * r1_d + 0.25 * timeStep * v1_d;
        }
    }
    //g_logger->log("[TVD] Sub‐paso 2: u^(2) construido.\n");
    boundaries.apply(particles);
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        particles[i].updateDensity(particles, densityUpdater, *kernel);
        particles[i].updatePressure(*eos);
        converter.conservedToPrimitives(particles[i], *eos);
    }
    //g_logger->log("[TVD] Sub‐paso 2: Conversión y actualización en u^(2) completada.\n");
    //stabilizeHighVariables();

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        converter.primitivesToConserved(particles[i], *eos);
    }
    // SUB-PASO 3
    //g_logger->log("[TVD] Iniciando sub‐paso 3\n");

    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (particles[i].isGhost) continue;
        particles[i].calculateAccelerationAndEnergyChangeRate(particles, *kernel, *dissipation, *eos);
        for (int d = 0; d < 3; ++d) {
            k3S[i][d] = timeStep * particles[i].acceleration[d];
        }
        k3E[i] = timeStep * particles[i].energyChangeRate;
    }
    //g_logger->log("[TVD] Sub‐paso 3: f(u^(2)) calculado.\n");
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        if (particles[i].isGhost) continue;
        std::array<double, 3> S2 = particles[i].specificMomentum;
        double E2 = particles[i].specificEnergy;
        for (int d = 0; d < 3; ++d) {
            particles[i].specificMomentum[d] = (1.0/3.0)*S0[i][d] + (2.0/3.0)*S2[d] + (2.0/3.0)*k3S[i][d];
        }
        particles[i].specificEnergy = (1.0/3.0)*E0[i] + (2.0/3.0)*E2 + (2.0/3.0)*k3E[i];
        std::array<double, 3> r2 = particles[i].position;
        for (int d = 0; d < 3; ++d) {
            particles[i].position[d] = (1.0/3.0)*pos0[i][d] + (2.0/3.0)*r2[d] + (2.0/3.0)*timeStep*particles[i].velocity[d];
        }
    }
    //g_logger->log("[TVD] Sub‐paso 3: u^(n+1) y posiciones finales construidas.\n");
    boundaries.apply(particles);
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        particles[i].updateDensity(particles, densityUpdater, *kernel);
        particles[i].updatePressure(*eos);
        converter.conservedToPrimitives(particles[i], *eos);
    }
    //g_logger->log("[TVD] Sub‐paso 3: Conversión final y actualización completadas.\n");
    //stabilizeHighVariables();
    #pragma omp parallel for
    for (size_t i = 0; i < numParticles; ++i) {
        converter.primitivesToConserved(particles[i], *eos);
    }
    time += timeStep;
    //g_logger->log("[TVD] Paso completado. Tiempo actualizado a " + std::to_string(time) + "\n");
}

//------------------------------------------------------------
// Método run(): ciclo principal de la simulación
//------------------------------------------------------------
void Simulation::run(double endTime) {
    // Inicializar el logger global (escribe en "simulation_log.txt")
    g_logger = std::make_unique<Logger>("simulation_log.txt");

    // Inicializar las partículas
    auto converter = VariableConverter(1e-12); // Tolerancia
    initialConditions->initializeParticles(particles, kernel, eos, N, x_min, x_max);


    int step = 0;
    double current_time = 0.0;
    std::filesystem::create_directory("outputs");

    // Inicializar el archivo de invariantes (sobrescribir si existe)
    {
        std::ofstream ofs("invariants.txt");
        if (ofs) {
            ofs << "Time\tStep\tTotalMass\tTotalEnergy\n";
            ofs.close();
        }
    }

    // Parámetros para dt adaptativo
    const int maxRetries = 10;
    g_logger->log("Comienza simulación.\n");

    while (current_time < endTime) {
        std::string filename = "output_step_" + std::to_string(step) + ".csv";
        if (step % 20 == 0){
            writeOutputCSV(filename);
        }

        double dt = calculateTimeStep();
        //double dt2 = calculateTimeStep2();
        //double dt = std::min(dt1,1.0e-8);
        int retries = 0;
        bool stepAccepted = false;
        std::vector<Particle> backupParticles = particles;

        while (!stepAccepted && retries < maxRetries) {
            rungeKuttaTVD3Step(dt, converter);

            if (!validateParticles()) {
                std::string msg = "[Adaptive dt] Paso " + std::to_string(step) +
                                  " con dt = " + std::to_string(dt) +
                                  " produjo estado inestable. Reduciendo dt y reintentando.\n";
                std::cerr << msg;
                g_logger->log(msg);
                dt /= 2.0;
                particles = backupParticles;
                retries++;
            } else {
                stepAccepted = true;
            }
        }

        if (!stepAccepted) {
            std::string msg = "[Adaptive dt] Error: No se pudo estabilizar el paso " + std::to_string(step) +
                              " después de " + std::to_string(maxRetries) + " reintentos. Abortando.\n";
            std::cerr << msg;
            g_logger->log(msg);
            exit(EXIT_FAILURE);
        } else {
            current_time += dt;
            step++;
            double totalMass = computeTotalMass();
            double totalEnergy = computeTotalEnergy();

            std::ostringstream logMessage;
            logMessage << "---------------------------------------------------------------------------\n"
                       << " \t STEP: " << step 
                       << "\t dt: " << dt 
                       << "\t time: " << current_time << "\n"
                       << " \t -> Total Mass:   " << totalMass << "\n"
                       << " \t -> Total Energy: " << totalEnergy << "\n"
                       << "---------------------------------------------------------------------------\n";
            std::string msg = logMessage.str();
            std::cout << msg;
            g_logger->log(msg);

            // Guardar invariantes en un archivo aparte
            appendInvariantsToFile("invariants.txt", current_time, step);

            retries = 0;
        }
    }
    
    writeOutputCSV("output_final.csv");
    std::string completion_msg = "Simulación completada exitosamente.\n";
    std::cout << completion_msg;
    g_logger->log(completion_msg);
    g_logger.reset();
}


void Simulation::writeOutputCSV(const std::string& filename) const {
    std::filesystem::create_directory("outputs");
    std::string fullPath = "outputs/" + filename;
    std::ofstream file(fullPath);
    if (!file) {
        throw std::runtime_error("No se pudo abrir el archivo para escritura en: " + fullPath);
    }
    file << std::fixed << std::setprecision(8);
    file << "t\t        IsGhost\t  x\t        y\t        z\t        vx\t        vy\t        vz\t        P\t        u\t        d\t        Sx\t        Sy\t        Sz\t        N\t        e\t        mass\t    gamma\t    h\t        Ome\n";
    for (const auto& particle : particles) {
        file << time << "\t"
             << particle.isGhost                << "\t"
             << particle.position[0]            << "\t"
             << particle.position[1]            << "\t"
             << particle.position[2]            << "\t"
             << particle.velocity[0]            << "\t"
             << particle.velocity[1]            << "\t"
             << particle.velocity[2]            << "\t"
             << particle.pressure               << "\t"
             << particle.specificInternalEnergy << "\t"
             << particle.density                << "\t"
             << particle.specificMomentum[0]    << "\t"
             << particle.specificMomentum[1]    << "\t"
             << particle.specificMomentum[2]    << "\t"
             << particle.baryonDensity          << "\t"
             << particle.specificEnergy         << "\t"
             << particle.mass                   << "\t"
             << 1.0 / std::sqrt(1.0 - MathUtils::vectorNormSquared(particle.velocity)) << "\t"
             << particle.h << "\t"
             << particle.Omega << "\n";
    }
    file.close();
    std::cout << "Datos escritos en " << fullPath << std::endl;
}

//------------------------------------------------------------
// Validación de las partículas: se revisan umbrales críticos
//------------------------------------------------------------
bool Simulation::validateParticles() const {
    bool valid = true;
    constexpr double MIN_DENSITY = 1e-14;
    constexpr double MIN_PRESSURE = 1e-14;
    constexpr double MIN_INTERNAL_ENERGY = 1e-14;
    constexpr double MAX_VELOCITY = 1.0 - 1e-10; // c=1

    for (const auto &p : particles) {
        if (p.density < MIN_DENSITY) {
            std::cerr << "[Validation] Densidad muy baja: " << p.density << std::endl;
            valid = false;
        }
        if (p.pressure < MIN_PRESSURE) {
            std::cerr << "[Validation] Presión muy baja: " << p.pressure << std::endl;
            valid = false;
        }
        if (p.specificInternalEnergy < MIN_INTERNAL_ENERGY) {
            std::cerr << "[Validation] Energía interna muy baja: " << p.specificInternalEnergy << std::endl;
            valid = false;
        }
        double vmag = MathUtils::vectorNorm(p.velocity);
        if (vmag >= MAX_VELOCITY) {
            std::cerr << "[Validation] Magnitud de velocidad demasiado alta: " << vmag << std::endl;
            valid = false;
        }
    }
    return valid;
}

double Simulation::computeTotalMass() const {
    double totalMass = 0.0;
    for (const auto &p : particles) {
        if (p.isGhost)
            continue;
        totalMass += p.mass;
    }
    return totalMass;
}

double Simulation::computeTotalEnergy() const {
    double totalEnergy = 0.0;
    for (const auto &p : particles) {
        if (p.isGhost)
            continue;
        double v2 = MathUtils::vectorNormSquared(p.velocity);
        double kinetic = 0.5 * p.mass * v2;
        double internal = p.mass * p.specificInternalEnergy;
        totalEnergy += (kinetic + internal);
    }
    return totalEnergy;
}

void Simulation::appendInvariantsToFile(const std::string& filename, double currentTime, int step) const {
    std::ofstream ofs;
    ofs.open(filename, std::ios::app); // Abrir en modo append para agregar información sin sobrescribir
    if (!ofs) {
        std::cerr << "Error al abrir el archivo de invariantes: " << filename << std::endl;
        return;
    }
    double totalMass = computeTotalMass();
    double totalEnergy = computeTotalEnergy();
    ofs << currentTime << "\t" << step << "\t" << totalMass << "\t" << totalEnergy << "\n";
    ofs.close();
}
