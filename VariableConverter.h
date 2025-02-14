#ifndef VARIABLECONVERTER_H
#define VARIABLECONVERTER_H

#include <array>
#include <cmath>
#include <iostream>
#include "Particle.h"
#include "EquationOfState.h"
#include "MathUtils.h"

/**
 * @brief Clase VariableConverter para convertir entre variables primitivas (rho, P, u, v)
 *        y variables conservadas (N, q, eHat) de cada partícula.
 */
class VariableConverter {
public:
    /**
     * @brief Constructor con tolerancia para los métodos iterativos (Newton-Raphson, bisección).
     * @param tolerance Tolerancia numérica para la convergencia.
     */
    explicit VariableConverter(double tolerance);

    /**
     * @brief Convierte las variables primitivas (density, pressure, internalEnergy, velocity)
     *        de la partícula en variables conservadas (baryonDensity=N, specificMomentum=q, specificEnergy=eHat).
     * @param particle  Referencia a la partícula donde leer las primitivas y escribir las conservadas.
     * @param eos       Ecuación de estado (si se necesitara).
     */
    void primitivesToConserved(Particle& particle, const EquationOfState& eos) const;

    /**
     * @brief Convierte las variables conservadas (N, q, eHat) en primitivas (rho, P, u, v)
     *        resolviendo la presión con Newton-Raphson (o bisección si falla).
     * @param particle  Referencia a la partícula donde leer las conservadas y escribir las primitivas.
     * @param eos       Ecuación de estado.
     */
    void conservedToPrimitives(Particle& particle, const EquationOfState& eos) const;

private:
    double tolerance_; ///< Tolerancia para la convergencia numérica
    
    /// Método principal para resolver la presión: Newton-Raphson con fallback a bisección
    double solvePressure( const Particle& particle, const EquationOfState& eos) const;

    /// Calcula f(P) y su derivada df/dP para Newton-Raphson
    std::pair<double, double> computeFunctionAndDerivative(double P, const Particle& particle, const EquationOfState& eos) const;

    /// Método Iterativo para encontrar la presión
    double root_finding(double P_guess, const Particle& particle, const EquationOfState& eos, double tolerance) const;
    double root_finding2(double P_guess, const Particle& particle, const EquationOfState& eos, double tolerance) const;

};

#endif // VARIABLECONVERTER_H
