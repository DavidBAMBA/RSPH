#include "VariableConverter.h"
#include "MathUtils.h"
#include "Logger.h"  // incluir el logger
#include <algorithm>
#include <cmath>
#include <iostream> // Se mantiene para otras operaciones, pero ya no se usan para errores
#include <array>
#include "Globals.h"

// -------------------------------------------------------------------------------------
// Constructor
// -------------------------------------------------------------------------------------
VariableConverter::VariableConverter(double tolerance)
    : tolerance_(tolerance)
{}

// Ejemplo de constantes comunes
constexpr double MIN_PRESSURE = 1e-12;
constexpr double MIN_DENSITY  = 1e-12;
constexpr double MIN_DENOM    = 1e-12;
constexpr double MIN_ENERGY   = 1e-12;
constexpr double MAX_V2       = 1.0 - 1e-10;

void VariableConverter::primitivesToConserved(Particle& particle, const EquationOfState& eos) const {
    // Verificar presión
    if (particle.pressure < 0.0) {
        logError("primitivesToConserved", "Presión negativa detectada", particle.pressure);
    }

    // Factor Lorentz
    double v2 = MathUtils::vectorNormSquared(particle.velocity);
    if (v2 >= 1.0) {
        logError("primitivesToConserved", "Velocidad mayor o igual a 1.0. Ajustando...");
        double factor = std::sqrt(MAX_V2 / v2);
        for (int i = 0; i < 3; ++i) {
            particle.velocity[i] *= factor;
        }
        v2 = MAX_V2;
    }
    double gamma = 1.0 / std::sqrt(1.0 - v2);

    // Variables primitivas
    double n = std::max(particle.density, MIN_DENSITY);
    double u = std::max(particle.specificInternalEnergy, MIN_PRESSURE);
    double P = std::max(particle.pressure, MIN_PRESSURE);

    // Cálculo de X
    double X = 1.0 + u + P / n;

    // Conservadas
    particle.baryonDensity = gamma * n;

    std::array<double, 3> qVec = {};
    for (int i = 0; i < 3; ++i) {
        qVec[i] = gamma * particle.velocity[i] * X;
    }
    particle.specificMomentum = qVec;

    particle.specificEnergy = gamma * X - P / (gamma * n);

    if (particle.specificEnergy < 0.0) {
        logError("primitivesToConserved", "Energía específica negativa calculada", particle.specificEnergy);
    }
}

void VariableConverter::conservedToPrimitives(Particle& particle, const EquationOfState& eos) const {
    
    double P_sol = solvePressure(particle, eos);
    if (P_sol <= 0.0) {
        logError("conservedToPrimitives", "Presión no válida después de resolver", P_sol);
        throw std::runtime_error("Convergencia de presión fallida en conservedToPrimitives");

    }

    double N_ = std::max(particle.baryonDensity, MIN_DENSITY);
    double q2 = MathUtils::vectorNormSquared(particle.specificMomentum);
    double eHat = particle.specificEnergy;

    if (N_ < 0.0 || eHat < 0.0) {
        logError("conservedToPrimitives",
                 "Valores críticos detectados antes de calcular γ: N = " + std::to_string(N_) +
                 ", eHat = " + std::to_string(eHat));
        throw std::runtime_error("Densidad o energía negativa detectada en conservedToPrimitives");

    }

    double denom = eHat + P_sol / N_;
    denom = std::max(denom, MIN_DENOM);

    double vc2 = q2 / (denom * denom);
    if (vc2 >= MAX_V2) {
        vc2 = MAX_V2;
        logError("conservedToPrimitives", "Magnitud de velocidad ajustada cerca de 1", vc2);
    }

    double gamma = 1.0 / std::sqrt(1.0 - vc2);
    double n = N_ / gamma;
    double u = (eHat / gamma)
             + (P_sol * (1.0 - gamma * gamma)) / (N_ * gamma)
             - 1.0;

    // Extrae la posición de la partícula (3D)
    double x = particle.position[0];
    double y = particle.position[1];
    double z = particle.position[2];
    std::array<double, 3> v;

    if (q2 > 1e-14) { // Evitar división por cero
        for (int i = 0; i < 3; ++i) {
            v[i] = particle.specificMomentum[i] / denom; 
        }
    } else {
        v = {0.0, 0.0, 0.0};
    }

    // --- Mensajes de log con posición incluida --- //

    if (n < 0.0) {
        std::string msg = "Densidad negativa calculada: n=" + std::to_string(n)
                        + ", gamma=" + std::to_string(gamma)
                        + ", eHat=" + std::to_string(eHat)
                        + ", P_sol=" + std::to_string(P_sol)
                        + ", N=" + std::to_string(N_)
                        + ", vc2=" + std::to_string(vc2)
                        + ", pos=(" + std::to_string(x) 
                        + ", " + std::to_string(y)
                        + ", " + std::to_string(z) + ")";
        logError("conservedToPrimitives", msg);
        throw std::runtime_error("Densidad negativa en conservedToPrimitives");

    }

    if (u < 0.0) {
        std::string msg = "Energía específica interna negativa calculada: u=" + std::to_string(u)
                        + ", gamma=" + std::to_string(gamma)
                        + ", eHat=" + std::to_string(eHat)
                        + ", P_sol=" + std::to_string(P_sol)
                        + ", N_=" + std::to_string(N_)
                        + ", vc2=" + std::to_string(vc2)
                        + ", pos=(" + std::to_string(x)
                        + ", " + std::to_string(y)
                        + ", " + std::to_string(z) + ")";
        logError("conservedToPrimitives", msg);
        throw std::runtime_error("Energía interna negativa en conservedToPrimitives");

    }

    // Asignar las variables primitivas calculadas
    particle.pressure = P_sol;
    particle.density  = n;
    particle.specificInternalEnergy = u;
    particle.velocity = v;
}


double VariableConverter::solvePressure(const Particle& particle, const EquationOfState& eos) const {
    // bracketing + bisección + Newton
    //double P_guess2 = (5.0 / 3.0 - 1.0) * particle.density * particle.specificInternalEnergy;
    double P_guess = particle.pressure;
    double P_sol = root_finding(P_guess, particle, eos, tolerance_);

    // Si root_finding devolvió negativo, no se pudo encontrar solución
    if (P_sol < 0.0) {
        logError("solvePressure", "root_finding no pudo converger; intentando con presión calculada", P_guess);
        /* P_sol = root_finding(P_guess2, particle, eos, tolerance_);
        if (P_sol < 0.0) {
            logError("solvePressure", "root_finding no pudo converger; P_sol negativo, p_guess", P_guess2);
            return particle.pressure;
        } */
    }
    return P_sol;
}

double VariableConverter::root_finding(double P_guess, const Particle& particle,
                                         const EquationOfState& eos, double tolerance) const {
    // Función a evaluar: f(P) = P_eos(n,u) - P,
    auto eval_f = [&](double P) -> double {
        return computeFunctionAndDerivative(P, particle, eos).first;
    };
    //std::cout<<"\n\n ----------AQUIII------------\n";

    // 1) Bracketing robusto: expandir hacia abajo y hacia arriba para encontrar un cambio de signo.
    double P_left  = std::max(P_guess, 1e-14);
    double f_left  = eval_f(P_left);

    if (std::fabs(f_left) < tolerance) return P_left;  // La conjetura inicial ya es buena

    double P_right = P_left;
    double f_right = f_left;
    
    // Parámetros de expansión (ajustados según la magnitud de f_left)
    const int max_bracket_steps = 100;
    double factorDown = (std::fabs(f_left) > 1e-5) ? 0.5 : 0.1;
    double factorUp   = (std::fabs(f_left) > 1e-5) ? 2.0 : 3.0;
    const double minBound = 1e-14;
    const double maxBound = 1e14;
    
    bool foundBracket = false;
    for (int i = 0; i < max_bracket_steps; ++i) {
        // Expansión hacia abajo:
        double testPDown = P_left * factorDown;
        if (testPDown < minBound)
            testPDown = minBound;
        double f_down = eval_f(testPDown);
        if (f_down * f_left < 0.0) {
            // Cambio de signo detectado entre testPDown y P_left
            P_right = P_left;
            f_right = f_left;
            P_left  = testPDown;
            f_left  = f_down;
            foundBracket = true;
            break;
        } else {
            P_left = testPDown;
            f_left = f_down;
        }
        // Expansión hacia arriba:
        double testPUp = P_right * factorUp;
        if (testPUp > maxBound)
            testPUp = maxBound;
        double f_up = eval_f(testPUp);
        if (f_up * f_right < 0.0) {
            P_left  = P_right;
            f_left  = f_right;
            P_right = testPUp;
            f_right = f_up;
            foundBracket = true;
            break;
        } else {
            P_right = testPUp;
            f_right = f_up;
        }
        // Si en algún extremo ya estamos dentro de la tolerancia:
        if (std::fabs(f_left) < tolerance)
            return P_left;
        if (std::fabs(f_right) < tolerance)
            return P_right;
    }
    if (!foundBracket) {
        logError("root_finding", "No se encontró intervalo de bracketing");
    }

    // 2) Aplicar el método de Brent con un máximo de 10 iteraciones.
    auto brentMethod = [&](double a, double b, double tol, int maxIter)
        -> std::pair<double, bool>
    {
        bool ok = false;
        double fa = eval_f(a), fb = eval_f(b);
        if (fa * fb > 0.0)
            return {a, false}; // No hay cambio de signo
        double c = a, fc = fa;
        double d = b - a, e = d;
        for (int iter = 1; iter <= maxIter; ++iter) {
            if (std::fabs(fa) < std::fabs(fb)) {
                std::swap(a, b);
                std::swap(fa, fb);
            }
            double tol_act = 1e-14 * std::fabs(b) + tol * 0.5;
            double m = 0.5 * (c - b);
            if (std::fabs(m) <= tol_act || std::fabs(fb) < tol) {
                ok = true;
                return {b, ok};
            }
            if (std::fabs(e) > tol_act && std::fabs(fa) > std::fabs(fb)) {
                double s = fb / fa;
                double p, q;
                if (a == c) { // método de la secante
                    p = 2.0 * m * s;
                    q = 1.0 - s;
                } else { // interpolación cuadrática inversa
                    double r = fb / fc;
                    double t = fa / fc;
                    p = s * (2.0 * m * r * (r - t) - (b - a) * (t - 1.0));
                    q = (r - 1.0)*(t - 1.0)*(s - 1.0);
                }
                if (p > 0.0)
                    q = -q;
                p = std::fabs(p);
                if (2.0 * p < std::min(3.0 * m * q - std::fabs(tol_act * q), std::fabs(e * q))) {
                    e = d;
                    d = p / q;
                } else {
                    e = d;
                    d = m;
                }
            } else {
                e = d;
                d = m;
            }
            a = b;
            fa = fb;
            if (std::fabs(d) > tol_act)
                b += d;
            else
                b += (m > 0.0 ? tol_act : -tol_act);
            fb = eval_f(b);
            if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
                c = a;
                fc = fa;
                e = d = b - a;
            }
        }
        logError("root_finding", "Brent no convergió en maxIter");
        return {b, false};
    };

    auto [pBrent, brentConverged] = brentMethod(P_left, P_right, tolerance, 20);
    if (brentConverged && std::fabs(eval_f(pBrent)) < tolerance)
        return pBrent;

    // 3) Fallback: Bisección con un máximo alto de iteraciones
    double f_left2  = eval_f(P_left);
    double f_right2 = eval_f(P_right);
    const int max_bisection = 20000;
    double pMid = 0.5 * (P_left + P_right);
    double fMid = eval_f(pMid);
    for (int i = 0; i < max_bisection; ++i) {
        if (std::fabs(fMid) < tolerance)
            break;
        if (f_left * fMid < 0.0) {
            P_right = pMid;
            f_right = fMid;
        } else {
            P_left = pMid;
            f_left = fMid;
        }
        pMid = 0.5 * (P_left + P_right);
        fMid = eval_f(pMid);
        if (std::fabs(P_right - P_left) < tolerance * pMid)
            break;
    }

    // 4) Refinamiento final con Newton (opcional)
    double pNewton = pMid;
    const int max_newton = 1000;
    const double damping = 0.1; // o incluso un valor menor si es necesario

    for (int i = 0; i < max_newton; ++i) {
        auto [f_val, df_val] = computeFunctionAndDerivative(pNewton, particle, eos);
        if (std::fabs(df_val) < 1e-15) {
            logError("root_finding", "Newton: derivada demasiado pequeña");
            break;
        }
        double pNext = pNewton - damping * f_val / df_val;
        if (pNext <= 1e-14)
            break;
        double rel_step = std::fabs((pNext - pNewton) / std::max(pNewton, 1e-14));
        pNewton = pNext;
        if (std::fabs(f_val) < tolerance && rel_step < tolerance)
            break;
    }
    if (pNewton <= 0.0 || std::fabs(eval_f(pNewton)) > 1e8){
        throw std::runtime_error("Fallo en la convergencia de presión en root_finding");

        return -1.0;
    }

    return pNewton;
}

double VariableConverter::root_finding2(double P_guess, const Particle& particle,
                                         const EquationOfState& eos, double tolerance) const {
    // Se parte de una conjetura inicial robusta
    double pNewton = std::max(P_guess, 1e-14);
    const int max_newton = 1000;
    const double damping = 1.0; // Puedes ajustar este factor si es necesario
    
    for (int i = 0; i < max_newton; ++i) {
        // Evaluar la función y su derivada en pNewton:
        auto [f_val, df_val] = computeFunctionAndDerivative(pNewton, particle, eos);
        
        // Si la derivada es demasiado pequeña, se interrumpe la iteración
        if (std::fabs(df_val) < 1e-15) {
            logError("root_finding", "Newton: derivada demasiado pequeña");
            break;
        }
        
        // Cálculo de la corrección según Newton-Raphson
        double pNext = pNewton - damping * f_val / df_val;
        if (pNext <= 1e-14)
            pNext = 1e-14; // Evitar valores no físicos
        
        double rel_step = std::fabs((pNext - pNewton) / std::max(pNewton, 1e-14));
        pNewton = pNext;
        
        // Si se cumple la tolerancia en función y en el paso relativo, se considera convergido
        if (std::fabs(f_val) < tolerance && rel_step < tolerance)
            break;
    }
    
    // Verificación final: si no convergió o se obtuvo un valor no físico, se lanza excepción
    if (pNewton <= 0.0 || std::fabs(computeFunctionAndDerivative(pNewton, particle, eos).first) > 1e8) {
        throw std::runtime_error("Fallo en la convergencia de presión en root_finding");
        return -1.0;
    }
    
    return pNewton;
}


std::pair<double, double> VariableConverter::computeFunctionAndDerivative(double P, 
                                                                          const Particle& particle, 
                                                                          const EquationOfState& eos) const {
    double N_   = particle.baryonDensity;
    double eHat = particle.specificEnergy;
    double q2   = MathUtils::vectorNormSquared(particle.specificMomentum);
    double GAMMA = eos.getGamma();

    double denom = eHat + P / std::max(N_, 1e-14);

    if (denom <= 1e-12) {
        denom = 1e-12;
        logError("computeFunctionAndDerivative", "denom ajustado a 1e-12 para evitar división por cero");
    }

    double v2 = q2 / (denom * denom);
    double one_minus_v2 = 1.0 - v2;
    if (one_minus_v2 <= 0.0) {
        logError("computeFunctionAndDerivative", "Error: 1 - v2 <= 0. (v2 = " + std::to_string(v2) + ")");
        // Registrar la posición de la partícula para ayudar en la depuración
        logError("computeFunctionAndDerivative", "DEBUG: Posición de la partícula: (" +
                 std::to_string(particle.position[0]) + ", " +
                 std::to_string(particle.position[1]) + ", " +
                 std::to_string(particle.position[2]) + ")");
        return {1e12, 1e-12};
    }

    double gamma = 1.0 / std::sqrt(one_minus_v2);
    double n = N_ / gamma;
    // u = eHat/gamma + [P*(1 - gamma^2)]/(N_*gamma) - 1
    double u = (eHat / gamma) + (P * (1.0 - gamma * gamma) / (N_ * gamma)) - 1.0;
    // P_eos = (GAMMA - 1)*n*u
    double P_eos = (GAMMA - 1.0) * n * u;

    
     // Comprobar si la densidad es negativa
    if (n < 0.0) {
        logError("computeFunctionAndDerivative", "Error: Densidad negativa calculada", n);
        n = MIN_DENSITY;
    }

    // u = eHat/gamma + [P*(1 - gamma^2)]/(N_*gamma) - 1
    //double u = (eHat / gamma) + (P * (1.0 - gamma * gamma) / (N_ * gamma)) - 1.0;

    if (u < 0.0) {
        logError("computeFunctionAndDerivative", "Error: energía u negativa calculada", u);
        u = MIN_ENERGY;
    }
    
    // P_eos = (GAMMA - 1)*n*u
    //double P_eos = (GAMMA - 1.0) * n * u;

    if (P_eos < 0.0) {
        logError("computeFunctionAndDerivative", "Error: Presión_eos negativa calculada", P_eos);
        P_eos = MIN_PRESSURE;
    } 

    // f(P) = P_eos - P
    double f = P_eos - P;

    // Derivadas
    double dP_dn = (GAMMA - 1.0) * u;
    double dP_du = (GAMMA - 1.0) * n;
    double dn_dP = (gamma * q2) / (denom * denom * denom);
    double du_dP = (N_ * q2 * P * gamma * gamma * gamma) / ((P + eHat * N_) * (P + eHat * N_) * (P + eHat * N_));

    // Extraer la posición de la partícula (3D)
    double x = particle.position[0];
    double y = particle.position[1];
    double z = particle.position[2];
    
    // Comprobar si du_dP es NaN o infinito
    if (std::isnan(du_dP) || std::isinf(du_dP)) {
        logError("computeFunctionAndDerivative", "Error: du_dP o dn_dP es inválido (NaN o infinito)");
        du_dP = 1.0e-12;
    }

    // df/dP = (dP_dn * dn_dP) + (dP_du * du_dP) - 1
    double df_dP = (dP_dn * dn_dP) + (dP_du * du_dP) - 1.0;
    //double cs2 = (GAMMA-1.0)*GAMMA*u / (1.0+GAMMA*u);

    //double df_dP = (dP_dn * dn_dP) + (dP_du * du_dP) - 1.0;
    //double df_dP  =MathUtils::vectorNormSquared(particle.velocity)*cs2 -1.0;
    // Comprobar si la derivada es NaN o infinito
    if (std::isnan(df_dP) || std::isinf(df_dP)) {
        logError("computeFunctionAndDerivative", "Error: df_dP es inválido (NaN o infinito)");
        df_dP = 1e-12; // Asignar un valor pequeño para evitar problemas
    }
    
    // Registrar información de depuración en el log solo si u, n o P_eos son negativos
    if (u < 0.0 || n < 0.0 || P_eos < 0.0) {
        
        std::string debugMsg = "DEBUG: Posición de la partícula: (" + std::to_string(x) + ", " +
                               std::to_string(y) + ", " + std::to_string(z) + ")\n" +
                               "DEBUG: N_ = " + std::to_string(N_) + ", eHat = " + std::to_string(eHat) +
                               ", q2 = " + std::to_string(q2) + ", GAMMA (de EoS) = " + std::to_string(GAMMA) + "\n" +
                               "DEBUG: denom = " + std::to_string(denom) + ", v2 = " + std::to_string(v2) +
                               ", one_minus_v2 = " + std::to_string(one_minus_v2) + "\n" +
                               "DEBUG: gamma = " + std::to_string(gamma) + ", n calculado = " + std::to_string(n) + "\n" +
                               "DEBUG: u calculado = " + std::to_string(u) + "\n" +
                               "DEBUG: P_eos = " + std::to_string(P_eos) + "\n" +
                               "DEBUG: f(P) = " + std::to_string(f) + ", df/dP = " + std::to_string(df_dP) + ", P_guess = " + std::to_string(particle.pressure);
        logError("computeFunctionAndDerivative_DEBUG", debugMsg);
        throw std::runtime_error("P_eos, n, u negativa en conservedToPrimitives");

    }

    return {f, df_dP};
}
