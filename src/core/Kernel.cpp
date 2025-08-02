// Kernel.cpp
#include "Kernel.h"
#include <cmath>
#include <algorithm>
#include "MathUtils.h"  

Kernel::Kernel(KernelType type) {
    configureKernel(type);
}

void Kernel::setKernelType(KernelType type) {
    configureKernel(type);
}

double Kernel::computeSigma() const {
    return sigma_const;
}

double Kernel::W(double r, double h) const {
    double q = r / h;
    return (computeSigma() / h) * f(q);
}

double Kernel::dWdh(double r, double h) const {
    double q = r / h;
    // Ejemplo de derivada: dW/dh = -sigma/h^2 [f(q) + q df/dq]
    return -(computeSigma() / (h * h)) * (f(q) + q * df(q));
}

double Kernel::F(double q, double h) const {
    // F = sigma/h^2 * df/dq
    return (computeSigma() / (h * h)) * df(q);
}

std::array<double, 3> Kernel::gradW(const std::array<double, 3>& r_vec, double h) const {
    double r = MathUtils::vectorNorm(r_vec);
    std::array<double, 3> r_hat = MathUtils::hatVector(r_vec);
    if (r == 0.0) return {0.0, 0.0, 0.0};
    double q = r / h;
    double F_value = F(q, h);
    return {F_value * r_hat[0], F_value * r_hat[1], F_value * r_hat[2]};
}

// --- Configurar el kernel según el tipo seleccionado ---
void Kernel::configureKernel(KernelType type) {
    switch (type) {
        case KernelType::M3:
            // M3: Cubic kernel
            // sigma (1D): 2/3
            sigma_const = 2.0 / 3.0;
            f = [](double q) -> double {
                double q2 = 2.0 - q;
                double q1 = 1.0 - q;
                if (q >= 0.0 && q < 1.0) {
                    return 0.25 * std::pow(q2, 3) - std::pow(q1, 3);
                } else if (q >= 1.0 && q < 2.0) {
                    return 0.25 * std::pow(q2, 3);
                }
                return 0.0;
            };
            df = [](double q) -> double {
                double q2 = 2.0 - q;
                double q1 = 1.0 - q;
                if (q >= 0.0 && q < 1.0) {
                    return -0.75 * std::pow(q2, 2) + 3.0 * std::pow(q1, 2);
                } else if (q >= 1.0 && q < 2.0) {
                    return -0.75 * std::pow(q2, 2);
                }
                return 0.0;
            };
            break;
        case KernelType::M5:
            // M5: Quartic kernel
            // sigma (1D): 1/24  
            sigma_const = 1.0 / 24.0;
            f = [](double q) -> double {
                double q25 = 2.5 - q;
                double q15 = 1.5 - q;
                double q05 = 0.5 - q;
                if (q >= 0.0 && q < 0.5) {
                    return std::pow(q25, 4) - 5.0 * std::pow(q15, 4) + 10.0 * std::pow(q05, 4);
                } else if (q >= 0.5 && q < 1.5) {
                    return std::pow(q25, 4) - 5.0 * std::pow(q15, 4);
                } else if (q >= 1.5 && q < 2.5) {
                    return std::pow(q25, 4);
                }
                return 0.0;
            };
            df = [](double q) -> double {
                double q25 = 2.5 - q;
                double q15 = 1.5 - q;
                double q05 = 0.5 - q;
                if (q >= 0.0 && q < 1.0) {
                    return -4.0 * std::pow(q25, 3) + 20.0 * std::pow(q15, 3) - 40.0 * std::pow(q05, 3);
                } else if (q >= 1.0 && q < 2.0) {
                    return -4.0 * std::pow(q25, 3) + 20.0 * std::pow(q15, 3);
                } else if (q >= 2.0 && q < 3.0) {
                    return -4.0 * std::pow(q25, 3);
                }
                return 0.0;
            };
            break;
        case KernelType::M6:
            // M6: Quintic kernel
            sigma_const = 1.0 / 120.0;
            f = [](double q) -> double {
                double q3 = 3.0 - q;
                double q2 = 2.0 - q;
                double q1 = 1.0 - q;
                if (q >= 0.0 && q < 1.0) {
                    return std::pow(q3, 5) - 6.0 * std::pow(q2, 5) + 15.0 * std::pow(q1, 5);
                } else if (q >= 1.0 && q < 2.0) {
                    return std::pow(q3, 5) - 6.0 * std::pow(q2, 5);
                } else if (q >= 2.0 && q < 3.0) {
                    return std::pow(q3, 5);
                }
                return 0.0;
            };
            df = [](double q) -> double {
                double q3 = 3.0 - q;
                double q2 = 2.0 - q;
                double q1 = 1.0 - q;
                if (q >= 0.0 && q < 1.0) {
                    return -5.0 * std::pow(q3, 4) + 30.0 * std::pow(q2, 4) - 75.0 * std::pow(q1, 4);
                } else if (q >= 1.0 && q < 2.0) {
                    return -5.0 * std::pow(q3, 4) + 30.0 * std::pow(q2, 4);
                } else if (q >= 2.0 && q < 3.0) {
                    return -5.0 * std::pow(q3, 4);
                }
                return 0.0;
            };
            break;
        default:
            // Por defecto, se usa el M6
            sigma_const = 1.0 / 120.0;
            f = [](double q) -> double { return 0.0; };
            df = [](double q) -> double { return 0.0; };
            break;
    }
}
