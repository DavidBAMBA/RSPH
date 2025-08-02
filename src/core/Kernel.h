// Kernel.h
#ifndef KERNEL_H
#define KERNEL_H

#include <functional>
#include <array>

enum class KernelType {
    M3, // Cubic kernel (M_3)
    M5, // Quartic kernel (M_5)
    M6  // Quintic kernel (M_6)
};

class Kernel {
public:
    // Constructor que recibe el tipo de kernel
    Kernel(KernelType type = KernelType::M6);

    // Método para cambiar el kernel si se desea en tiempo de ejecución
    void setKernelType(KernelType type);

    // Funciones públicas para el kernel y sus derivados
    double W(double r, double h) const;
    double dWdh(double r, double h) const;
    double F(double q, double h) const;
    std::array<double, 3> gradW(const std::array<double, 3>& r_vec, double h) const;

private:
    // Normalization constant (para 1D en este ejemplo)
    double computeSigma() const;

    // Punteros (o std::function) a las funciones f(q) y df(q)
    std::function<double(double)> f;
    std::function<double(double)> df;

    // La constante de normalización se asigna automáticamente según el kernel
    double sigma_const;

    // Método interno para configurar las funciones según el tipo de kernel
    void configureKernel(KernelType type);
};

#endif // KERNEL_H
