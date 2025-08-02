#include "ShockTubeIC.h"
#include "DensityUpdater.h"
#include "VariableConverter.h"
#include "EquationOfState.h"
#include "Kernel.h"
#include "InitialConditionTypes.h"
#include <iostream>
#include <cmath>

/*───────────────────────────────────────────────────────────*
 *  Utilidades y tabla de parámetros por defecto
 *───────────────────────────────────────────────────────────*/

namespace {

struct SideParams {
    double rho;
    double P;
    std::array<double,3> v = {0.0,0.0,0.0};
};

struct TubeParams {
    SideParams L, R;
    double     hFactor       = 1.2;  // h ≃ hFactor·m/ρ
    double     ghostFraction = 0.2;  // % de partículas fantasma en cada extremo
};

TubeParams baseParams(InitialConditionType t)
{
    TubeParams p;
    switch (t) {
        case InitialConditionType::TEST_RSOD:
            p.L = {10.0,     40.0/3.0};
            p.R = { 1.0,      1.0e-5};
            p.hFactor = 1.4;          break;

        case InitialConditionType::TEST_RSOD2:
            p.L = {10.0,   4000.0/3.0};
            p.R = { 1.0,      1.0e-5};
            p.hFactor = 1.2;          break;

        case InitialConditionType::TEST_RSOD3:
            p.L = {10.0,  40000.0/3.0};
            p.R = { 1.0,      1.0e-5};
            p.hFactor = 1.2;          break;

        case InitialConditionType::TEST_SB:
            p.L = {1.0, 1000.0};
            p.R = {1.0,    0.01};
            p.hFactor = 1.4;          break;

        case InitialConditionType::TEST_PERTUB_SIN:
            p.L = {5.0,  50.0};
            p.R = {2.3,  5.0};
            p.hFactor = 1.2;          break;

        case InitialConditionType::TEST_TRANS_VEL:
            p.L = {1.0, 1000.0, {0.0, 0.0, 0.0}};
            p.R = {1.0,    0.1, {0.0, 0.99, 0.0}};
            p.hFactor = 1.2;          break;

        case InitialConditionType::NR_SOD:
            p.L = {1.0, 1.0};
            p.R = {0.125, 0.1};
            break;

        default:  // place‑holders
            p.L = {1.0, 1.0};
            p.R = {1.0, 1.0e-2};
    }
    return p;
}

} // namespace anónimo

/*───────────────────────────────────────────────────────────*
 *              ShockTubeIC::generate(...)
 *───────────────────────────────────────────────────────────*/

void ShockTubeIC::generate(std::vector<Particle>&          particles,
                           std::shared_ptr<Kernel>         kernel,
                           std::shared_ptr<EquationOfState> eos,
                           int    N,
                           double x_min,
                           double x_max)
{
    const TubeParams P   = baseParams(type_);
    const double     gamma   = eos->getGamma();

    /*—— dominio extendido (20 %) ———————————————*/
    const double x0   = 0.5*(x_min + x_max);          // discontinuidad
    const double extL = x_min - 0.2*(x_max - x_min);
    const double extR = x_max + 0.2*(x_max - x_min);

    const double VL   = x0  - extL;
    const double VR   = extR - x0;
    const double mL   = P.L.rho*VL;
    const double mR   = P.R.rho*VR;
    const double Mtot = mL + mR;

    int NL = static_cast<int>(N*mL/Mtot);
    int NR = N - NL;

    NL += static_cast<int>(P.ghostFraction*NL);
    NR += static_cast<int>(P.ghostFraction*NR);

    const double dxL   = VL/ NL;
    const double dxR   = VR/ NR;
    const double mp    = Mtot / (NL + NR);

    auto makeParticle =
        [&](double x,double rho,double Pgas,const std::array<double,3>& v)
    {
        const double u = Pgas / ((gamma-1.0)*rho);
        Particle p({x,0.0,0.0}, v, mp, u);
        p.density       = rho;
        p.pressure      = Pgas;
        p.h             = P.hFactor*mp/rho;
        p.baryonDensity = rho;
        return p;
    };

    /*—— lado izquierdo (siempre constante) ———————*/
    for(int i=0;i<NL;++i){
        double x = extL + (i+0.5)*dxL;
        particles.push_back(makeParticle(x, P.L.rho, P.L.P, P.L.v));
    }

    /*—— lado derecho: casos especiales / genérico ———*/
    if (type_ == InitialConditionType::TEST_PERTUB_SIN)
    {
        for(int i=0;i<NR;++i){
            double x   = x0 + (i+0.5)*dxR;
            double rho = 2.0 + 0.3*std::sin(50.0*x);     // idéntico al original
            particles.push_back(makeParticle(x, rho, P.R.P, P.R.v));
        }
    }
    else if (type_ == InitialConditionType::TEST_TRANS_VEL)
    {
        for(int i=0;i<NR;++i){
            double x = x0 + (i+0.5)*dxR;
            particles.push_back(makeParticle(x, P.R.rho, P.R.P, P.R.v));
        }
    }
    else   // situación “normal” (constantes a cada lado)
    {
        for(int i=0;i<NR;++i){
            double x = x0 + (i+0.5)*dxR;
            particles.push_back(makeParticle(x, P.R.rho, P.R.P, P.R.v));
        }
    }

    /*—— marcar fantasmas ————————————————*/
    const int ghostsL = static_cast<int>(std::ceil(NL*P.ghostFraction));
    const int ghostsR = static_cast<int>(std::ceil(NR*P.ghostFraction));
    for(int i=0;i<ghostsL;++i) particles[i].isGhost = true;
    for(int i=0;i<ghostsR;++i) particles[particles.size()-1-i].isGhost = true;

    /*—— densidad, presión, variables conservadas ———*/
    DensityUpdater    rhoUpd(1.2,1e-6,false,0.1);
    VariableConverter conv(1e-10);

    #pragma omp parallel for
    for(std::size_t i=0;i<particles.size();++i){
        auto& p = particles[i];
        p.updateDensity(particles, rhoUpd, *kernel);
        p.updatePressure(*eos);
        conv.primitivesToConserved(p,*eos);
    }
}
