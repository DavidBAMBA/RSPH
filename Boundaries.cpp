#include "Boundaries.h"
#include <algorithm>

Boundaries::Boundaries(double x_min, double x_max, BoundaryType type)
    : x_min_(x_min), x_max_(x_max), type_(type) {}

void Boundaries::apply(std::vector<Particle>& particles) {
    switch (type_) {
        case BoundaryType::PERIODIC:
            applyPeriodic(particles);
            break;
        case BoundaryType::FIXED:
            applyFixed(particles);
            break;
        case BoundaryType::OPEN:
            applyOpen(particles);
            break;
    }
}

void Boundaries::applyPeriodic(std::vector<Particle>& particles) {
    for (auto& p : particles) {
        if (p.position[0] < x_min_) {
            p.position[0] = x_max_ - (x_min_ - p.position[0]);
        } else if (p.position[0] > x_max_) {
            p.position[0] = x_min_ + (p.position[0] - x_max_);
        }
    }
}

void Boundaries::applyFixed(std::vector<Particle>& particles) {
    // Recolectar todas las partículas fantasma
    std::vector<Particle*> ghostParticles;
    for (auto& p : particles) {
        if (p.isGhost)
            ghostParticles.push_back(&p);
    }

    // Ordenar las partículas fantasma por posición en x (de menor a mayor)
    std::sort(ghostParticles.begin(), ghostParticles.end(), [](const Particle* a, const Particle* b) {
        return a->position[0] < b->position[0];
    });

    // Fijar las 5 partículas más a la izquierda
    size_t numLeft = std::min<size_t>(5, ghostParticles.size());
    for (size_t i = 0; i < numLeft; ++i) {
        Particle* ghost = ghostParticles[i];
        ghost->velocity = {0.0, 0.0, 0.0}; // Fijar la velocidad a cero
    }

    // Fijar las 5 partículas más a la derecha
    size_t totalGhosts = ghostParticles.size();
    size_t numRight = std::min<size_t>(5, ghostParticles.size());
    for (size_t i = 0; i < numRight; ++i) {
        Particle* ghost = ghostParticles[totalGhosts - 1 - i];
        ghost->velocity = {0.0, 0.0, 0.0}; // Fijar la velocidad a cero
    }
}


void Boundaries::applyOpen(std::vector<Particle>& particles) {
    particles.erase(std::remove_if(particles.begin(), particles.end(),
        [&](const Particle& p) {
            return (p.position[0] < x_min_ || p.position[0] > x_max_);
        }),
        particles.end());
}