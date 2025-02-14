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
    // Variables para identificar las partículas fantasma más extremas
    Particle* leftMostGhost = nullptr;
    Particle* rightMostGhost = nullptr;

    // Buscar las partículas fantasma más extremas
    for (auto& p : particles) {
        if (!p.isGhost) continue; // Sólo aplicamos a partículas fantasma

        if (p.position[0] < x_min_) {
            // Buscar la partícula con posición más pequeña (extremo izquierdo)
            if (!leftMostGhost || p.position[0] < leftMostGhost->position[0]) {
                leftMostGhost = &p;
            }
        } else if (p.position[0] > x_max_) {
            // Buscar la partícula con posición más grande (extremo derecho)
            if (!rightMostGhost || p.position[0] > rightMostGhost->position[0]) {
                rightMostGhost = &p;
            }
        }
    }

    // Aplicar condiciones de frontera a las partículas fantasma más extremas
    if (leftMostGhost) {
        leftMostGhost->velocity = {0.0, 0.0, 0.0}; // Fijar velocidad
        leftMostGhost->position[0] = leftMostGhost->position[0]; // Opcional, mantener posición fija
    }

    if (rightMostGhost) {
        rightMostGhost->velocity = {0.0, 0.0, 0.0}; // Fijar velocidad
        rightMostGhost->position[0] = rightMostGhost->position[0]; // Opcional, mantener posición fija
    }
}


void Boundaries::applyOpen(std::vector<Particle>& particles) {
    particles.erase(std::remove_if(particles.begin(), particles.end(),
        [&](const Particle& p) {
            return (p.position[0] < x_min_ || p.position[0] > x_max_);
        }),
        particles.end());
}