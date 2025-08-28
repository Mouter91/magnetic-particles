#pragma once
#include "shapes/shapes.hpp"
#include "render/render.hpp"
#include "physics/physics.cuh"
#include "core/config.h"

class MagneticParticles {
   public:
    MagneticParticles(const AppConfig* config);
    ~MagneticParticles() {
        Cleanup();
    }

    MagneticParticles(const MagneticParticles&) = delete;
    MagneticParticles& operator=(const MagneticParticles&) = delete;

    MagneticParticles(MagneticParticles&&) noexcept = delete;
    MagneticParticles& operator=(MagneticParticles&&) noexcept = delete;

    void Initialize(int n);
    void PhysicsParticles(float dt);
    void Render();
    void Cleanup();
    ParticleSoA& GetSoA();

    void SetDomainConfig(const DomainConfig* config);
    void SetPhysicsConfig(const PhysicsConfig* config);

   private:
    MeshGL mesh_particles;
    std::vector<Sphere> particleSpheres;
    ParticleSoA particleData;

    const DomainConfig* domain_conf = nullptr;
    const PhysicsConfig* physics_conf = nullptr;
};
