#pragma once
#include "shapes/shapes.hpp"
#include "render/render.hpp"
#include "physics/physics.cuh"

class MagneticParticles {
   public:
    void Initialize(int n) {
        particleSpheres.clear();
        particleSpheres.reserve(n);

        std::vector<float4> h_center;
        const float radius = 0.1f;

        for (int i = 0; i < n; ++i) {
            float range = 5.0f;  // диапазон по каждой оси
            glm::vec3 pos;
            pos.x = ((float)rand() / RAND_MAX) * range - range / 2.0f;
            pos.y = ((float)rand() / RAND_MAX) * range - range / 2.0f;
            pos.z = ((float)rand() / RAND_MAX) * range - range / 2.0f;

            particleSpheres.emplace_back(pos, radius);
            h_center.emplace_back(make_float4(pos.x, pos.y, pos.z, radius));
        }

        MeshData sphereMesh = Sphere({0, 0, 0}, radius).GenerateMesh(20);

        mesh_particles.Setup(sphereMesh, static_cast<GLsizei>(n));

        {
            cudaGraphicsResource* instRes = mesh_particles.GetCudaVBOinst();
            float4* d_posr = nullptr;
            size_t bytes = 0;

            cudaGraphicsMapResources(1, &instRes, 0);
            cudaGraphicsResourceGetMappedPointer((void**)&d_posr, &bytes, instRes);

            const size_t need = static_cast<size_t>(n) * sizeof(float4);

            cudaMemcpy(d_posr, h_center.data(), need, cudaMemcpyHostToDevice);

            cudaDeviceSynchronize();
            cudaGraphicsUnmapResources(1, &instRes, 0);
        }

        particleData.numParticles = n;

        cudaMalloc(&particleData.velocities, n * sizeof(float3));
        cudaMalloc(&particleData.forces, n * sizeof(float3));
        cudaMalloc(&particleData.magneticMoments, n * sizeof(float3));

        cudaMemset(particleData.velocities, 0, n * sizeof(float3));
        cudaMemset(particleData.forces, 0, n * sizeof(float3));

        {
            std::vector<float3> hMom(n, make_float3(0, 0, 1));
            cudaMemcpy(particleData.magneticMoments, hMom.data(), n * sizeof(float3),
                       cudaMemcpyHostToDevice);
        }
    }

    void PhysicsParticles() {
        cudaGraphicsResource* instRes = mesh_particles.GetCudaVBOinst();

        float4* posr = nullptr;
        size_t bytes = 0;

        cudaGraphicsMapResources(1, &instRes, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&posr, &bytes, instRes);

        const int N = particleData.numParticles;

        float dt = 0.003f;

        CalculationPosition(posr, particleData.velocities, particleData.numParticles, dt);

        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &instRes, 0);
    }

    void Render() {
        PhysicsParticles();
        mesh_particles.Render(particleData.numParticles);
    }

    void Cleanup();
    ParticleSoA& GetSoA();

   private:
    MeshGL mesh_particles;
    std::vector<Sphere> particleSpheres;
    ParticleSoA particleData;
};
