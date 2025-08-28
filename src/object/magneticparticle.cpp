#include "magneticparticle.hpp"

MagneticParticles::MagneticParticles(const AppConfig* config) {
    SetDomainConfig(&config->domain);
    SetPhysicsConfig(&config->physics);
}

void MagneticParticles::Initialize(int n) {
    particleSpheres.clear();
    particleSpheres.reserve(n);

    std::vector<float4> h_center;
    const float radius = 0.1f;
    const glm::vec3 halfL = 0.5f * domain_conf->L;
    const glm::vec3 lo = -halfL + glm::vec3(radius);
    const glm::vec3 hi = halfL - glm::vec3(radius);

    h_center.reserve(n);

    for (int i = 0; i < n; ++i) {
        glm::vec3 pos;

        const float ux = float(rand()) / RAND_MAX;
        const float uy = float(rand()) / RAND_MAX;
        const float uz = float(rand()) / RAND_MAX;

        pos.x = lo.x + ux * (hi.x - lo.x);
        pos.y = lo.y + uy * (hi.y - lo.y);
        pos.z = lo.z + uz * (hi.z - lo.z);

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
        float3 L = make_float3(domain_conf->L.x, domain_conf->L.y, domain_conf->L.z);
        float3 gravity =
            make_float3(physics_conf->gravity.x, physics_conf->gravity.y, physics_conf->gravity.z);
        UploadParams(L, gravity, physics_conf->bounce, physics_conf->v_eps);
    }

    {
        std::vector<float3> hMom(n, make_float3(0, 0, 1));
        cudaMemcpy(particleData.magneticMoments, hMom.data(), n * sizeof(float3),
                   cudaMemcpyHostToDevice);
    }
}

void MagneticParticles::PhysicsParticles(float dt) {
    cudaGraphicsResource* instRes = mesh_particles.GetCudaVBOinst();

    float4* posr = nullptr;
    size_t bytes = 0;

    cudaGraphicsMapResources(1, &instRes, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&posr, &bytes, instRes);

    const int N = particleData.numParticles;

    CalculationPosition(posr, particleData.velocities, particleData.numParticles, dt);

    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &instRes, 0);
}

void MagneticParticles::Render() {
    mesh_particles.Render(particleData.numParticles);
}

void MagneticParticles::Cleanup() {
    if (particleData.velocities) {
        cudaFree(particleData.velocities);
        particleData.velocities = nullptr;
    }
    if (particleData.forces) {
        cudaFree(particleData.forces);
        particleData.forces = nullptr;
    }
    if (particleData.magneticMoments) {
        cudaFree(particleData.magneticMoments);
        particleData.magneticMoments = nullptr;
    }

    particleData.numParticles = 0;
    particleSpheres.clear();
    particleSpheres.shrink_to_fit();

    cudaDeviceSynchronize();
}

ParticleSoA& MagneticParticles::GetSoA() {
    return particleData;
}

void MagneticParticles::SetDomainConfig(const DomainConfig* config) {
    domain_conf = config;
}
void MagneticParticles::SetPhysicsConfig(const PhysicsConfig* config) {
    physics_conf = config;
}
