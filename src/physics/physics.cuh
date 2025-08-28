#pragma once
#include <iostream>
#include <cuda_runtime.h>

struct ParticleSoA {
    float4* centers;
    float3* velocities;
    float3* forces;
    float3* magneticMoments;
    int numParticles;
};

struct PhysicsParams {
    float3 L;
    float3 gravity;
    float bounce;
    float v_eps;
};

void CalculationPosition(float4* center, float3* velocities, int N, float dt);

void UploadParams(float3 L, float3 gravity, float bounce, float v_eps);
