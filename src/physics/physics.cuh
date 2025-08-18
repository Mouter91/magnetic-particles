#pragma once
#include <iostream>

struct ParticleSoA {
    float4* centers;
    float3* velocities;
    float3* forces;
    float3* magneticMoments;
    int numParticles;
};

void CalculationPosition(float4* center, float3* velocities, int N, float dt);
