#include "physics.cuh"

__global__ void UpdatePositionsKernel( float4* centre, float3* velocities, int N, float dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    float3 g = make_float3(0.0f, -9.81f, 0.0f);
    float3 p = make_float3(centre[i].x, centre[i].y, centre[i].z);
    float r = centre[i].w;

    float3 v = make_float3(velocities[i].x, velocities[i].y, velocities[i].z);

    v.x += g.x * dt;
    v.y += g.y * dt;
    v.z += g.z * dt;

    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;

    centre[i].x = p.x; centre[i].y = p.y; centre[i].z = p.z;
    velocities[i] = v;
}

void CalculationPosition(float4* centre, float3* velocities, int N, float dt) {
    int thread = 256;
    int block = (N + thread - 1) / thread;

    UpdatePositionsKernel<<<block, thread>>>(centre, velocities, N, dt);
}
