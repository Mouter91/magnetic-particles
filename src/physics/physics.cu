#include "physics.cuh"

__constant__ PhysicsParams cParams;

__global__ void UpdatePositionsKernel(float4* centre, float3* velocities, int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    const float3 g = cParams.gravity;
    float3 p = make_float3(centre[i].x, centre[i].y, centre[i].z);
    float3 v = make_float3(velocities[i].x, velocities[i].y, velocities[i].z);

    v.x += g.x * dt;
    v.y += g.y * dt;
    v.z += g.z * dt;

    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;

    const float r = centre[i].w;
    const float minX = -0.5f * cParams.L.x + r;
    const float maxX = 0.5f * cParams.L.x - r;

    if (p.x < minX) {
        const float pen = minX - p.x;
        p.x = minX + pen;
        v.x = -v.x * cParams.bounce;
    } else if (p.x > maxX) {
        const float pen = p.x - maxX;
        p.x = maxX - pen;
        v.x = -v.x * cParams.bounce;
    }

    const float minY = -0.5f * cParams.L.y + r;
    const float maxY = 0.5f * cParams.L.y - r;

    if (p.y < minY) {
        const float pen = minY - p.y;
        p.y = minY + pen;
        v.y = -v.y * cParams.bounce;
    } else if (p.y > maxY) {
        const float pen = p.y - maxY;
        p.y = maxY - pen;
        v.y = -v.y * cParams.bounce;
    }

    const float minZ = -0.5f * cParams.L.z + r;
    const float maxZ = 0.5f * cParams.L.z - r;

    if (p.z < minZ) {
        const float pen = minZ - p.z;
        p.z = minZ + pen;
        v.z = -v.z * cParams.bounce;
    } else if (p.z > maxZ) {
        const float pen = p.z - maxZ;
        p.z = maxZ - pen;
        v.z = -v.z * cParams.bounce;
    }
    p.x = fminf(fmaxf(p.x, minX), maxX);
    p.y = fminf(fmaxf(p.y, minY), maxY);
    p.z = fminf(fmaxf(p.z, minZ), maxZ);

    if (fabsf(v.x) < cParams.v_eps)
        v.x = 0.0f;
    if (fabsf(v.y) < cParams.v_eps)
        v.y = 0.0f;
    if (fabsf(v.z) < cParams.v_eps)
        v.z = 0.0f;

    centre[i].x = p.x;
    centre[i].y = p.y;
    centre[i].z = p.z;
    velocities[i] = v;
}

void CalculationPosition(float4* centre, float3* velocities, int N, float dt) {
    int thread = 256;
    int block = (N + thread - 1) / thread;

    UpdatePositionsKernel<<<block, thread>>>(centre, velocities, N, dt);
}

void UploadParams(float3 L, float3 gravity, float bounce, float v_eps) {
    PhysicsParams param;
    param.L = L;
    param.gravity = gravity;
    param.bounce = bounce;
    param.v_eps = v_eps;
    cudaMemcpyToSymbol(cParams, &param, sizeof(param));
}
