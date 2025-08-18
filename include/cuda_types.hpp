// include/cuda/cuda_types.hpp
#pragma once
#include <cuda_runtime.h>

struct Vertex {
    float3 pos;
    float3 norm;
};

struct MagneticParticles {
    float3 position;
    float3 magneticMoment;
    float3 force;
};
