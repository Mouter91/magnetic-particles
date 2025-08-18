#pragma once
#include "cuda_types.hpp"

void kernel_smooth_normals(int vertexCount, int triangleCount, Vertex* vertices,
                           const unsigned int* indices);
