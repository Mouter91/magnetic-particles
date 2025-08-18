#pragma once

#include "shapes/shapes.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class MeshGL {
   public:
    void Setup(const MeshData& mesh, GLsizei instanceCount);
    void Cleanup();
    void Render(GLsizei instanceCount) const;
    ~MeshGL();

    cudaGraphicsResource* GetCudaVBOinst() {
        return cudaVBOinstances;
    }

   private:
    GLuint VAO = 0;
    GLuint VBO_mesh = 0;
    GLuint VBO_instances = 0;
    GLuint EBO = 0;
    GLsizei indexCount = 0;

    cudaGraphicsResource* cudaVBOinstances = nullptr;

    void SetupMesh(const MeshData& mesh);
    void CreateInstanceBuffer(GLsizei count);
};
