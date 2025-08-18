#include "render/render.hpp"

MeshGL::~MeshGL() {
    Cleanup();
}

void MeshGL::Setup(const MeshData& mesh, GLsizei instanceCount) {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &EBO);

    SetupMesh(mesh);
    CreateInstanceBuffer(instanceCount);
}

void MeshGL::SetupMesh(const MeshData& mesh) {
    glGenBuffers(1, &VBO_mesh);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_mesh);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(glm::vec3), mesh.vertices.data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(GLuint), mesh.indices.data(),
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    indexCount = static_cast<GLsizei>(mesh.indices.size());

    glBindVertexArray(0);
}

void MeshGL::CreateInstanceBuffer(GLsizei count) {
    glGenBuffers(1, &VBO_instances);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_instances);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
    glEnableVertexAttribArray(1);

    glVertexAttribDivisor(1, 1);
    cudaGraphicsGLRegisterBuffer(&cudaVBOinstances, VBO_instances, cudaGraphicsMapFlagsNone);

    glBindVertexArray(0);
}

void MeshGL::Render(GLsizei instanceCount) const {
    glBindVertexArray(VAO);
    glDrawElementsInstanced(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0, instanceCount);
    glBindVertexArray(0);
}

void MeshGL::Cleanup() {
    if (cudaVBOinstances) {
        cudaGraphicsUnregisterResource(cudaVBOinstances);
        cudaVBOinstances = nullptr;
    }
    glDeleteBuffers(1, &VBO_mesh);
    glDeleteBuffers(1, &VBO_instances);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO);
}
