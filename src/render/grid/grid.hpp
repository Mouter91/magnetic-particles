#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "core/shader.h"

class Grid {
   public:
    Grid(const glm::vec3& L, const glm::ivec3& N);

    void Create();
    void Destroy();

    void SetBox(const glm::vec3& L);
    void SetCells(const glm::ivec3& N);

    void Render(const glm::mat4& view, const glm::mat4& proj);

   private:
    glm::vec3 box_size;
    glm::ivec3 cell_num;

    GLuint VBO = 0, VAO = 0;
    Shader grid_shader;

    GLint u_L = -1;
    GLint u_N = -1;
    GLint u_axis = -1;

    void uploadBaseLine();
    void bindUniforms(int axis);
};
