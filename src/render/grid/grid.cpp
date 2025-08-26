#include "render/grid/grid.hpp"
#include <array>

Grid::Grid(const glm::vec3& L, const glm::ivec3& N)
    : box_size(L),
      cell_num(glm::max(N, glm::ivec3(1))),
      grid_shader("../shaders/ver_grid.sh", "../shaders/frg_grid.sh") {
}

void Grid::Create() {
    if (VAO || VBO)
        return;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    uploadBaseLine();
    glBindVertexArray(0);
}

void Grid::Destroy() {
    if (VBO) {
        glDeleteBuffers(1, &VBO);
        VBO = 0;
    }
    if (VAO) {
        glDeleteVertexArrays(1, &VAO);
        VAO = 0;
    }
}

void Grid::SetBox(const glm::vec3& L) {
    if (L.x <= 0.f || L.y <= 0.f || L.z <= 0.f)
        return;
    box_size = L;

    if (!VAO || !VBO)
        return;
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    uploadBaseLine();
    glBindVertexArray(0);
}

void Grid::SetCells(const glm::ivec3& N) {
    cell_num = glm::max(N, glm::ivec3(1));
}

void Grid::Render(const glm::mat4& view, const glm::mat4& proj) {
    if (!VAO)
        return;

    grid_shader.use();
    grid_shader.setMat4("uView", view);
    grid_shader.setMat4("uProj", proj);

    // 2) Инварианты для всех проходов: L, N, (цвет)
    grid_shader.setVec3("L", box_size);

    const GLint locN = glGetUniformLocation(grid_shader.GetID(), "N");
    glUniform3i(locN, cell_num.x, cell_num.y, cell_num.z);

    const GLint locColor = glGetUniformLocation(grid_shader.GetID(), "uColor");
    if (locColor >= 0)
        glUniform3f(locColor, 0.8f, 0.8f, 0.8f);

    glBindVertexArray(VAO);

    grid_shader.setInt("axis", 0);
    {
        const GLsizei inst = GLsizei((cell_num.y + 1) * (cell_num.z + 1));
        glDrawArraysInstanced(GL_LINES, 0, 2, inst);
    }

    grid_shader.setInt("axis", 1);
    {
        const GLsizei inst = GLsizei((cell_num.x + 1) * (cell_num.z + 1));
        glDrawArraysInstanced(GL_LINES, 0, 2, inst);
    }

    grid_shader.setInt("axis", 2);
    {
        const GLsizei inst = GLsizei((cell_num.x + 1) * (cell_num.y + 1));
        glDrawArraysInstanced(GL_LINES, 0, 2, inst);
    }

    glBindVertexArray(0);
}

void Grid::uploadBaseLine() {
    const std::array<glm::vec3, 2> verts = {glm::vec3{-box_size.x * 0.5f, 0.f, 0.f},
                                            glm::vec3{+box_size.x * 0.5f, 0.f, 0.f}};

    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
}

void Grid::bindUniforms(int axis) {
    grid_shader.setVec3("L", box_size);
    grid_shader.setInt("axis", axis);

    const GLint locN = glGetUniformLocation(grid_shader.GetID(), "N");
    glUniform3i(locN, cell_num.x, cell_num.y, cell_num.z);

    const GLint locColor = glGetUniformLocation(grid_shader.GetID(), "uColor");
    if (locColor >= 0)
        glUniform3f(locColor, 0.8f, 0.8f, 0.8f);
}
