#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glfw/glfw3.h>
#include <glm/gtc/constants.hpp>

struct MeshData {
    std::vector<glm::vec3> vertices;
    std::vector<GLuint> indices;
};

class Sphere {
   public:
    Sphere(glm::vec3 pos, float r) : position(pos), radius(r) {
    }

    MeshData GenerateMesh(int resolution = 16) const;

    glm::vec3 GetPosition() const {
        return position;
    }
    float GetRadius() const {
        return radius;
    }

    void SetPosition(const glm::vec3& pos) {
        position = pos;
    }
    void SetRadius(float r) {
        radius = r;
    }

   private:
    glm::vec3 position;
    glm::vec3 velocity;
    float radius;
};

struct Cube {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 size;
};
