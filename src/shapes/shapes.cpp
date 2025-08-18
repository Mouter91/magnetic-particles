#include <shapes/shapes.hpp>


MeshData Sphere::GenerateMesh(int resolution) const {
    MeshData mesh;

    for (int y = 0; y <= resolution; ++y) {
        for (int x = 0; x <= resolution; ++x) {
            float u = x / static_cast<float>(resolution);
            float v = y / static_cast<float>(resolution);

            float theta = u * 2.0f * glm::pi<float>(); // долгота
            float phi   = v * glm::pi<float>();        // широта

            glm::vec3 pos;
            pos.x = radius * sin(phi) * cos(theta) + position.x;
            pos.y = radius * cos(phi) + position.y;
            pos.z = radius * sin(phi) * sin(theta) + position.z;

            mesh.vertices.push_back(pos);
        }
    }

    for (int y = 0; y < resolution; y++) {
        for (int x = 0; x < resolution; x++) {
            int i0 = y * (resolution + 1) + x;
            int i1 = i0 + 1;
            int i2 = i0 + (resolution + 1);
            int i3 = i2 + 1;

            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);

            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }

    return mesh;
}
