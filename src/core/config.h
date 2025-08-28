#pragma once

#include <glm/glm.hpp>

namespace WindowConfig {
inline constexpr unsigned int width = 800;
inline constexpr unsigned int height = 600;
inline const char* title = "OpenGL window";
}  // namespace WindowConfig

struct DomainConfig {
    glm::vec3 L{2.0f, 2.0f, 2.0f};  // размеры куба
    glm::ivec3 N{32, 32, 32};       // число ячеек
};

struct PhysicsConfig {
    float dtFixed{0.003f};
    glm::vec3 gravity{0.0f, -9.8f, 0.0f};
    float bounce{0.8f};
    float v_eps{1e-3f};
    bool clamp{false};  // режим: clamp или отражение
};

struct RenderConfig {
    glm::vec3 axisColors[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    bool drawFrame{true};
};

struct TimeConfig {
    float deltaTime{0.0f};
    float lastFrame{0.0f};
    bool fixedStep{true};
    float timeScale{1.0f};
};

struct AppConfig {
    DomainConfig domain;
    PhysicsConfig physics;
    RenderConfig render;
    TimeConfig time;
};
