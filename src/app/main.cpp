#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "core/config.h"
#include "core/input_handler.h"
#include "core/shader.h"
#include "render/grid/grid.hpp"
#include "object/magneticparticle.hpp"

#include <iostream>

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window =
        glfwCreateWindow(WindowConfig::width, WindowConfig::height, "Sphere", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to load OpenGL via glad" << std::endl;
        return -1;
    }

    glfwSetFramebufferSizeCallback(
        window, [](GLFWwindow *, int width, int height) { glViewport(0, 0, width, height); });

    AppConfig config;
    config.domain.L = {8, 8, 8};
    config.domain.N = {20, 20, 20};
    config.time.lastFrame = glfwGetTime();
    float accumulator = 0.0f;

    InputHandler key_bind(window);

    Camera camera;
    key_bind.BindCamera(&camera);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetWindowUserPointer(window, &camera);

    glfwSetCursorPosCallback(window, [](GLFWwindow *win, double xpos, double ypos) {
        static bool firstMouse = true;
        static float lastX = static_cast<float>(WindowConfig::width) / 2,
                     lastY = static_cast<float>(WindowConfig::height) / 2;

        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;

        lastX = xpos;
        lastY = ypos;

        auto *cam = static_cast<Camera *>(glfwGetWindowUserPointer(win));
        if (cam) {
            cam->ProcessMouseMovement(xoffset, yoffset);
        }
    });

    glfwSetScrollCallback(window, [](GLFWwindow *win, double xoffset, double yoffset) {
        auto *cam = static_cast<Camera *>(glfwGetWindowUserPointer(win));
        if (cam) {
            cam->ProcessMouseScroll(yoffset);
        }
    });

    Shader shader("../shaders/ver_sphere.sh", "../shaders/frg_sphere.sh");
    MagneticParticles particles(&config);
    particles.Initialize(50000);

    Grid grid(config.domain.L, config.domain.N);
    grid.Create();

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        config.time.deltaTime = currentFrame - config.time.lastFrame;
        config.time.lastFrame = currentFrame;

        key_bind.Process();
        key_bind.ProcessCamera(config.time.deltaTime);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();

        const glm::mat4 &view = camera.GetViewMatrix();
        glm::mat4 projection = glm::perspective(
            glm::radians(camera.GetZoom()),
            (float)WindowConfig::width / (float)WindowConfig::height, 0.1f, 100.0f);

        shader.setMat4("uProj", projection);
        shader.setMat4("uView", view);

        grid.Render(view, projection);

        shader.use();

        if (config.time.fixedStep == true) {
            accumulator += config.time.deltaTime * config.time.timeScale;
            while (accumulator >= config.physics.dtFixed) {
                particles.PhysicsParticles(config.physics.dtFixed);
                accumulator -= config.physics.dtFixed;
            }
        } else {
            particles.PhysicsParticles(config.time.deltaTime * config.time.timeScale);
        }

        particles.Render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
