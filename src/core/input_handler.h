#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "camera.h"

class InputHandler {
   public:
    InputHandler() = delete;

    InputHandler(const InputHandler&) = delete;
    InputHandler& operator=(const InputHandler&) = delete;

    explicit InputHandler(GLFWwindow* win) : window(win) {
    }

    InputHandler(InputHandler&&) = delete;
    InputHandler& operator=(InputHandler&&) = delete;

    ~InputHandler() = default;

    void BindCamera(Camera* cam);
    void Process();
    void ProcessCamera(float deltaTime);

   private:
    GLFWwindow* window;
    Camera* camera = nullptr;
};
