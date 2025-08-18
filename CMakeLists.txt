cmake_minimum_required(VERSION 3.20)
project(Sphere LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Найдем CUDA
find_package(CUDA REQUIRED)

# Добавляем пути для заголовочных файлов CUDA
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

set(CUDA_TOOLKIT_ROOT_DIR "C:/PROGRA~1/NVIDIA~2/CUDA/v12.9")

include_directories(
    ${PROJECT_SOURCE_DIR}/include    
    ${CMAKE_SOURCE_DIR}/src 
)

# Добавляем glad
add_library(glad STATIC src/app/glad.c)

# Добавляем GLFW и GLM
add_subdirectory(external/GLFW)
add_subdirectory(external/GLM)

# Список исходников
set(SOURCES
    src/app/main.cpp
    src/shapes/shapes.cpp
    src/core/shader.cpp
    src/render/render.cpp
    src/physics/physics.cu  # Добавляем .cu файл для компиляции через nvcc
)

# Указываем, что этот файл компилируется с помощью CUDA
set_source_files_properties(src/physics/physics.cu PROPERTIES LANGUAGE CUDA)

# Целевой исполняемый файл
add_executable(main ${SOURCES})

# Добавляем include директории для GLFW и GLM
target_include_directories(main PRIVATE external/GLM external/GLFW)

# Линковка
target_link_libraries(main PRIVATE glad glfw glm::glm opengl32)

# Параметры компиляции для CUDA
target_compile_options(main PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}>"
)


