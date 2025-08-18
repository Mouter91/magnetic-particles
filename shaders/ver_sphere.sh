#version 330 core
layout(location = 0) in vec3 pos; 
layout(location = 1) in vec4 center;

uniform mat4 uView;
uniform mat4 uProj;

void main() {
    vec3 worldPos = center.xyz + pos * center.w;
    gl_Position = uProj * uView * vec4(worldPos, 1.0);
}
