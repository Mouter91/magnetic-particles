#version 330 core
layout(location=0) in vec3 inPos;

uniform vec3  L;      // размеры куба (Lx,Ly,Lz)
uniform ivec3 N;      // число ячеек (Nx,Ny,Nz) >= 1
uniform int   axis;   // 0=X, 1=Y, 2=Z
uniform mat4 uView;
uniform mat4 uProj;

void main() {
    vec3 halfL = 0.5 * L;
    float x0 = -halfL.x;
    float y0 = -halfL.y;
    float z0 = -halfL.z;

    vec3 step = vec3(
        L.x / float(max(N.x, 1)),
        L.y / float(max(N.y, 1)),
        L.z / float(max(N.z, 1))
    );

    vec3 base;
    vec3 offset;
    int instanceID = gl_InstanceID;

    if (axis == 0) {
        base = vec3(inPos.x, 0.0, 0.0);

        int Ny1 = N.y + 1;
        int k = instanceID / Ny1;
        int j = instanceID % Ny1;

        float y = y0 + float(j) * step.y;
        float z = z0 + float(k) * step.z;
        offset = vec3(0.0, y, z);
    } else if (axis == 1) {
        base = vec3(0.0, inPos.x, 0.0);

        int Nx1 = N.x + 1;
        int k = instanceID / Nx1;
        int i = instanceID % Nx1;

        float x = x0 + float(i) * step.x;
        float z = z0 + float(k) * step.z;
        offset = vec3(x, 0.0, z);
    } else { // axis == 2
        base = vec3(0.0, 0.0, inPos.x);

        int Nx1 = N.x + 1;
        int j = instanceID / Nx1;
        int i = instanceID % Nx1;

        float x = x0 + float(i) * step.x;
        float y = y0 + float(j) * step.y;
        offset = vec3(x, y, 0.0);
    }

    vec3 world = base + offset;
    gl_Position = uProj * uView * vec4(world, 1.0);
}

