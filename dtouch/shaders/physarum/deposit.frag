// Physarum deposit — writes the blended deposit weight; additively blended.
in float v_dep;
in vec3 v_mask;
layout(location = 0) out vec4 f_color;
void main() {
    f_color = vec4(v_mask * v_dep, 1.0);
}
