// Physarum deposit — writes the blended deposit weight; additively blended.
in float v_dep;
layout(location = 0) out vec4 f_color;
void main() {
    f_color = vec4(v_dep, 0.0, 0.0, 1.0);
}
