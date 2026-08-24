// Full-screen triangle. Feed three vertices (-1,-1) (3,-1) (-1,3); every
// fragment pass in the physarum engine is drawn with this.
in vec2 in_vert;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
