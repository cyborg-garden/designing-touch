#version 330 core
// Stage 4: hard bit-depth reduction. np.round is round-half-to-even —
// GLSL roundEven is the exact counterpart. Rendered into a float target:
// k/levels is generally NOT on the u8 lattice, and the dither stage's
// gamma path quantises with +0.5 semantics that an 8-bit store would break.
uniform sampler2D u_src;
uniform float u_levels;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 v = texelFetch(u_src, ivec2(gl_FragCoord.xy), 0).rgb;
    f_color = vec4(roundEven(v * u_levels) / u_levels, 1.0);
}
