// Physarum diffuse + decay — one axis of a separable box blur over the trail.
//
// Run twice per frame with fullscreen.vert:
//   H: u_src = trail,  u_add = laid deposits (u_use_add = 1), u_dir = (1,0),
//      u_scale = 1 / (2r+1)                      -> tmp
//   V: u_src = tmp,    u_use_add = 0,            u_dir = (0,1),
//      u_scale = decay / (2r+1)                  -> next trail
// Edges wrap (the agents already do). Only .r is read and written.

uniform sampler2D u_src;
uniform sampler2D u_add;
uniform int u_use_add;
uniform ivec2 u_dir;
uniform int u_radius;      // box half-width in px; 0 = no diffusion
uniform ivec2 u_grid;
uniform float u_scale;

layout(location = 0) out vec4 f_color;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    float acc = 0.0;
    for (int i = -u_radius; i <= u_radius; i++) {
        ivec2 q = c + u_dir * i;
        // wrap via floor, not integer %: GLSL ES 3.00 leaves % undefined
        // when an operand is negative, and q is negative at the low edge
        q -= u_grid * ivec2(floor(vec2(q) / vec2(u_grid)));
        acc += texelFetch(u_src, q, 0).r;
        if (u_use_add == 1) acc += texelFetch(u_add, q, 0).r;
    }
    f_color = vec4(acc * u_scale, 0.0, 0.0, 1.0);
}
