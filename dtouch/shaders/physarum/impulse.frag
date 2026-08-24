// Physarum impulses — burst, wave and gather, applied over the agent texture.
//
//   u_mode 1 (burst):  each agent, with probability u_frac, teleports to a
//     gaussian of radius u_radius around u_center with a fresh heading.
//   u_mode 2 (wave):   every heading points away from u_center.
//   u_mode 3 (gather): only agents within u_radius of u_center (torus
//     distance) teleport, with probability u_frac, into a tight gaussian
//     (0.15 * u_radius) at the center — a LOCAL rush that never drains the
//     rest of the organism (the react/"spell" impulse).
// Drawn with fullscreen.vert; writes the next agent texture (ping-pong).

uniform sampler2D u_agents;
uniform ivec2 u_grid;
uniform int u_aw;
uniform int u_mode;
uniform vec2 u_center;
uniform float u_frac;
uniform float u_radius;
uniform uint u_salt;

layout(location = 0) out vec4 f_agent;

uint hash(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}
float rnd(inout uint s) { s = hash(s); return float(s) * (1.0 / 4294967296.0); }

void main() {
    ivec2 ac = ivec2(gl_FragCoord.xy);
    uint idx = uint(ac.y * u_aw + ac.x);
    vec4 a = texelFetch(u_agents, ac, 0);
    if (u_mode == 1) {
        uint s = hash(idx * 0x9e3779b9u + u_salt);
        if (rnd(s) < u_frac) {
            float u1 = max(rnd(s), 1e-7);
            float u2 = rnd(s) * 6.2831853;
            float r = sqrt(-2.0 * log(u1)) * u_radius;      // Box-Muller
            a.xy = mod(u_center + vec2(cos(u2), sin(u2)) * r, vec2(u_grid));
            a.z = rnd(s) * 6.2831853;
        }
    } else if (u_mode == 2) {
        a.z = atan(a.y - u_center.y, a.x - u_center.x);
    } else if (u_mode == 3) {
        vec2 g = vec2(u_grid);
        vec2 d = a.xy - u_center;
        d -= g * floor(d / g + 0.5);          // shortest torus offset
        if (dot(d, d) < u_radius * u_radius) {
            uint s = hash(idx * 0x9e3779b9u + u_salt);
            if (rnd(s) < u_frac) {
                float u1 = max(rnd(s), 1e-7);
                float u2 = rnd(s) * 6.2831853;
                float r = sqrt(-2.0 * log(u1)) * (0.15 * u_radius);
                a.xy = mod(u_center + vec2(cos(u2), sin(u2)) * r, g);
                a.z = rnd(s) * 6.2831853;
            }
        }
    }
    f_agent = a;
}
