#version 330 core
// Stages 1+2 of the rack in one pass: per-row horizontal scan drift
// (np.roll of each row by drift_px[y]) followed by the chroma shift
// (np.roll of channels 2 and 0 along x). Both only move pixels
// horizontally, so they compose into one wrapped fetch per channel:
// chan c at x reads (x - chroma_shift_c - drift[y]) mod w.
uniform sampler2D u_src;
uniform sampler2D u_drift;   // (1, h) R32F per-row shift; used when u_use_drift
uniform int u_use_drift;
uniform int u_shift_r;       // channel index 2
uniform int u_shift_b;       // channel index 0
uniform int u_w;

layout(location = 0) out vec4 f_color;

int wrapx(int x) {
    return ((x % u_w) + u_w) % u_w;
}

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    int d = (u_use_drift == 1)
        ? int(texelFetch(u_drift, ivec2(0, xy.y), 0).r) : 0;
    float c0 = texelFetch(u_src, ivec2(wrapx(xy.x - d - u_shift_b), xy.y), 0).r;
    float c1 = texelFetch(u_src, ivec2(wrapx(xy.x - d), xy.y), 0).g;
    float c2 = texelFetch(u_src, ivec2(wrapx(xy.x - d - u_shift_r), xy.y), 0).b;
    f_color = vec4(c0, c1, c2, 1.0);
}
