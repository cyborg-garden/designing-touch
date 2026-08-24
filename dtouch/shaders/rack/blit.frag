#version 330 core
// Offset texel copy inside a confined viewport — the glitch tile's capture
// (frame region -> tile texture) and replay (tile -> frame region).
uniform sampler2D u_src;
uniform ivec2 u_src_off;     // origin of the region to read in u_src
uniform ivec2 u_dst_off;     // viewport origin (gl_FragCoord is absolute)

layout(location = 0) out vec4 f_color;

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy) - u_dst_off + u_src_off;
    f_color = vec4(texelFetch(u_src, xy, 0).rgb, 1.0);
}
