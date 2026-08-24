#version 330 core
// The rack's last pass: nearest-neighbour upscale of the dithered working
// image (cv2 INTER_NEAREST convention: src = floor(x * src/dst), clamped —
// scale 1 when dithering ran at full res or is off), the CRT scan-line
// overlay ((y % pitch) < pitch/2 rows dimmed), the [0,1] clip, and the
// final u8 encode. The CPU path's (out*255).astype(uint8) TRUNCATES, so
// floor here, then the unorm8 store of an exact k/255 is lossless.
uniform sampler2D u_src;
uniform ivec2 u_src_size;
uniform vec2 u_scale;        // src_size / out_size
uniform int u_scan_p;        // scanline pitch; 0 = scanlines off
uniform int u_scan_half;     // p // 2
uniform float u_scan_dim;    // 1 - scanline_strength

layout(location = 0) out vec4 f_color;

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    ivec2 s = min(ivec2(floor(vec2(xy) * u_scale)), u_src_size - 1);
    vec3 v = texelFetch(u_src, s, 0).rgb;
    if (u_scan_p > 0 && (xy.y % u_scan_p) < u_scan_half) v *= u_scan_dim;
    v = clamp(v, 0.0, 1.0);
    f_color = vec4(floor(v * 255.0) / 255.0, 1.0);
}
