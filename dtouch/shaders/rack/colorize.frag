#version 330 core
// Luminance -> RGB at grid res. Mirrors dtouch.modes.physarum._colorize
// (palette: lut[(lum*255).astype(u8)]) and the "video" palette
// (lum * (0.25 + 0.75*color), truncated to u8). The video color is the
// small camera frame bilinearly resized to the grid, cv2 INTER_LINEAR
// convention: src = (dst + 0.5) * (src_size / dst_size) - 0.5.
uniform sampler2D u_lum;     // R8, grid res
uniform sampler2D u_lut;     // 256x1 palette (RGB, only when u_mode == 0)
uniform sampler2D u_video;   // small camera RGB (only when u_mode == 1)
uniform int u_mode;          // 0 = palette LUT, 1 = video-lit
uniform ivec2 u_vsize;       // video texture size
uniform vec2 u_vscale;       // u_vsize / grid_size

layout(location = 0) out vec4 f_color;

vec3 bilin(sampler2D t, ivec2 sz, vec2 dst, vec2 scale) {
    vec2 s = (dst + 0.5) * scale - 0.5;
    vec2 s0 = floor(s);
    vec2 f = s - s0;
    ivec2 i0 = ivec2(s0);
    ivec2 ia = clamp(i0, ivec2(0), sz - 1);
    ivec2 ib = clamp(i0 + 1, ivec2(0), sz - 1);
    vec3 c00 = texelFetch(t, ivec2(ia.x, ia.y), 0).rgb;
    vec3 c10 = texelFetch(t, ivec2(ib.x, ia.y), 0).rgb;
    vec3 c01 = texelFetch(t, ivec2(ia.x, ib.y), 0).rgb;
    vec3 c11 = texelFetch(t, ivec2(ib.x, ib.y), 0).rgb;
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    float lum = texelFetch(u_lum, xy, 0).r;
    if (u_mode == 0) {
        // the u8 round-trip is exact on the CPU (tested); +0.5 keeps it
        // exact here too against the unorm8 -> float conversion
        int idx = int(floor(lum * 255.0 + 0.5));
        f_color = vec4(texelFetch(u_lut, ivec2(idx, 0), 0).rgb, 1.0);
    } else {
        vec3 c = bilin(u_video, u_vsize, gl_FragCoord.xy - 0.5, u_vscale);
        vec3 v = lum * (0.25 + 0.75 * c);
        // the CPU path truncates: (v * 255).astype(uint8)
        f_color = vec4(floor(v * 255.0) / 255.0, 1.0);
    }
}
