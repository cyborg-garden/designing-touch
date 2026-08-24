#version 330 core
// Grid-res colorized frame -> output res (cv2 INTER_LINEAR convention),
// then the optional video background: screen blend
// out = 1 - (1-p)(1-bg*mix), with bg*mix rounded to u8 first
// (cv2.convertScaleAbs) — mirrors dtouch.modes.particles.composite_video_bg.
uniform sampler2D u_src;     // colorized grid RGBA8
uniform sampler2D u_cam;     // camera RGB (only when u_mix > 0)
uniform ivec2 u_src_size;
uniform ivec2 u_cam_size;
uniform vec2 u_src_scale;    // src_size / out_size
uniform vec2 u_cam_scale;    // cam_size / out_size
uniform float u_mix;         // 0 = no video bg

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
    vec2 dst = gl_FragCoord.xy - 0.5;
    vec3 p = bilin(u_src, u_src_size, dst, u_src_scale);
    // cv2.resize on uint8 rounds to u8; keep the value on the u8 lattice
    // so the screen blend sees what the CPU path sees
    p = floor(p * 255.0 + 0.5) / 255.0;
    if (u_mix > 0.0) {
        vec3 bg = bilin(u_cam, u_cam_size, dst, u_cam_scale) * u_mix;
        bg = floor(bg * 255.0 + 0.5) / 255.0;      // convertScaleAbs rounds
        p = 1.0 - (1.0 - p) * (1.0 - bg);
    }
    f_color = vec4(p, 1.0);
}
