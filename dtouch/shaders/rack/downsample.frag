#version 330 core
// Full res -> the dither working res, cv2 INTER_LINEAR convention
// (pure bilinear at the mapped coordinate — matches cv2's float path
// exactly; verified maxdiff 0.0 against cv2.resize on float32).
uniform sampler2D u_src;
uniform ivec2 u_src_size;
uniform vec2 u_scale;        // src_size / dst_size

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
    f_color = vec4(bilin(u_src, u_src_size, gl_FragCoord.xy - 0.5, u_scale), 1.0);
}
