// Physarum agent update — one Jones step per texel of the agent texture.
//
// Agent texture (RGBA32F): x, y, heading, spare. Drawn over the agent
// texture with fullscreen.vert; writes the next agent texture (ping-pong).
// Per agent: read the matte under it (the pen) and blend the field/body
// behavior points, sense trail+food at three points ahead, turn toward the
// strongest, step, wrap, and — with probability u_reseed — respawn onto the
// lit subject (matte * luma) by rejection sampling.
//
// Only the .r channel of u_trail / u_matte / u_gray is read.

uniform sampler2D u_agents;
uniform sampler2D u_trail;
uniform sampler2D u_matte;
uniform sampler2D u_gray;
uniform ivec2 u_grid;      // trail size (gw, gh)
uniform int u_aw;          // agent texture width
uniform vec2 u_sense;      // (field point, body point) — sensor distance, px
uniform vec2 u_spread;     // sensor half-angle, rad
uniform vec2 u_turn;       // rotation when a side sensor wins, rad
uniform vec2 u_step;       // stride per frame, px
uniform float u_gain;      // tempo: scales sense + step
uniform float u_food;      // how strongly luma is added to what sensors read
uniform float u_reseed;    // per-agent respawn probability this frame
uniform float u_wmax;      // upper bound of matte*clamp(gray,.05,1); <= 0: respawn uniformly
uniform uint u_salt;       // per-frame random salt
uniform float u_satcap;    // sensed-trail soft cap (absolute units); <= 0 = off
uniform float u_jitter;    // per-step heading wobble, rad; 0 = off
uniform float u_hetero;    // 0..1 blend toward the 3-sub-population sense split

layout(location = 0) out vec4 f_agent;

uint hash(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}
float rnd(inout uint s) { s = hash(s); return float(s) * (1.0 / 4294967296.0); }

ivec2 cell(vec2 p) { return ivec2(mod(floor(p), vec2(u_grid))); }

float food(vec2 p) {
    ivec2 c = cell(p);
    float t = texelFetch(u_trail, c, 0).r;
    // sensor saturation: softly cap the sensed trail so a fat vein reads the
    // same as a merely strong thin one (anti-thoroughfare; see dtouch.physarum)
    if (u_satcap > 0.0) t = u_satcap * (1.0 - exp(-t / u_satcap));
    return t + u_food * texelFetch(u_gray, c, 0).r;
}

void main() {
    ivec2 ac = ivec2(gl_FragCoord.xy);
    uint idx = uint(ac.y * u_aw + ac.x);
    vec4 a = texelFetch(u_agents, ac, 0);
    vec2 p = a.xy;
    float h = a.z;
    uint s = hash(idx * 0x9e3779b9u + u_salt);

    // the pen: blend field -> body by the matte under the agent
    float t = texelFetch(u_matte, cell(p), 0).r;
    float sense = mix(u_sense.x, u_sense.y, t) * u_gain;
    if (u_hetero > 0.0) {
        // 3 sub-populations by agent index (float mod — ES 3.00 has no
        // integer % for this): short / mid / long sense ranges, blended in
        // by u_hetero. Multi-scale sensing grows multi-scale structure.
        float g3 = mod(float(idx), 3.0);
        float m = (g3 < 0.5) ? 0.45 : (g3 < 1.5) ? 1.0 : 1.9;
        sense *= 1.0 + (m - 1.0) * u_hetero;
    }
    float spread = mix(u_spread.x, u_spread.y, t);
    float turn = mix(u_turn.x, u_turn.y, t);
    float stp = mix(u_step.x, u_step.y, t) * u_gain;

    // Jones steering: hold when ahead wins; coin flip when ahead loses to
    // both sides; otherwise turn toward the stronger side.
    float fc = food(p + vec2(cos(h), sin(h)) * sense);
    float fl = food(p + vec2(cos(h - spread), sin(h - spread)) * sense);
    float fr = food(p + vec2(cos(h + spread), sin(h + spread)) * sense);
    float dir;
    if (fc > fl && fc > fr) dir = 0.0;
    else if (fc < fl && fc < fr) dir = (rnd(s) < 0.5) ? -1.0 : 1.0;
    else dir = (fl > fr) ? -1.0 : 1.0;
    h += dir * turn;
    // per-step heading wobble: highways stop being perfectly straight
    // attractors and the mold keeps probing sideways
    if (u_jitter > 0.0) h += (rnd(s) * 2.0 - 1.0) * u_jitter;

    p += vec2(cos(h), sin(h)) * stp;
    p = mod(p, vec2(u_grid));

    // recycle a trickle of agents onto the lit subject (matte * luma)
    if (u_reseed > 0.0 && rnd(s) < u_reseed) {
        vec2 g = vec2(u_grid);
        if (u_wmax <= 0.0) {
            p = vec2(rnd(s), rnd(s)) * g;
            h = rnd(s) * 6.2831853;
        } else {
            for (int i = 0; i < 16; i++) {
                vec2 c = vec2(rnd(s), rnd(s)) * g;
                ivec2 ci = cell(c);
                float w = texelFetch(u_matte, ci, 0).r
                        * clamp(texelFetch(u_gray, ci, 0).r, 0.05, 1.0);
                if (rnd(s) * u_wmax < w) {
                    p = c;
                    h = rnd(s) * 6.2831853;
                    break;
                }
            }
        }
    }
    f_agent = vec4(p, h, a.w);
}
