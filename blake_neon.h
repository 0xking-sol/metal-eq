// Equihash 200,9 round-0 BLAKE2b on the CPU (Apple Silicon): the final block over a fixed 128-byte
// midstate, where only message words m0 (header bytes 128..135) and m1 (header bytes 136..139 |
// index << 32) are nonzero. Output: the 7 state words the GPU's round 0 produces (bytes 0..49 used).
//   hash_scalar: one input. hash_hybrid3: two inputs in NEON lanes (XAR = xor + rotate in one
//   instruction) plus one on the scalar pipes, interleaved; 2.4x the scalar rate on one core.
#pragma once
#include <arm_neon.h>
#include <stdint.h>

static const uint64_t BN_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

// The 12 rounds, fully unrolled. Each row is one round's sigma permutation.
#define BN_ROUND(G, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15) \
    G(0,4,8,12, s0,s1);  G(1,5,9,13, s2,s3);   G(2,6,10,14, s4,s5); G(3,7,11,15, s6,s7); \
    G(0,5,10,15, s8,s9); G(1,6,11,12, s10,s11); G(2,7,8,13, s12,s13); G(3,4,9,14, s14,s15);
#define BN_ALL_ROUNDS(G) \
    BN_ROUND(G,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15) \
    BN_ROUND(G, 14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3) \
    BN_ROUND(G, 11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4) \
    BN_ROUND(G,  7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8) \
    BN_ROUND(G,  9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13) \
    BN_ROUND(G,  2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9) \
    BN_ROUND(G, 12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11) \
    BN_ROUND(G, 13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10) \
    BN_ROUND(G,  6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5) \
    BN_ROUND(G, 10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0) \
    BN_ROUND(G,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15) \
    BN_ROUND(G, 14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3)

typedef struct { uint64_t h[8]; uint64_t m0, m1lo; } BnMid;   // midstate + constant message bits

static inline uint64_t bn_ror64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

#define BN_MS(i) ((i) == 0 ? m0 : (i) == 1 ? m1 : 0)
#define BN_GS(a,b,c,d,x,y) do { \
    v[a] += v[b] + BN_MS(x); v[d] = bn_ror64(v[d] ^ v[a], 32); v[c] += v[d]; v[b] = bn_ror64(v[b] ^ v[c], 24); \
    v[a] += v[b] + BN_MS(y); v[d] = bn_ror64(v[d] ^ v[a], 16); v[c] += v[d]; v[b] = bn_ror64(v[b] ^ v[c], 63); } while (0)

#define BN_MV(i) ((i) == 0 ? M0 : (i) == 1 ? M1 : Z)
#define BN_GV(a,b,c,d,x,y) do { \
    w[a] = vaddq_u64(vaddq_u64(w[a], w[b]), BN_MV(x)); w[d] = vxarq_u64(w[d], w[a], 32); \
    w[c] = vaddq_u64(w[c], w[d]);                      w[b] = vxarq_u64(w[b], w[c], 24); \
    w[a] = vaddq_u64(vaddq_u64(w[a], w[b]), BN_MV(y)); w[d] = vxarq_u64(w[d], w[a], 16); \
    w[c] = vaddq_u64(w[c], w[d]);                      w[b] = vxarq_u64(w[b], w[c], 63); } while (0)

#define BN_GH(a,b,c,d,x,y) do { BN_GV(a,b,c,d,x,y); BN_GS(a,b,c,d,x,y); } while (0)

static inline void hash_scalar(const BnMid *md, uint32_t index, uint64_t out[7])
{
    const uint64_t m0 = md->m0, m1 = md->m1lo | ((uint64_t)index << 32);
    uint64_t v[16];
    for (int i = 0; i < 8; i++) v[i] = md->h[i];
    for (int i = 0; i < 8; i++) v[8 + i] = BN_IV[i];
    v[12] ^= 144;      // bytes hashed: 128 + 16
    v[14] = ~v[14];    // final block
    BN_ALL_ROUNDS(BN_GS)
    for (int i = 0; i < 7; i++) out[i] = md->h[i] ^ v[i] ^ v[i + 8];
}

// inputs i0, i1 in the NEON lanes, i2 on the scalar pipes
static inline void hash_hybrid3(const BnMid *md, uint32_t i0, uint32_t i1, uint32_t i2,
                                uint64_t out0[7], uint64_t out1[7], uint64_t out2[7])
{
    const uint64x2_t Z = vdupq_n_u64(0), M0 = vdupq_n_u64(md->m0);
    const uint64x2_t M1 = vcombine_u64(vcreate_u64(md->m1lo | ((uint64_t)i0 << 32)),
                                       vcreate_u64(md->m1lo | ((uint64_t)i1 << 32)));
    const uint64_t m0 = md->m0, m1 = md->m1lo | ((uint64_t)i2 << 32);
    uint64x2_t w[16];
    uint64_t v[16];
    for (int i = 0; i < 8; i++) { w[i] = vdupq_n_u64(md->h[i]); v[i] = md->h[i]; }
    for (int i = 0; i < 8; i++) { w[8 + i] = vdupq_n_u64(BN_IV[i]); v[8 + i] = BN_IV[i]; }
    w[12] = veorq_u64(w[12], vdupq_n_u64(144));
    w[14] = veorq_u64(w[14], vdupq_n_u64(~0ULL));
    v[12] ^= 144;
    v[14] = ~v[14];
    BN_ALL_ROUNDS(BN_GH)
    for (int i = 0; i < 7; i++)
      {
        uint64x2_t o = veorq_u64(vdupq_n_u64(md->h[i]), veorq_u64(w[i], w[i + 8]));
        out0[i] = vgetq_lane_u64(o, 0);
        out1[i] = vgetq_lane_u64(o, 1);
        out2[i] = md->h[i] ^ v[i] ^ v[i + 8];
      }
}
