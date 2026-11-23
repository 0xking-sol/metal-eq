// metal-eq v2: a bucketed Equihash (200,9) solver for Apple GPUs (Metal Shading Language).
//
// Copyright (c) 2026, the metal-eq authors. The MIT License (MIT): permission is hereby granted,
// free of charge, to any person obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without restriction, including without
// limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, subject to the following condition: this notice shall be included in
// all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
// WARRANTY OF ANY KIND. (The BLAKE2b round code follows SILENTARMY, (c) 2016 Marc Bevand, MIT.)
//
// The idea (Wagner's algorithm with buckets, as in Tromp's CPU solver): every round keeps its
// entries in NBUCKETS buckets, keyed by the top 12 bits of the next 20-bit digit. One threadgroup
// processes one bucket at a time in on-chip memory: it reads each entry once, groups the entries by
// the remaining 8 bits of the digit, XORs every colliding pair, and appends the results to the
// buckets of the next round. So reads are sequential and writes are appends, instead of v1's
// scattered writes into a 384 MB table.
//
// Entry bit strings are stored MSB-first in 32-bit words. After round r (r = 0..8) an entry holds
//   rest_r (the low 8 bits of digit r) || digit r+1 || ... || digit 9      (8 + 20*(9-r) bits)
// and its "tree": the input index (round 0), or (parent bucket << 20 | slot a << 10 | slot b).

#include <metal_stdlib>
using namespace metal;

#define NBUCKETS        4096        // 2^12: the top 12 bits of a digit
#define NRESTS          256         // 2^8: the low 8 bits of a digit
#ifndef NSLOTS
#define NSLOTS          640         // slots per bucket; mean ~512 (degenerate floods can overflow: no yield effect seen)
#endif
#ifndef TILE
#define TILE            1           // record layout: slot-major tiles of TILE buckets (1 = bucket-major, v4.3)
#endif
#ifndef BPT
#define BPT             1           // buckets per threadgroup in v2_round/v2_final, one after another (1 = v4.3)
#endif
#ifndef LIST_CAP
#define LIST_CAP        12          // entries per (bucket, rest) list; the mean is 2
#endif
#ifndef MAX_PAIRS
#define MAX_PAIRS       512         // pairs buffered per bucket; overflow is processed inline
#endif
#ifndef TG
#define TG              256         // threads per threadgroup in rounds 1..8 and final (one bucket each)
#endif
#ifndef DIAG
#define DIAG            0           // profiling only (wrong results): 2 rounds no global writes, 3 rounds no atomics, 4 rounds no tree writes/reads
#endif
#define DATA_WORDS      6           // the stride of an entry: round 0 needs 188 bits
#ifndef MAX_CANDS
#define MAX_CANDS       4096
#endif

#define WORDS(bits)     (((bits) + 31) / 32)
// Index (in records) of slot s of bucket b. TILE 1 keeps each bucket's slots together (reads are
// sequential, but the appends of a round scatter over the whole buffer); larger tiles interleave
// TILE buckets slot by slot, so appends that run at the same fill level land close together.
#define SIDX(b, s)      ((((b) / TILE) * NSLOTS + (s)) * TILE + ((b) % TILE))
static_assert((TILE & (TILE - 1)) == 0 && NBUCKETS % TILE == 0, "TILE: a power of 2 dividing NBUCKETS");
static_assert((BPT & (BPT - 1)) == 0 && NBUCKETS % BPT == 0, "BPT: a power of 2 dividing NBUCKETS");
// v4: every entry is stored as its data words followed by its tree word, in one record, and every
// round has its own buffer (the host needs all the trees to expand solutions).
#define E_WORDS(r)      (WORDS(OUT_BITS(r)) + 1)

static_assert(NSLOTS <= 1024 && NSLOTS <= 65535, "slot ids are packed into 10 bits (trees) and ushort lists");
#define IN_BITS(r)      (8 + 20 * (10 - (r)))   // bits of an entry that round r reads
#define OUT_BITS(r)     (8 + 20 * (9 - (r)))    // bits of an entry that round r writes
// v4.2 split records: when the last data word of a round's entry holds <= 16 bits (rounds 1, 4, 6, 7),
// those bits go to a side array of 1 or 2 bytes per entry instead of a mostly-empty word.
// MAIN_W = whole data words in the main record; REC_W = main record words incl. the tree word.
constexpr uint MAIN_W(uint r) { return (OUT_BITS(r) % 32 == 0 || OUT_BITS(r) % 32 > 16) ? WORDS(OUT_BITS(r)) : OUT_BITS(r) / 32; }
constexpr uint SIDE_B(uint r) { return (OUT_BITS(r) % 32 == 0 || OUT_BITS(r) % 32 > 16) ? 0 : (OUT_BITS(r) % 32 <= 8 ? 1 : 2); }
constexpr uint REC_W(uint r)  { return MAIN_W(r) + 1; }

typedef struct
{
    uint    n;
    uint    cand[MAX_CANDS];        // final-round pairs: bucket << 20 | slot a << 10 | slot b
} cands_t;

constant ulong blake_iv[8] =
{
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

static inline ulong rotr64(ulong x, uint n)
{
    return (x >> n) | (x << (64 - n));
}

// 64-bit rotations built from 32-bit halves: Apple GPUs have 32-bit ALUs, and a 32-bit rotate plus a
// bitfield insert per half is cheaper than emulated 64-bit shifts.
#ifndef FAST_ROT
#define FAST_ROT 0       // measured slower (1.05 vs 0.96 ms): the compiler does 64-bit rotates better
#endif
static inline ulong rotr32_64(ulong x)      // rotate right by 32: swap the halves
{
    uint2 h = as_type<uint2>(x);
    return as_type<ulong>(uint2(h.y, h.x));
}
template <uint N>                           // rotate right by N, 0 < N < 32
static inline ulong rotr_lt32(ulong x)
{
    uint2 h = as_type<uint2>(x);            // h.x = low half, h.y = high half
    uint rl = rotate(h.x, 32u - N), rh = rotate(h.y, 32u - N);   // rl low bits = lo >> N
    // new low = (lo >> N) | (hi << (32 - N)): the top N bits are the low N bits of hi
    // new high = (hi >> N) | (lo << (32 - N)): the top N bits are the low N bits of lo
    return as_type<ulong>(uint2(insert_bits(rl, h.y, 32u - N, N), insert_bits(rh, h.x, 32u - N, N)));
}
static inline ulong rotr63_64(ulong x)      // rotate right by 63 = rotate left by 1
{
    uint2 h = as_type<uint2>(x);
    uint rl = rotate(h.x, 1u), rh = rotate(h.y, 1u);
    // new low = (lo << 1) | (hi >> 31); new high = (hi << 1) | (lo >> 31)
    return as_type<ulong>(uint2(insert_bits(rl, rh, 0u, 1u), insert_bits(rh, rl, 0u, 1u)));
}

#if FAST_ROT
#define mix(va, vb, vc, vd, x, y) \
    va = (va + vb + x); \
    vd = rotr32_64(vd ^ va); \
    vc = (vc + vd); \
    vb = rotr_lt32<24>(vb ^ vc); \
    va = (va + vb + y); \
    vd = rotr_lt32<16>(vd ^ va); \
    vc = (vc + vd); \
    vb = rotr63_64(vb ^ vc);
#else
#define mix(va, vb, vc, vd, x, y) \
    va = (va + vb + x); \
    vd = rotr64(vd ^ va, 32); \
    vc = (vc + vd); \
    vb = rotr64(vb ^ vc, 24); \
    va = (va + vb + y); \
    vd = rotr64(vd ^ va, 16); \
    vc = (vc + vd); \
    vb = rotr64(vb ^ vc, 63);
#endif

// byte n (0..55) of the BLAKE2b output words h[], which are little-endian
#define HBYTE(h, n)     ((uint)((h)[(n) / 8] >> (8 * ((n) % 8))) & 0xff)
// big-endian 32-bit word made of output bytes n..n+3
#define BEWORD(h, n)    ((HBYTE(h, n) << 24) | (HBYTE(h, (n) + 1) << 16) | \
                         (HBYTE(h, (n) + 2) << 8) | HBYTE(h, (n) + 3))

/*
** Append one round-0 entry: the 200-bit hash in w[0..6] (MSB-first, zero padded).
*/
static inline void store0(thread const uint *w, uint index, device uint *data,
        device atomic_uint *nslots)
{
    uint bucket = w[0] >> 20;
#if DIAG == 5
    if ((w[1] ^ w[3] ^ w[5]) == 0x12345678u)   // BLAKE2b only (profiling): keep w alive
        data[index & 0xffff] = w[2];
    return;
#endif
#if DIAG == 1
    uint slot = (index >> 13) % NSLOTS;     // fake slot: measures round 0 without atomics
#else
    uint slot = atomic_fetch_add_explicit(&nslots[bucket], 1u, memory_order_relaxed);
    if (slot >= NSLOTS)
        return;
#endif
#if DIAG == 6
    if ((w[1] ^ w[3]) == 0x12345678u)          // BLAKE2b + atomics only (profiling)
        data[slot] = w[2];
    return;
#endif
    device uint *d = data + SIDX(bucket, slot) * E_WORDS(0);
    // keep bits [12, 200): rest_0 || digits 1..9
    for (uint k = 0; k < WORDS(OUT_BITS(0)); k++)
        d[k] = (w[k] << 12) | (w[k + 1] >> 20);
    d[WORDS(OUT_BITS(0))] = index;
}

/*
** Round 0: one BLAKE2b per input index, two 200-bit hashes per BLAKE2b.
** blake_state is the state after the first 128-byte block; the second block is the last 12
** (zero) nonce bytes plus the 4-byte input index, so only message word 1 is non-zero.
*/
static inline void store_hashes(thread const ulong *h, uint input, device uint *data,
        device atomic_uint *nslots);

static inline void round0_one(constant ulong *blake_state, uint input, device uint *data,
        device atomic_uint *nslots)
{
    ulong v[16];
    ulong word1 = (ulong)input << 32;
    for (uint k = 0; k < 8; k++)
      {
        v[k] = blake_state[k];
        v[8 + k] = blake_iv[k];
      }
    v[12] ^= 140 + 4;       // bytes hashed: the 140-byte header + the 4-byte index
    v[14] ^= (ulong)-1;     // last block
    // the message schedule of BLAKE2b, with only word 1 non-zero
    mix(v[0], v[4], v[8],  v[12], 0, word1);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], word1, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, word1);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], 0, word1);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, word1);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], word1, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], word1, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, word1);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], word1, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], word1, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, word1);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], 0, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);
    mix(v[0], v[4], v[8],  v[12], 0, 0);
    mix(v[1], v[5], v[9],  v[13], 0, 0);
    mix(v[2], v[6], v[10], v[14], 0, 0);
    mix(v[3], v[7], v[11], v[15], 0, 0);
    mix(v[0], v[5], v[10], v[15], word1, 0);
    mix(v[1], v[6], v[11], v[12], 0, 0);
    mix(v[2], v[7], v[8],  v[13], 0, 0);
    mix(v[3], v[4], v[9],  v[14], 0, 0);

    ulong h[7];
    for (uint k = 0; k < 7; k++)
        h[k] = blake_state[k] ^ v[k] ^ v[k + 8];
    store_hashes(h, input, data, nslots);
}

/*
** Append the two entries of one BLAKE2b output h (input index `input`).
*/
static inline void store_hashes(thread const ulong *h, uint input, device uint *data,
        device atomic_uint *nslots)
{
    // hash A = output bytes 0..24, hash B = bytes 25..49 (big-endian bit strings)
    uint w[8];
    w[0] = BEWORD(h, 0);
    w[1] = BEWORD(h, 4);
    w[2] = BEWORD(h, 8);
    w[3] = BEWORD(h, 12);
    w[4] = BEWORD(h, 16);
    w[5] = BEWORD(h, 20);
    w[6] = HBYTE(h, 24) << 24;
    w[7] = 0;
    store0(w, input * 2, data, nslots);
    w[0] = BEWORD(h, 25);
    w[1] = BEWORD(h, 29);
    w[2] = BEWORD(h, 33);
    w[3] = BEWORD(h, 37);
    w[4] = BEWORD(h, 41);
    w[5] = BEWORD(h, 45);
    w[6] = HBYTE(h, 49) << 24;
    store0(w, input * 2 + 1, data, nslots);
}

/*
** --cpu-sg N: in each 256-thread round-0 threadgroup, the first N simdgroups store hashes that the
** CPU computed (cpu_hashes: 7 words per input, for inputs t*256 + [0, 32N) of each threadgroup t)
** and the others compute theirs. Loading is memory work and hashing is arithmetic, so every core
** does both at once; the kernel's register use is that of the hashing path, as before.
*/
#ifndef CPU_SG
#define CPU_SG 0
#endif

kernel void v2_round0(constant ulong *blake_state [[buffer(0)]],
        device uint *data [[buffer(1)]],
        device atomic_uint *nslots [[buffer(3)]],
        constant uint &input_base [[buffer(4)]],
        device const ulong *cpu_hashes [[buffer(5)]],
        uint gid [[thread_position_in_grid]],
        uint tg [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]])
{
#if CPU_SG
    if (tid < CPU_SG * 32)          // whole simdgroups: no divergence
      {
        device const ulong *src = cpu_hashes + (tg * CPU_SG * 32 + tid) * 7;
        ulong h[7];
        for (uint k = 0; k < 7; k++)
            h[k] = src[k];
        store_hashes(h, input_base + gid, data, nslots);
        return;
      }
#endif
    round0_one(blake_state, input_base + gid, data, nslots);   // round 0 can run in chunks
}

/*
** Rounds 1..8: one threadgroup per bucket of the previous round.
*/
// Two entries whose trees share a child subtree can only combine into a tree with duplicate
// leaves, which is never a valid solution. Skipping them early saves work in every later round.
static inline bool shares_child(uint ta, uint tb)
{
    if ((ta >> 20) != (tb >> 20))
        return false;   // different parent buckets: no shared child
    uint a0 = (ta >> 10) & 0x3ff, a1 = ta & 0x3ff, b0 = (tb >> 10) & 0x3ff, b1 = tb & 0x3ff;
    return a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1;
}

template <uint R>
static void xor_and_store(threadgroup uint (*cache)[WORDS(IN_BITS(R))], threadgroup uint *tcache,
        uint bucket_in, uint sa, uint sb,
        device uint *data_out, device uchar *side_out, device atomic_uint *nslots_out)
{
    constexpr uint IN_W = WORDS(IN_BITS(R));
    constexpr uint OUT_W = WORDS(OUT_BITS(R));
    if (R >= 2 && shares_child(tcache[sa], tcache[sb]))
        return;
    uint x[IN_W + 1];
    for (uint k = 0; k < IN_W; k++)
        x[k] = cache[sa][k] ^ cache[sb][k];
    x[IN_W] = 0;
    // x = 0^8 || digit R || ... ; drop the 8 zero bits and the 12 bits that pick the bucket
    uint o[OUT_W];
    uint nonzero = 0;
    for (uint k = 0; k < OUT_W; k++)
      {
        o[k] = (x[k] << 20) | (x[k + 1] >> 12);
        nonzero |= o[k];
      }
    if (!nonzero)
        return;     // duplicate inputs XOR to zero: never part of a valid solution
    uint bucket = (x[0] >> 12) & (NBUCKETS - 1);
#if DIAG == 3
    uint slot = (sa * 7 + sb) % NSLOTS;
    if (bucket_in == 0xffffffff)
        atomic_fetch_add_explicit(&nslots_out[bucket], 1u, memory_order_relaxed);
#else
    uint slot = atomic_fetch_add_explicit(&nslots_out[bucket], 1u, memory_order_relaxed);
    if (slot >= NSLOTS)
        return;
#endif
#if DIAG == 2
    if (o[0] != 0x12345678u)
        return;
#endif
    device uint *d = data_out + SIDX(bucket, slot) * REC_W(R);   // main data words + tree word
    for (uint k = 0; k < MAIN_W(R); k++)
        d[k] = o[k];
    d[MAIN_W(R)] = (bucket_in << 20) | (sa << 10) | sb;
    if (SIDE_B(R) == 1)
        side_out[SIDX(bucket, slot)] = (uchar)(o[MAIN_W(R)] >> 24);
    else if (SIDE_B(R) == 2)
        ((device ushort *)side_out)[SIDX(bucket, slot)] = (ushort)(o[MAIN_W(R)] >> 16);
}

template <uint R>
kernel void v2_round(device const uint *data_in [[buffer(0)]],
        device uint *data_out [[buffer(1)]],
        device const uint *nslots_in [[buffer(2)]],
        device atomic_uint *nslots_out [[buffer(3)]],
        device const uchar *side_in [[buffer(4)]],
        device uchar *side_out [[buffer(5)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]])
{
    constexpr uint IN_W = WORDS(IN_BITS(R));
    constexpr uint IN_MW = MAIN_W(R - 1), IN_SB = SIDE_B(R - 1), E = REC_W(R - 1);
    threadgroup uint cache[NSLOTS][IN_W];
    threadgroup uint tcache[(R >= 2) ? NSLOTS : 1];
    threadgroup ushort lists[NRESTS][LIST_CAP];
    threadgroup atomic_uint list_len[NRESTS];
    threadgroup uint pairs[MAX_PAIRS];
    threadgroup atomic_uint npairs;

    for (uint bi = 0; bi < BPT; bi++)
      {
        uint b = tgid * BPT + bi;
        if (bi)     // every thread is done with the previous bucket's lists, pairs and cache
            threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = tid; r < NRESTS; r += TG)
            atomic_store_explicit(&list_len[r], 0u, memory_order_relaxed);
        if (!tid)
            atomic_store_explicit(&npairs, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // read each entry of the bucket once, and list it under its 8 rest bits
        uint n = min(nslots_in[b], (uint)NSLOTS);
        for (uint s = tid; s < n; s += TG)
          {
            device const uint *e = data_in + SIDX(b, s) * E;
            for (uint k = 0; k < IN_MW; k++)
                cache[s][k] = e[k];
            if (IN_SB == 1)
                cache[s][IN_MW] = (uint)side_in[SIDX(b, s)] << 24;
            else if (IN_SB == 2)
                cache[s][IN_MW] = (uint)((device const ushort *)side_in)[SIDX(b, s)] << 16;
            if (R >= 2)
                tcache[s] = e[IN_MW];
            uint rest = cache[s][0] >> 24;
            uint pos = atomic_fetch_add_explicit(&list_len[rest], 1u, memory_order_relaxed);
            if (pos < LIST_CAP)
                lists[rest][pos] = (ushort)s;
          }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // every pair in a list collides on the whole 20-bit digit
        for (uint r = tid; r < NRESTS; r += TG)
          {
            uint len = min(atomic_load_explicit(&list_len[r], memory_order_relaxed), (uint)LIST_CAP);
            for (uint i = 0; i + 1 < len; i++)
                for (uint j = i + 1; j < len; j++)
                  {
                    uint sa = lists[r][i], sb = lists[r][j];
                    uint idx = atomic_fetch_add_explicit(&npairs, 1u, memory_order_relaxed);
                    if (idx < MAX_PAIRS)
                        pairs[idx] = (sa << 16) | sb;
                    else    // buffer full: do it here rather than lose the pair
                        xor_and_store<R>(cache, tcache, b, sa, sb, data_out, side_out, nslots_out);
                  }
          }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint np = min(atomic_load_explicit(&npairs, memory_order_relaxed), (uint)MAX_PAIRS);
        for (uint p = tid; p < np; p += TG)
            xor_and_store<R>(cache, tcache, b, pairs[p] >> 16, pairs[p] & 0xffff, data_out, side_out, nslots_out);
      }
}

template [[host_name("v2_round1")]] kernel void v2_round<1>(device const uint *, device uint *,
        device const uint *, device atomic_uint *, device const uchar *, device uchar *, uint, uint);
template [[host_name("v2_round2")]] kernel void v2_round<2>(device const uint *, device uint *,
        device const uint *, device atomic_uint *, device const uchar *, device uchar *, uint, uint);
template [[host_name("v2_round3")]] kernel void v2_round<3>(device const uint *, device uint *,
        device const uint *, device atomic_uint *, device const uchar *, device uchar *, uint, uint);
template [[host_name("v2_round4")]] kernel void v2_round<4>(device const uint *, device uint *,
        device const uint *, device atomic_uint *, device const uchar *, device uchar *, uint, uint);
template [[host_name("v2_round5")]] kernel void v2_round<5>(device const uint *, device uint *,
        device const uint *, device atomic_uint *, device const uchar *, device uchar *, uint, uint);
template [[host_name("v2_round6")]] kernel void v2_round<6>(device const uint *, device uint *,
        device const uint *, device atomic_uint *, device const uchar *, device uchar *, uint, uint);
template [[host_name("v2_round7")]] kernel void v2_round<7>(device const uint *, device uint *,
        device const uint *, device atomic_uint *, device const uchar *, device uchar *, uint, uint);
template [[host_name("v2_round8")]] kernel void v2_round<8>(device const uint *, device uint *,
        device const uint *, device atomic_uint *, device const uchar *, device uchar *, uint, uint);

/*
** v5 "lean" rounds (an experiment): the same algorithm as v2_round, with a small on-chip footprint so
** that several threadgroups fit per GPU core and overlap their phases.
**  - Entries are grouped by rest with a counting sort (counts, prefix sums, an index array) instead of
**    fixed per-rest lists.
**  - Threads take pairs by number: a binary search over the prefix sums of pair counts finds the rest
**    group, then the pair inside it. So there is no pairs buffer and the load is balanced.
**  - Entry data is not copied on-chip; each pair reads its two records from device memory, which
**    the counting pass just pulled into the GPU caches.
** Needs TG == NRESTS (one thread per rest in the prefix sums).
*/
template <uint R, bool CACHED>
kernel void v5_round(device const uint *data_in [[buffer(0)]],
        device uint *data_out [[buffer(1)]],
        device const uint *nslots_in [[buffer(2)]],
        device atomic_uint *nslots_out [[buffer(3)]],
        device const uchar *side_in [[buffer(4)]],
        device uchar *side_out [[buffer(5)]],
        uint b [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
{
    static_assert(TG == NRESTS, "v5 needs one thread per rest");
    constexpr uint IN_W = WORDS(IN_BITS(R));
    constexpr uint IN_MW = MAIN_W(R - 1), IN_SB = SIDE_B(R - 1);
    constexpr uint E = REC_W(R - 1);            // input record: main data words + tree word
    constexpr uint OUT_W = WORDS(OUT_BITS(R));
    threadgroup atomic_uint cnt[NRESTS];
    threadgroup ushort start[NRESTS + 1];       // first position of each rest group in idx[]
    threadgroup uint pstart[NRESTS + 1];        // first pair number of each rest group
    threadgroup ushort idx[NSLOTS];             // slots, grouped by rest
    threadgroup uchar rest_of[NSLOTS];
    threadgroup uint tcache[(R >= 2) ? NSLOTS : 1];
    threadgroup uint sum_c[TG / 32], sum_p[TG / 32];
    threadgroup uint cache[CACHED ? NSLOTS : 1][IN_W];   // v6: entry data on-chip

    atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. count the entries of each rest
    uint n = min(nslots_in[b], (uint)NSLOTS);
    for (uint s = tid; s < n; s += TG)
      {
        uint w0 = IN_MW ? data_in[SIDX(b, s) * E] :
                (IN_SB == 1 ? (uint)side_in[SIDX(b, s)] << 24
                            : (uint)((device const ushort *)side_in)[SIDX(b, s)] << 16);
        uint rest = w0 >> 24;
        rest_of[s] = (uchar)rest;
        if (CACHED)
          {
            for (uint k = 0; k < IN_MW; k++)
                cache[s][k] = data_in[SIDX(b, s) * E + k];
            if (IN_SB == 1)
                cache[s][IN_MW] = (uint)side_in[SIDX(b, s)] << 24;
            else if (IN_SB == 2)
                cache[s][IN_MW] = (uint)((device const ushort *)side_in)[SIDX(b, s)] << 16;
          }
        if (R >= 2)
            tcache[s] = data_in[SIDX(b, s) * E + IN_MW];
        atomic_fetch_add_explicit(&cnt[rest], 1u, memory_order_relaxed);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. prefix sums of entries and pairs per rest (thread tid = rest tid)
    uint c = atomic_load_explicit(&cnt[tid], memory_order_relaxed);
    uint pc = c ? c * (c - 1) / 2 : 0;
    uint ce = simd_prefix_exclusive_sum(c), pe = simd_prefix_exclusive_sum(pc);
    if (lane == 31)
      {
        sum_c[sg] = ce + c;
        sum_p[sg] = pe + pc;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base_c = 0, base_p = 0;
    for (uint g = 0; g < sg; g++)
      {
        base_c += sum_c[g];
        base_p += sum_p[g];
      }
    start[tid] = (ushort)(base_c + ce);
    pstart[tid] = base_p + pe;
    if (tid == NRESTS - 1)
      {
        start[NRESTS] = (ushort)(base_c + ce + c);
        pstart[NRESTS] = base_p + pe + pc;
      }
    atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed);   // reused as a fill counter
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. place each slot in its rest group
    for (uint s = tid; s < n; s += TG)
      {
        uint rest = rest_of[s];
        uint pos = start[rest] + atomic_fetch_add_explicit(&cnt[rest], 1u, memory_order_relaxed);
        idx[pos] = (ushort)s;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. every pair inside a rest group collides on the whole digit; take pairs by number
    uint npairs = pstart[NRESTS];
    for (uint p = tid; p < npairs; p += TG)
      {
        uint lo = 0, hi = NRESTS;               // the last rest with pstart[rest] <= p
        while (hi - lo > 1)
          {
            uint mid = (lo + hi) >> 1;
            if (pstart[mid] <= p)
                lo = mid;
            else
                hi = mid;
          }
        uint q = p - pstart[lo];
        uint cg = start[lo + 1] - start[lo];
        uint i = 0;
        while (q >= cg - 1 - i)
          {
            q -= cg - 1 - i;
            i++;
          }
        uint sa = idx[start[lo] + i], sb = idx[start[lo] + i + 1 + q];
        if (R >= 2 && shares_child(tcache[sa], tcache[sb]))
            continue;
        uint x[IN_W + 1];
        if (CACHED)
            for (uint k = 0; k < IN_W; k++)
                x[k] = cache[sa][k] ^ cache[sb][k];
        else
          {
            device const uint *ea = data_in + SIDX(b, sa) * E, *eb = data_in + SIDX(b, sb) * E;
            for (uint k = 0; k < IN_MW; k++)
                x[k] = ea[k] ^ eb[k];
            if (IN_SB == 1)
                x[IN_MW] = ((uint)side_in[SIDX(b, sa)] ^ (uint)side_in[SIDX(b, sb)]) << 24;
            else if (IN_SB == 2)
                x[IN_MW] = ((uint)((device const ushort *)side_in)[SIDX(b, sa)] ^
                            (uint)((device const ushort *)side_in)[SIDX(b, sb)]) << 16;
          }
        x[IN_W] = 0;
        uint o[OUT_W];
        uint nonzero = 0;
        for (uint k = 0; k < OUT_W; k++)
          {
            o[k] = (x[k] << 20) | (x[k + 1] >> 12);
            nonzero |= o[k];
          }
        if (!nonzero)
            continue;
        uint bucket = (x[0] >> 12) & (NBUCKETS - 1);
        uint slot = atomic_fetch_add_explicit(&nslots_out[bucket], 1u, memory_order_relaxed);
        if (slot >= NSLOTS)
            continue;
        device uint *d = data_out + SIDX(bucket, slot) * REC_W(R);
        for (uint k = 0; k < MAIN_W(R); k++)
            d[k] = o[k];
        d[MAIN_W(R)] = (b << 20) | (sa << 10) | sb;
        if (SIDE_B(R) == 1)
            side_out[SIDX(bucket, slot)] = (uchar)(o[MAIN_W(R)] >> 24);
        else if (SIDE_B(R) == 2)
            ((device ushort *)side_out)[SIDX(bucket, slot)] = (ushort)(o[MAIN_W(R)] >> 16);
      }
}

#define V5_INST(N) template [[host_name("v5_round" #N)]] kernel void v5_round<N, false>(device const uint *, \
        device uint *, device const uint *, device atomic_uint *, device const uchar *, device uchar *, \
        uint, uint, uint, uint); \
    template [[host_name("v6_round" #N)]] kernel void v5_round<N, true>(device const uint *, \
        device uint *, device const uint *, device atomic_uint *, device const uchar *, device uchar *, \
        uint, uint, uint, uint);
V5_INST(1)
V5_INST(2)
V5_INST(3)
V5_INST(4)
V5_INST(5)
V5_INST(6)
V5_INST(7)
V5_INST(8)

/*
** Final round: in each bucket of round 8, pairs whose remaining 28 bits (rest_8 || digit 9) are
** equal XOR to zero on all 200 bits. Report them; the host expands and checks them.
*/
kernel void v2_final(device const uint *data_in [[buffer(0)]],
        device const uint *nslots_in [[buffer(1)]],
        device cands_t *cands [[buffer(2)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]])
{
    threadgroup uint cache[NSLOTS];
    threadgroup uint tcache[NSLOTS];
    threadgroup ushort lists[NRESTS][LIST_CAP];
    threadgroup atomic_uint list_len[NRESTS];

    for (uint bi = 0; bi < BPT; bi++)
      {
        uint b = tgid * BPT + bi;
        if (bi)     // every thread is done with the previous bucket's lists and cache
            threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = tid; r < NRESTS; r += TG)
            atomic_store_explicit(&list_len[r], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint n = min(nslots_in[b], (uint)NSLOTS);
        for (uint s = tid; s < n; s += TG)
          {
            device const uint *e = data_in + SIDX(b, s) * REC_W(8);     // 1 data word + tree
            uint w = e[0];
            cache[s] = w;
            tcache[s] = e[1];
            uint pos = atomic_fetch_add_explicit(&list_len[w >> 24], 1u, memory_order_relaxed);
            if (pos < LIST_CAP)
                lists[w >> 24][pos] = (ushort)s;
          }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = tid; r < NRESTS; r += TG)
          {
            uint len = min(atomic_load_explicit(&list_len[r], memory_order_relaxed), (uint)LIST_CAP);
            for (uint i = 0; i + 1 < len; i++)
                for (uint j = i + 1; j < len; j++)
                  {
                    uint sa = lists[r][i], sb = lists[r][j];
                    if ((cache[sa] ^ cache[sb]) >> 4)
                        continue;   // the 20 bits of digit 9 differ
                    if (shares_child(tcache[sa], tcache[sb]))
                        continue;   // duplicate leaves
                    device atomic_uint *cn = (device atomic_uint *)&cands->n;
                    uint c = atomic_fetch_add_explicit(cn, 1u, memory_order_relaxed);
                    if (c < MAX_CANDS)
                        cands->cand[c] = (b << 20) | (sa << 10) | sb;
                  }
          }
      }
}
