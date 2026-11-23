// Equihash (200,9) solver kernels for Apple GPUs (Metal Shading Language).
//
// A port of SILENTARMY's OpenCL kernels (input.cl):
//   Copyright (c) 2016 Marc Bevand. The MIT License (MIT).
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//   and associated documentation files (the "Software"), to deal in the Software without
//   restriction, including without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
//   Software is furnished to do so, subject to the following conditions: The above copyright
//   notice and this permission notice shall be included in all copies or substantial portions of
//   the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
//
// The algorithm and the hash-table layout are unchanged (see the layout notes in input.cl). The
// host passes the tunables below as preprocessor macros when it compiles this file at runtime.

#include <metal_stdlib>
using namespace metal;

#define PARAM_N                 200
#define PARAM_K                 9
#define PREFIX                  (PARAM_N / (PARAM_K + 1))
#define NR_INPUTS               (1 << PREFIX)
#define APX_NR_ELMS_LOG         (PREFIX + 1)
#ifndef NR_ROWS_LOG
#define NR_ROWS_LOG             20
#endif
#define OPTIM_SIMPLIFY_ROUND    1
#if NR_ROWS_LOG == 18
#define OVERHEAD                3
#elif NR_ROWS_LOG == 19
#define OVERHEAD                5
#elif NR_ROWS_LOG == 20 && OPTIM_SIMPLIFY_ROUND
#define OVERHEAD                6
#elif NR_ROWS_LOG == 20
#define OVERHEAD                9
#endif
#define NR_ROWS                 (1 << NR_ROWS_LOG)
#define NR_SLOTS                ((1 << (APX_NR_ELMS_LOG - NR_ROWS_LOG)) * OVERHEAD)
#define SLOT_LEN                32
#define ZCASH_BLOCK_HEADER_LEN  140
#define ZCASH_HASH_LEN          50
#define MAX_SOLS                10
#ifndef COLL_DATA_SIZE_PER_TH
#define COLL_DATA_SIZE_PER_TH   (NR_SLOTS * 5)
#endif
// threads per threadgroup in the round kernels (sizes the threadgroup arrays)
#ifndef DIAG_MODE
#define DIAG_MODE               0   /* profiling only: 1 skip XOR+store, 2 XOR no store, 3 atomic only, 4 writes only */
#endif
#ifndef TG_SIZE
#define TG_SIZE                 64
#endif
#if (NR_SLOTS < 16)
#define BITS_PER_ROW            4
#define ROWS_PER_UINT           8
#define ROW_MASK                0x0F
#else
#define BITS_PER_ROW            8
#define ROWS_PER_UINT           4
#define ROW_MASK                0xFF
#endif
#define xi_offset_for_round(round)  (8 + ((round) / 2) * 4)
#if NR_ROWS_LOG == 20
#define FIRST_WORDS_LEN         1   /* unused: every pair in a row collides */
#else
#define FIRST_WORDS_LEN         ((NR_SLOTS + 2) * TG_SIZE)
#endif

typedef struct
{
    uint    nr;
    uint    likely_invalids;
    uchar   valid[MAX_SOLS];
    uint    values[MAX_SOLS][(1 << PARAM_K)];
} sols_t;

constant ulong blake_iv[8] =
{
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

/*
** Reset the row counters of a hash table.
*/
kernel void kernel_init_ht(device uint *rowCounters [[buffer(0)]],
        uint gid [[thread_position_in_grid]])
{
    rowCounters[gid] = 0;
}

static inline device atomic_uint *as_atomic(device uint *p)
{
    return (device atomic_uint *)p;
}

/*
** Store an Xi in the hash table, see input.cl.
** Return 0 if successfully stored, or 1 if the row overflowed.
*/
static uint ht_store(uint round, device char *ht, uint i,
        ulong xi0, ulong xi1, ulong xi2, ulong xi3, device uint *rowCounters)
{
    uint row;
#if NR_ROWS_LOG == 18
    if (!(round % 2))
        row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
    else
        row = ((xi0 & 0xc0000) >> 2) |
            ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
            ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
    if (!(round % 2))
        row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
    else
        row = ((xi0 & 0xe0000) >> 1) |
            ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
            ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
    if (!(round % 2))
        row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
    else
        row = ((xi0 & 0xf0000) >> 0) |
            ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
            ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif
    xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
    xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
    xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
    device char *p = ht + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row / ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW * (row % ROWS_PER_UINT);
#if DIAG_MODE == 4
    uint cnt = (i ^ (i >> 7)) % NR_SLOTS;   // fake slot, no atomic (profiling only)
#else
    uint xcnt = atomic_fetch_add_explicit(as_atomic(rowCounters + rowIdx), 1u << rowOffset,
            memory_order_relaxed);
    uint cnt = (xcnt >> rowOffset) & ROW_MASK;
#endif
    if (cnt >= NR_SLOTS)
      {
        // avoid overflows
        atomic_fetch_sub_explicit(as_atomic(rowCounters + rowIdx), 1u << rowOffset,
                memory_order_relaxed);
        return 1;
      }
    p += cnt * SLOT_LEN + xi_offset_for_round(round);
#if DIAG_MODE == 3
    if (i == 0xffffffffu)       // never true: atomic only (profiling only)
        *(device uint *)(p - 4) = i;
    return 0;
#endif
    // store "i" (always 4 bytes before Xi)
    *(device uint *)(p - 4) = i;
    if (round == 0 || round == 1)
      {
        // store 24 bytes
        *(device ulong *)(p + 0) = xi0;
        *(device ulong *)(p + 8) = xi1;
        *(device ulong *)(p + 16) = xi2;
      }
    else if (round == 2)
      {
        // store 20 bytes
        *(device uint *)(p + 0) = (uint)xi0;
        *(device ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
        *(device ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
      }
    else if (round == 3)
      {
        // store 16 bytes
        *(device uint *)(p + 0) = (uint)xi0;
        *(device ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
        *(device uint *)(p + 12) = (uint)(xi1 >> 32);
      }
    else if (round == 4)
      {
        // store 16 bytes
        *(device ulong *)(p + 0) = xi0;
        *(device ulong *)(p + 8) = xi1;
      }
    else if (round == 5)
      {
        // store 12 bytes
        *(device ulong *)(p + 0) = xi0;
        *(device uint *)(p + 8) = (uint)xi1;
      }
    else if (round == 6 || round == 7)
      {
        // store 8 bytes
        *(device uint *)(p + 0) = (uint)xi0;
        *(device uint *)(p + 4) = (uint)(xi0 >> 32);
      }
    else if (round == 8)
      {
        // store 4 bytes
        *(device uint *)(p + 0) = (uint)xi0;
      }
    return 0;
}

static inline ulong rotr64(ulong x, uint n)
{
    return (x >> n) | (x << (64 - n));
}

#define mix(va, vb, vc, vd, x, y) \
    va = (va + vb + x); \
    vd = rotr64(vd ^ va, 32); \
    vc = (vc + vd); \
    vb = rotr64(vb ^ vc, 24); \
    va = (va + vb + y); \
    vd = rotr64(vd ^ va, 16); \
    vc = (vc + vd); \
    vb = rotr64(vb ^ vc, 63);

/*
** Round 0: BLAKE2b of the header + each input index, two Xi per hash.
** blake_state is the state after the first 128-byte block; the second block is the last 12
** (zero) nonce bytes plus the 4-byte input index, so only message word 1 is non-zero.
*/
kernel void kernel_round0(constant ulong *blake_state [[buffer(0)]],
        device char *ht [[buffer(1)]],
        device uint *rowCounters [[buffer(2)]],
        uint tid [[thread_position_in_grid]],
        uint nthreads [[threads_per_grid]])
{
    ulong v[16];
    uint inputs_per_thread = NR_INPUTS / nthreads;
    uint input = tid * inputs_per_thread;
    uint input_end = (tid + 1) * inputs_per_thread;
    while (input < input_end)
      {
        // shift "i" to occupy the high 32 bits of the second ulong word in the message block
        ulong word1 = (ulong)input << 32;
        v[0] = blake_state[0];
        v[1] = blake_state[1];
        v[2] = blake_state[2];
        v[3] = blake_state[3];
        v[4] = blake_state[4];
        v[5] = blake_state[5];
        v[6] = blake_state[6];
        v[7] = blake_state[7];
        v[8] =  blake_iv[0];
        v[9] =  blake_iv[1];
        v[10] = blake_iv[2];
        v[11] = blake_iv[3];
        v[12] = blake_iv[4];
        v[13] = blake_iv[5];
        v[14] = blake_iv[6];
        v[15] = blake_iv[7];
        // mix in length of data
        v[12] ^= ZCASH_BLOCK_HEADER_LEN + 4 /* length of "i" */;
        // last block
        v[14] ^= (ulong)-1;

        // round 1
        mix(v[0], v[4], v[8],  v[12], 0, word1);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 2
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], word1, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 3
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, word1);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 4
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, word1);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 5
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, word1);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 6
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], word1, 0);
        // round 7
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], word1, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 8
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, word1);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 9
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], word1, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 10
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], word1, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 11
        mix(v[0], v[4], v[8],  v[12], 0, word1);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 12
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], word1, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);

        // compress v into the blake state; this produces the 50-byte hash (two Xi values)
        ulong h[7];
        h[0] = blake_state[0] ^ v[0] ^ v[8];
        h[1] = blake_state[1] ^ v[1] ^ v[9];
        h[2] = blake_state[2] ^ v[2] ^ v[10];
        h[3] = blake_state[3] ^ v[3] ^ v[11];
        h[4] = blake_state[4] ^ v[4] ^ v[12];
        h[5] = blake_state[5] ^ v[5] ^ v[13];
        h[6] = (blake_state[6] ^ v[6] ^ v[14]) & 0xffff;

        // store the two Xi values in the hash table
        ht_store(0, ht, input * 2,
                h[0],
                h[1],
                h[2],
                h[3], rowCounters);
        ht_store(0, ht, input * 2 + 1,
                (h[3] >> 8) | (h[4] << (64 - 8)),
                (h[4] >> 8) | (h[5] << (64 - 8)),
                (h[5] >> 8) | (h[6] << (64 - 8)),
                (h[6] >> 8), rowCounters);
        input++;
      }
}

#if NR_ROWS_LOG == 18 && NR_SLOTS <= (1 << 7)
#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 14) | ((slot1 & 0x7f) << 7) | (slot0 & 0x7f))
#define DECODE_ROW(REF)   (REF >> 14)
#define DECODE_SLOT1(REF) ((REF >> 7) & 0x7f)
#define DECODE_SLOT0(REF) (REF & 0x7f)
#elif NR_ROWS_LOG == 19 && NR_SLOTS <= (1 << 6)
#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 13) | ((slot1 & 0x3f) << 6) | (slot0 & 0x3f)) /* 1 spare bit */
#define DECODE_ROW(REF)   (REF >> 13)
#define DECODE_SLOT1(REF) ((REF >> 6) & 0x3f)
#define DECODE_SLOT0(REF) (REF & 0x3f)
#elif NR_ROWS_LOG == 20 && NR_SLOTS <= (1 << 6)
#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 12) | ((slot1 & 0x3f) << 6) | (slot0 & 0x3f))
#define DECODE_ROW(REF)   (REF >> 12)
#define DECODE_SLOT1(REF) ((REF >> 6) & 0x3f)
#define DECODE_SLOT0(REF) (REF & 0x3f)
#else
#error "unsupported NR_ROWS_LOG"
#endif

/*
** Read a ulong that is only 4-byte aligned.
*/
static inline ulong half_aligned_long(device ulong *p, uint offset)
{
    device char *c = (device char *)p;
    return ((ulong)*(device uint *)(c + offset + 0)) |
        (((ulong)*(device uint *)(c + offset + 4)) << 32);
}

static inline uint well_aligned_int(device ulong *p, uint offset)
{
    return *(device uint *)((device char *)p + offset);
}

/*
** XOR a pair of Xi values computed at "round - 1" and store the result in the hash table being
** built for "round", see input.cl.
** Return 0 if successfully stored, or 1 if the row overflowed.
*/
static uint xor_and_store(uint round, device char *ht_dst, uint row,
        uint slot_a, uint slot_b, device ulong *a, device ulong *b,
        device uint *rowCounters)
{
    ulong xi0 = 0, xi1 = 0, xi2 = 0;
    if (round == 1 || round == 2)
      {
        // xor 24 bytes
        xi0 = a[0] ^ b[0];
        xi1 = a[1] ^ b[1];
        xi2 = a[2] ^ b[2];
        if (round == 2)
          {
            // skip padding byte
            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
            xi1 = (xi1 >> 8) | (xi2 << (64 - 8));
            xi2 = (xi2 >> 8);
          }
      }
    else if (round == 3)
      {
        // xor 20 bytes
        xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
        xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
        xi2 = well_aligned_int(a, 16) ^ well_aligned_int(b, 16);
      }
    else if (round == 4 || round == 5)
      {
        // xor 16 bytes
        xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
        xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
        xi2 = 0;
        if (round == 4)
          {
            // skip padding byte
            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
            xi1 = (xi1 >> 8);
          }
      }
    else if (round == 6)
      {
        // xor 12 bytes
        xi0 = a[0] ^ b[0];
        xi1 = *(device uint *)(a + 1) ^ *(device uint *)(b + 1);
        xi2 = 0;
        // skip padding byte
        xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
        xi1 = (xi1 >> 8);
      }
    else if (round == 7 || round == 8)
      {
        // xor 8 bytes
        xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
        xi1 = 0;
        xi2 = 0;
        if (round == 8)
          {
            // skip padding byte
            xi0 = (xi0 >> 8);
          }
      }
    // invalid solutions (which start happening in round 5) have duplicate inputs and xor to
    // zero, so discard them
    if (!xi0 && !xi1)
        return 0;
#if DIAG_MODE == 2
    if (xi0 == 0x5555aaaa5555aaaaUL)    // never true; keeps the loads alive
        *(device ulong *)ht_dst = xi1 ^ xi2;
    return 0;
#endif
    return ht_store(round, ht_dst, ENCODE_INPUTS(row, slot_a, slot_b),
            xi0, xi1, xi2, 0, rowCounters);
}

/*
** One Equihash round: each thread handles one row of the previous table, finds the colliding
** pairs in it, and the whole threadgroup then XORs and stores those pairs.
*/
static void equihash_round(uint round,
        device char *ht_src,
        device char *ht_dst,
        threadgroup uchar *first_words_data,
        threadgroup uint *collisionsData,
        threadgroup atomic_uint *collisionsNum,
        device uint *rowCountersSrc,
        device uint *rowCountersDst,
        uint tid, uint tlid, uint tgsize)
{
    uint xi_offset = xi_offset_for_round(round - 1);
#if NR_ROWS_LOG != 20
    threadgroup uchar *first_words = &first_words_data[(NR_SLOTS + 2) * tlid];
#endif
#if NR_ROWS_LOG == 18
    uchar mask = ((!(round % 2)) ? 0x03 : 0x30);
#elif NR_ROWS_LOG == 19
    uchar mask = ((!(round % 2)) ? 0x01 : 0x10);
#endif
    // METAL PORT: every colliding pair must survive, because a solution is a tree of 511 pairs
    // and losing any one of them loses the solution. SILENTARMY sizes the shared buffer for
    // the worst row on every thread (NR_SLOTS * 5 entries each), which costs most of the
    // threadgroup memory. Here the buffer is sized for the typical load (COLL_DATA_SIZE_PER_TH
    // per thread), and a thread that finds it full processes its extra pairs itself.
    if (tlid == 0)
        atomic_store_explicit(collisionsNum, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device char *p = ht_src + tid * NR_SLOTS * SLOT_LEN;
    uint rowIdx = tid / ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW * (tid % ROWS_PER_UINT);
    uint cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
    cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round
    if (cnt > 1)
      {
        p += xi_offset;
#if NR_ROWS_LOG != 20
        for (uint i = 0; i < cnt; i++, p += SLOT_LEN)
            first_words[i] = (*(device uchar *)p) & mask;
#endif
        // find collisions
        for (uint i = 0; i < cnt - 1; i++)
          {
#if NR_ROWS_LOG != 20
            uchar data_i = first_words[i];
#endif
            uint collision = (tid << 10) | (i << 5) | (i + 1);
            for (uint j = i + 1; j < cnt; j++, collision++)
              {
#if NR_ROWS_LOG != 20
                if (data_i != first_words[j])
                    continue;
#endif
                // (with NR_ROWS_LOG == 20 the row is the whole colliding PREFIX, so every
                // pair in a row collides)
                uint index = atomic_fetch_add_explicit(collisionsNum, 1u, memory_order_relaxed);
                if (index < COLL_DATA_SIZE_PER_TH * TG_SIZE)
                    collisionsData[index] = collision;
                else
                  {
                    device char *ptr = ht_src + tid * NR_SLOTS * SLOT_LEN + xi_offset;
                    xor_and_store(round, ht_dst, tid, i, j,
                            (device ulong *)(ptr + i * SLOT_LEN),
                            (device ulong *)(ptr + j * SLOT_LEN), rowCountersDst);
                  }
              }
          }
      }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint totalCollisions = min(atomic_load_explicit(collisionsNum, memory_order_relaxed),
            (uint)(COLL_DATA_SIZE_PER_TH * TG_SIZE));
    for (uint index = tlid; index < totalCollisions; index += tgsize)
      {
        uint collision = collisionsData[index];
        uint collisionThreadId = collision >> 10;
        uint i = (collision >> 5) & 0x1F;
        uint j = collision & 0x1F;
        device char *ptr = ht_src + collisionThreadId * NR_SLOTS * SLOT_LEN + xi_offset;
        device ulong *a = (device ulong *)(ptr + i * SLOT_LEN);
        device ulong *b = (device ulong *)(ptr + j * SLOT_LEN);
#if DIAG_MODE != 1
        xor_and_store(round, ht_dst, collisionThreadId, i, j, a, b, rowCountersDst);
#endif
      }
}

/*
** kernel_round1 ... kernel_round7
*/
#define KERNEL_ROUND(N) \
kernel void kernel_round ## N(device char *ht_src [[buffer(0)]], \
        device char *ht_dst [[buffer(1)]], \
        device uint *rowCountersSrc [[buffer(2)]], \
        device uint *rowCountersDst [[buffer(3)]], \
        uint tid [[thread_position_in_grid]], \
        uint tlid [[thread_position_in_threadgroup]], \
        uint tgsize [[threads_per_threadgroup]]) \
{ \
    threadgroup uchar first_words_data[FIRST_WORDS_LEN]; \
    threadgroup uint collisionsData[COLL_DATA_SIZE_PER_TH * TG_SIZE]; \
    threadgroup atomic_uint collisionsNum; \
    equihash_round(N, ht_src, ht_dst, first_words_data, collisionsData, &collisionsNum, \
            rowCountersSrc, rowCountersDst, tid, tlid, tgsize); \
}
KERNEL_ROUND(1)
KERNEL_ROUND(2)
KERNEL_ROUND(3)
KERNEL_ROUND(4)
KERNEL_ROUND(5)
KERNEL_ROUND(6)
KERNEL_ROUND(7)

// kernel_round8 also resets the solutions buffer for kernel_sols
kernel void kernel_round8(device char *ht_src [[buffer(0)]],
        device char *ht_dst [[buffer(1)]],
        device uint *rowCountersSrc [[buffer(2)]],
        device uint *rowCountersDst [[buffer(3)]],
        device sols_t *sols [[buffer(4)]],
        uint tid [[thread_position_in_grid]],
        uint tlid [[thread_position_in_threadgroup]],
        uint tgsize [[threads_per_threadgroup]])
{
    threadgroup uchar first_words_data[FIRST_WORDS_LEN];
    threadgroup uint collisionsData[COLL_DATA_SIZE_PER_TH * TG_SIZE];
    threadgroup atomic_uint collisionsNum;
    equihash_round(8, ht_src, ht_dst, first_words_data, collisionsData, &collisionsNum,
            rowCountersSrc, rowCountersDst, tid, tlid, tgsize);
    if (!tid)
      {
        sols->nr = 0;
        sols->likely_invalids = 0;
      }
}

static inline uint expand_ref(device char *ht, uint xi_offset, uint row, uint slot)
{
    return *(device uint *)(ht + row * NR_SLOTS * SLOT_LEN + slot * SLOT_LEN + xi_offset - 4);
}

/*
** Expand references to inputs. Return 1 if so far the solution appears valid, or 0 otherwise
** (an invalid solution has duplicate inputs, which is detected at round 0).
*/
static uint expand_refs(thread uint *ins, uint nr_inputs, device char *ht0, device char *ht1,
        uint round)
{
    device char *ht = (round % 2) ? ht1 : ht0;
    uint i = nr_inputs - 1;
    uint j = nr_inputs * 2 - 1;
    uint xi_offset = xi_offset_for_round(round);
    int dup_to_watch = -1;
    while (true)
      {
        ins[j] = expand_ref(ht, xi_offset, DECODE_ROW(ins[i]), DECODE_SLOT1(ins[i]));
        ins[j - 1] = expand_ref(ht, xi_offset, DECODE_ROW(ins[i]), DECODE_SLOT0(ins[i]));
        if (!round)
          {
            if (dup_to_watch == -1)
                dup_to_watch = ins[j];
            else if (ins[j] == (uint)dup_to_watch || ins[j - 1] == (uint)dup_to_watch)
                return 0;
          }
        if (!i)
            break;
        i--;
        j -= 2;
      }
    return 1;
}

/*
** Expand a potential solution to its 512 inputs and, if it still looks valid, store it.
*/
static void potential_sol(device char *ht0, device char *ht1, device sols_t *sols,
        uint ref0, uint ref1)
{
    uint values_tmp[(1 << PARAM_K)];
    uint nr_values = 0;
    values_tmp[nr_values++] = ref0;
    values_tmp[nr_values++] = ref1;
    uint round = PARAM_K - 1;
    do
      {
        round--;
        if (!expand_refs(values_tmp, nr_values, ht0, ht1, round))
            return;
        nr_values *= 2;
      }
    while (round > 0);
    // solution appears valid, copy it to sols
    uint sol_i = atomic_fetch_add_explicit(as_atomic(&sols->nr), 1u, memory_order_relaxed);
    if (sol_i >= MAX_SOLS)
        return;
    for (uint i = 0; i < (1 << PARAM_K); i++)
        sols->values[sol_i][i] = values_tmp[i];
    sols->valid[sol_i] = 1;
}

/*
** Scan the last hash table for solutions.
*/
kernel void kernel_sols(device char *ht0 [[buffer(0)]],
        device char *ht1 [[buffer(1)]],
        device sols_t *sols [[buffer(2)]],
        device uint *rowCountersSrc [[buffer(3)]],
        uint tid [[thread_position_in_grid]])
{
    device char *htabs_last = ((PARAM_K - 1) % 2) ? ht1 : ht0; // table filled at the last round
    uint xi_offset = xi_offset_for_round(PARAM_K - 1);
    // in the final hash table, look for a match on both the bits that are part of the previous
    // PREFIX colliding bits, and the last PREFIX bits
    uint mask = 0xffffff;
    device char *a = htabs_last + tid * NR_SLOTS * SLOT_LEN;
    uint rowIdx = tid / ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW * (tid % ROWS_PER_UINT);
    uint cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
    cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
    a += xi_offset;
    bool found = false;
    uint ref_i = 0, ref_j = 0;
    for (uint i = 0; i < cnt && !found; i++, a += SLOT_LEN)
      {
        uint a_data = (*(device uint *)a) & mask;
        device char *b = a + SLOT_LEN;
        for (uint j = i + 1; j < cnt; j++, b += SLOT_LEN)
          {
            if (a_data == ((*(device uint *)b) & mask))
              {
                ref_i = *(device uint *)(a - 4);
                ref_j = *(device uint *)(b - 4);
                found = true;
                break;
              }
          }
      }
    if (found)
        potential_sol(ht0, ht1, sols, ref_i, ref_j);
}
