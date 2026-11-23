// metal-eq: an Equihash (200,9) solver for Apple GPUs, using Metal.
//
// It started as a port of SILENTARMY (Copyright (c) 2016 Marc Bevand, The MIT License; the full
// notice is in LICENSE), which is still here as --v1. The default solver is our own bucketed
// design in equihash_v2.metal. What is different from SILENTARMY:
//  * Metal instead of OpenCL, so there is no OpenCL-on-Metal translation layer;
//  * several Equihash runs are in flight at once (--inflight). One process keeps the GPU busy:
//    the CPU checks the solutions of one run while the GPU works on the next ones. (SILENTARMY
//    waits for each run, so it needs ~16 processes to fill this GPU.)
//
// Modes:
//   metal-eq --nonces N [--verify-out FILE]   benchmark on a fixed header
//   metal-eq --profile RUNS                   GPU time per pass
//   metal-eq --mining                        SILENTARMY's mining protocol on stdin/stdout:
//       in:  "<target hex> <job id> <header hex, 108 bytes> <nonce left part hex>"
//       out: "sol: <job id> <ntime> <nonce right part hex> fd4005<solution hex>"
//            "status: <solutions> <shares>"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "blake.h"
#include "cpuhash.h"
#include "sha256.h"
#include "kernel_src.h"
#include "kernel_v2_src.h"

#define PARAM_N                 200
#define PARAM_K                 9
#define PREFIX                  (PARAM_N / (PARAM_K + 1))
#define NR_INPUTS               (1 << PREFIX)
#define APX_NR_ELMS_LOG         (PREFIX + 1)
#define NR_ROWS_LOG             20
#define OVERHEAD                6
#define NR_ROWS                 (1 << NR_ROWS_LOG)
#define NR_SLOTS                ((1 << (APX_NR_ELMS_LOG - NR_ROWS_LOG)) * OVERHEAD)
#define SLOT_LEN                32
#define HT_SIZE                 ((size_t)NR_ROWS * NR_SLOTS * SLOT_LEN)
#define ROWS_PER_UINT           ((NR_SLOTS < 16) ? 8 : 4)
#define ZCASH_BLOCK_HEADER_LEN  140
#define ZCASH_BLOCK_OFFSET_NTIME (4 + 3 * 32)
#define ZCASH_NONCE_LEN         32
#define ZCASH_SOLSIZE_LEN       3
#define ZCASH_SOLSIZE_HEX       "fd4005"
#define ZCASH_SOL_LEN           ((1 << PARAM_K) * (PREFIX + 1) / 8)
#define N_ZERO_BYTES            12
#define ZCASH_HASH_LEN          50
#define MAX_SOLS                10
#define SHA256_TARGET_LEN       32
#define SOL_SIZE                ((1 << PARAM_K) * 4)

// v2 (bucketed) layout; keep in sync with equihash_v2.metal
#define V2_NBUCKETS             4096
#define V2_NSLOTS               640
#define V2_DATA_WORDS           6
#define V2_MAX_CANDS            4096
#define V2_ROUNDS               9       // rounds 0..8 keep their trees
// words per entry record in round r: the data words, then the tree word (equihash_v2.metal E_WORDS)
static const unsigned v2_e_words[V2_ROUNDS] = { 7, 6, 6, 5, 4, 4, 3, 2, 2 };  // v4.2 split records
// bytes per entry in each round's side array (the last <=16 data bits; equihash_v2.metal SIDE_B)
static const unsigned v2_side_b[V2_ROUNDS] = { 0, 1, 0, 0, 2, 0, 1, 2, 0 };
static size_t v2_side_off[V2_ROUNDS + 1];
static size_t v2_ent_off[V2_ROUNDS + 1];    // byte offset of each round's records; [V2_ROUNDS] = total
typedef struct
{
    uint32_t    n;
    uint32_t    cand[V2_MAX_CANDS];
} cands_t;

typedef struct
{
    uint32_t    nr;
    uint32_t    likely_invalids;
    uint8_t     valid[MAX_SOLS];
    uint32_t    values[MAX_SOLS][(1 << PARAM_K)];
} sols_t;

static int verbose = 0;
static int mining = 0;
static unsigned inflight = 3;       // best for v2 on the M5 Pro
static unsigned tg_size = 64;
static unsigned coll_per_th = 8;    // shared pair buffer per thread; overflow is handled inline
static unsigned round0_threads = 1 << 17;
static unsigned diag_mode = 0;
static unsigned tile = 1;           // --tile N: v2 record layout (see SIDX in equihash_v2.metal)
static unsigned bpt = 1;            // --bpt N: buckets per threadgroup in the v2 rounds and final (BPT)
static int use_v2 = 1;            // the bucketed solver (default); --v1 = the SILENTARMY port
static unsigned v2_tg = 256, v2_tg0 = 256;   // v2 threadgroup sizes: rounds 1..8 + final, round 0
static unsigned v2_max_pairs = 1024;        // v2 pairs buffered per bucket (overflow is processed inline)

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLCommandQueue> queue0;      // --twoq: round 0 (compute-bound) runs on its own queue,
static id<MTLSharedEvent> r0_event;     //   overlapping the memory-bound rounds of earlier runs
static uint64_t r0_seq;
static int two_queues = 0;
static int cpu_sg = -1;             // --cpu-sg N: the CPU hashes N/8 of round 0's inputs (see the kernel);
                                    //   default 1 when benchmarking or mining with v2, else 0
static unsigned cpu_threads = 6;    // --cpu-threads N: CPU hashing threads
static int lean = 6;                // --lean N: 5 = v5 (no data cache), 6 = v6 (cached), 0 = v2/v4
static unsigned lean_from = 5;      // --lean-from R: lean kernels from round R on (v4.1: v6 for rounds 5..8)

static const char *round_kernel_for(unsigned r)
{
    if (lean && r >= lean_from)
        return lean == 5 ? "v5" : "v6";
    return "v2";
}
static id<MTLComputePipelineState> p_init, p_sols, p_round[PARAM_K];
static id<MTLComputePipelineState> p2_round[PARAM_K], p2_final;

// totals
static uint64_t total_sols, total_shares, total_runs, total_dropped;

static void fatal(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void fatal(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    exit(1);
}

static uint64_t now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000 + t.tv_nsec / 1000;
}

static void randomize(void *p, size_t l)
{
    arc4random_buf(p, l);
}

static char *s_hexdump(const void *_a, uint32_t a_len)
{
    const uint8_t *a = _a;
    static char buf[4096];
    uint32_t i;
    for (i = 0; i < a_len && i + 2 < sizeof (buf); i++)
        sprintf(buf + i * 2, "%02x", a[i]);
    buf[i * 2] = 0;
    return buf;
}

static uint8_t hex2val(const char *base, size_t off)
{
    const char c = base[off];
    if (c >= '0' && c <= '9')           return c - '0';
    else if (c >= 'a' && c <= 'f')      return 10 + c - 'a';
    else if (c >= 'A' && c <= 'F')      return 10 + c - 'A';
    fatal("Invalid hex char at offset %zd: ...%c...\n", off, c);
}

/*
** One Equihash run in flight: its own hash tables, so runs never share memory.
*/
@interface Slot : NSObject
{
@public
    uint8_t header[ZCASH_BLOCK_HEADER_LEN];
    char job_id[256];
    uint8_t target[SHA256_TARGET_LEN];
    size_t fixed_nonce_bytes;
}
@property id<MTLBuffer> ht0, ht1, cnt0, cnt1, sols, blake;
@property id<MTLBuffer> ent, side, nsl, cands;   // v2: per-round entry records (data + tree), side bytes, counters
@property id<MTLBuffer> cpuh;                    // --cpu-sg: round-0 hashes computed by the CPU
@property id<MTLCommandBuffer> cb;
@end
@implementation Slot
@end

static id<MTLBuffer> new_buffer(size_t len, MTLResourceOptions opts)
{
    id<MTLBuffer> b = [device newBufferWithLength:len options:opts];
    if (!b)
        fatal("could not allocate a %zu-byte GPU buffer\n", len);
    return b;
}

static Slot *new_slot(void)
{
    Slot *s = [Slot new];
    s.blake = new_buffer(64, MTLResourceStorageModeShared);
    if (use_v2)
      {
        size_t off = 0;
        for (unsigned r = 0; r < V2_ROUNDS; r++)
          {
            v2_ent_off[r] = off;
            off += (size_t)V2_NBUCKETS * V2_NSLOTS * v2_e_words[r] * 4;
          }
        v2_ent_off[V2_ROUNDS] = off;
        // the CPU reads the tree words back to expand candidate solutions
        s.ent = new_buffer(off, MTLResourceStorageModeShared);
        size_t soff = 0;
        for (unsigned r = 0; r < V2_ROUNDS; r++)
          {
            v2_side_off[r] = v2_side_b[r] ? soff : 0;
            soff += (size_t)V2_NBUCKETS * V2_NSLOTS * v2_side_b[r];
          }
        v2_side_off[V2_ROUNDS] = soff;
        s.side = new_buffer(soff, MTLResourceStorageModePrivate);
        s.nsl = new_buffer((size_t)V2_ROUNDS * V2_NBUCKETS * 4, MTLResourceStorageModeShared);
        s.cands = new_buffer(sizeof (cands_t), MTLResourceStorageModeShared);
        if (cpu_sg)
            s.cpuh = new_buffer((size_t)(NR_INPUTS / 256) * cpu_sg * 32 * 7 * 8, MTLResourceStorageModeShared);
        return s;
      }
    s.ht0 = new_buffer(HT_SIZE, MTLResourceStorageModePrivate);
    s.ht1 = new_buffer(HT_SIZE, MTLResourceStorageModePrivate);
    s.cnt0 = new_buffer(NR_ROWS / ROWS_PER_UINT * 4, MTLResourceStorageModePrivate);
    s.cnt1 = new_buffer(NR_ROWS / ROWS_PER_UINT * 4, MTLResourceStorageModePrivate);
    s.sols = new_buffer(sizeof (sols_t), MTLResourceStorageModeShared);
    return s;
}

static id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib, NSString *name)
{
    NSError *err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:name];
    if (!f)
        fatal("kernel %s not found\n", name.UTF8String);
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:f error:&err];
    if (!p)
        fatal("pipeline %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String);
    return p;
}

static void init_metal(void)
{
    device = MTLCreateSystemDefaultDevice();
    if (!device)
        fatal("no Metal device\n");
    queue = [device newCommandQueue];
    queue0 = [device newCommandQueue];
    r0_event = [device newSharedEvent];
    if (use_v2)
      {
        NSString *src2 = [[NSString alloc] initWithBytes:equihash_v2_metal
                length:equihash_v2_metal_len encoding:NSUTF8StringEncoding];
        NSError *err = nil;
        MTLCompileOptions *opts2 = [MTLCompileOptions new];
        // the host and the kernel must agree on these (a mismatch garbles every solution)
        opts2.preprocessorMacros = @{ @"DIAG": @(diag_mode), @"TG": @(v2_tg),
                                      @"NSLOTS": @(V2_NSLOTS), @"MAX_CANDS": @(V2_MAX_CANDS), @"TILE": @(tile), @"BPT": @(bpt),
                                      @"MAX_PAIRS": @(v2_max_pairs), @"CPU_SG": @(cpu_sg) };
        id<MTLLibrary> lib2 = [device newLibraryWithSource:src2 options:opts2 error:&err];
        if (!lib2)
            fatal("Metal compile failed: %s\n", err.localizedDescription.UTF8String);
        p2_round[0] = pipeline(lib2, @"v2_round0");
        for (unsigned r = 1; r < PARAM_K; r++)
            p2_round[r] = pipeline(lib2, [NSString stringWithFormat:@"%s_round%u",
                    round_kernel_for(r), r]);
        p2_final = pipeline(lib2, @"v2_final");
        if (verbose)
            fprintf(stderr, "%s: v2 (bucketed), %u runs in flight (%.2f GB)\n",
                    device.name.UTF8String, inflight,
                    inflight * (double)v2_ent_off[V2_ROUNDS] / 1e9);
        return;
      }
    NSString *src = [[NSString alloc] initWithBytes:equihash_metal length:equihash_metal_len
                                           encoding:NSUTF8StringEncoding];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.preprocessorMacros = @{
        @"TG_SIZE": @(tg_size),
        @"COLL_DATA_SIZE_PER_TH": @(coll_per_th),
        @"NR_ROWS_LOG": @(NR_ROWS_LOG),
        @"DIAG_MODE": @(diag_mode),
    };
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib)
        fatal("Metal compile failed: %s\n", err.localizedDescription.UTF8String);
    p_init = pipeline(lib, @"kernel_init_ht");
    p_sols = pipeline(lib, @"kernel_sols");
    for (unsigned r = 0; r < PARAM_K; r++)
        p_round[r] = pipeline(lib, [NSString stringWithFormat:@"kernel_round%u", r]);
    if (verbose)
        fprintf(stderr, "%s: %u runs in flight (%.1f GB of hash tables), threadgroup %u, "
                "round0 threads %u\n", device.name.UTF8String, inflight,
                inflight * 2.0 * HT_SIZE / 1e9, tg_size, round0_threads);
}

/*
** v2: clear the bucket counters and the candidate count (start of a run).
*/
static void v2_reset(id<MTLCommandBuffer> cb, Slot *s)
{
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:s.nsl range:NSMakeRange(0, s.nsl.length) value:0];
    [blit fillBuffer:s.cands range:NSMakeRange(0, 4) value:0];
    [blit endEncoding];
}

/*
** v2: encode pass 0..8 (rounds) or 9 (final) of a run.
*/
static void v2_encode_pass(id<MTLComputeCommandEncoder> enc, Slot *s, unsigned pass)
{
    const size_t nsl_round = (size_t)V2_NBUCKETS * 4;
    if (pass == 0)
      {
        [enc setComputePipelineState:p2_round[0]];
        [enc setBuffer:s.blake offset:0 atIndex:0];
        [enc setBuffer:s.ent offset:v2_ent_off[0] atIndex:1];
        [enc setBuffer:s.nsl offset:0 atIndex:3];
        uint32_t base = 0;
        [enc setBytes:&base length:4 atIndex:4];
        if (cpu_sg)
            [enc setBuffer:s.cpuh offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(NR_INPUTS, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(v2_tg0, 1, 1)];
      }
    else if (pass < PARAM_K)
      {
        [enc setComputePipelineState:p2_round[pass]];
        [enc setBuffer:s.ent offset:v2_ent_off[pass - 1] atIndex:0];
        [enc setBuffer:s.ent offset:v2_ent_off[pass] atIndex:1];
        [enc setBuffer:s.nsl offset:(pass - 1) * nsl_round atIndex:2];
        [enc setBuffer:s.nsl offset:pass * nsl_round atIndex:3];
        [enc setBuffer:s.side offset:v2_side_off[pass - 1] atIndex:4];
        [enc setBuffer:s.side offset:v2_side_off[pass] atIndex:5];
        unsigned groups = strcmp(round_kernel_for(pass), "v2") ? V2_NBUCKETS : V2_NBUCKETS / bpt;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(v2_tg, 1, 1)];
      }
    else
      {
        [enc setComputePipelineState:p2_final];
        [enc setBuffer:s.ent offset:v2_ent_off[PARAM_K - 1] atIndex:0];
        [enc setBuffer:s.nsl offset:(PARAM_K - 1) * nsl_round atIndex:1];
        [enc setBuffer:s.cands offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(V2_NBUCKETS / bpt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(v2_tg, 1, 1)];
      }
}

/*
** v2: round 0 of slot s in chunks: chunk c of nchunks (inputs [c*N/n, (c+1)*N/n)).
*/
static void v2_encode_r0_chunk(id<MTLComputeCommandEncoder> enc, Slot *s, unsigned c, unsigned n)
{
    uint32_t base = c * (NR_INPUTS / n);
    [enc setComputePipelineState:p2_round[0]];
    [enc setBuffer:s.blake offset:0 atIndex:0];
    [enc setBuffer:s.ent offset:v2_ent_off[0] atIndex:1];
    [enc setBuffer:s.nsl offset:0 atIndex:3];
    [enc setBytes:&base length:4 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(NR_INPUTS / n, 1, 1) threadsPerThreadgroup:MTLSizeMake(v2_tg0, 1, 1)];
}

static void set_blake(Slot *s)
{
    blake2b_state_t blake;
    zcash_blake2b_init(&blake, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
    zcash_blake2b_update(&blake, s->header, 128, 0);
    memcpy(s.blake.contents, blake.h, sizeof (blake.h));
}

/*
** --cpu-sg: the CPU's share of round 0 for the slot's header (blocks until done, ~3 ms).
*/
static void cpu_round0(Slot *s)
{
    uint8_t last16[16] = {0};
    memcpy(last16, s->header + 128, ZCASH_BLOCK_HEADER_LEN - 128);
    cpuhash_run((const uint64_t *)s.blake.contents, last16, cpu_sg * 32, NR_INPUTS / 256,
                (uint64_t *)s.cpuh.contents);
}

/*
** --overlap: one command buffer runs rounds 1..8 + final of slot s and, alongside each of the 8
** rounds (concurrent dispatch, no barrier between the two), 1/8 of round 0 of the next slot t.
** Round 0 is compute-bound and the rounds are memory-bound, so they partly overlap on the GPU.
*/
static int overlap = 0;
static unsigned overlap_chunks = 8;     // --overlap-chunks N: round 0 of the next run in N chunks

static void submit_overlapped(Slot *s, Slot *t)
{
    const size_t nsl_round = (size_t)V2_NBUCKETS * 4;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:s.nsl range:NSMakeRange(nsl_round, s.nsl.length - nsl_round) value:0];
    [blit fillBuffer:s.cands range:NSMakeRange(0, 4) value:0];
    if (t)
        [blit fillBuffer:t.nsl range:NSMakeRange(0, nsl_round) value:0];
    [blit endEncoding];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    for (unsigned pass = 1; pass <= PARAM_K; pass++)
      {
        v2_encode_pass(enc, s, pass);
        if (t && pass <= overlap_chunks)
            v2_encode_r0_chunk(enc, t, pass - 1, overlap_chunks);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
    [enc endEncoding];
    [cb commit];
    s.cb = cb;
}

/*
** Queue one Equihash run for the header in the slot.
*/
static void submit(Slot *s)
{
    set_blake(s);
    if (use_v2 && two_queues)
      {
        // round 0 on queue0, then signal; rounds 1..8 + final on queue, after waiting for it
        uint64_t seq = ++r0_seq;
        id<MTLCommandBuffer> cb0 = [queue0 commandBuffer];
        v2_reset(cb0, s);
        id<MTLComputeCommandEncoder> enc0 = [cb0 computeCommandEncoder];
        v2_encode_pass(enc0, s, 0);
        [enc0 endEncoding];
        [cb0 encodeSignalEvent:r0_event value:seq];
        [cb0 commit];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [cb encodeWaitForEvent:r0_event value:seq];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (unsigned pass = 1; pass <= PARAM_K; pass++)
            v2_encode_pass(enc, s, pass);
        [enc endEncoding];
        [cb commit];
        s.cb = cb;
        return;
      }
    if (use_v2)
      {
        if (cpu_sg)
            cpu_round0(s);      // blocking: done while the GPU works on the runs queued before this one
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        v2_reset(cb, s);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (unsigned pass = 0; pass <= PARAM_K; pass++)
            v2_encode_pass(enc, s, pass);
        [enc endEncoding];
        [cb commit];
        s.cb = cb;
        return;
      }

    id<MTLBuffer> ht[2] = { s.ht0, s.ht1 };
    id<MTLBuffer> cnt[2] = { s.cnt0, s.cnt1 };
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder]; // serial: in-order dispatches
    for (unsigned round = 0; round < PARAM_K; round++)
      {
        [enc setComputePipelineState:p_init];
        [enc setBuffer:cnt[round % 2] offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(NR_ROWS / ROWS_PER_UINT, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:p_round[round]];
        if (!round)
          {
            [enc setBuffer:s.blake offset:0 atIndex:0];
            [enc setBuffer:ht[0] offset:0 atIndex:1];
            [enc setBuffer:cnt[0] offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(round0_threads, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
          }
        else
          {
            [enc setBuffer:ht[(round - 1) % 2] offset:0 atIndex:0];
            [enc setBuffer:ht[round % 2] offset:0 atIndex:1];
            [enc setBuffer:cnt[(round - 1) % 2] offset:0 atIndex:2];
            [enc setBuffer:cnt[round % 2] offset:0 atIndex:3];
            if (round == PARAM_K - 1)
                [enc setBuffer:s.sols offset:0 atIndex:4];
            [enc dispatchThreads:MTLSizeMake(NR_ROWS, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
          }
      }
    [enc setComputePipelineState:p_sols];
    [enc setBuffer:ht[0] offset:0 atIndex:0];
    [enc setBuffer:ht[1] offset:0 atIndex:1];
    [enc setBuffer:s.sols offset:0 atIndex:2];
    [enc setBuffer:cnt[(PARAM_K - 1) % 2] offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(NR_ROWS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    s.cb = cb;
}

/*
** Encode a solution as per the Zcash spec: 512 inputs of 21 bits, big-endian bit order.
*/
static void store_encoded_sol(uint8_t *out, uint32_t *inputs, uint32_t n)
{
    uint32_t byte_pos = 0;
    int32_t bits_left = PREFIX + 1;
    uint8_t x = 0;
    uint8_t x_bits_used = 0;
    while (byte_pos < n)
      {
        if (bits_left >= 8 - x_bits_used)
          {
            x |= inputs[byte_pos] >> (bits_left - 8 + x_bits_used);
            bits_left -= 8 - x_bits_used;
            x_bits_used = 8;
          }
        else if (bits_left > 0)
          {
            uint32_t mask = ~(-1 << (8 - x_bits_used));
            mask = ((~mask) >> bits_left) & mask;
            x |= (inputs[byte_pos] << (8 - x_bits_used - bits_left)) & mask;
            x_bits_used += bits_left;
            bits_left = 0;
          }
        else
          {
            byte_pos++;
            bits_left = PREFIX + 1;
          }
        if (x_bits_used == 8)
          {
            *out++ = x;
            x = x_bits_used = 0;
          }
      }
}

static void sort_pair(uint32_t *a, uint32_t len)
{
    uint32_t *b = a + len;
    uint32_t tmp, need_sorting = 0;
    for (uint32_t i = 0; i < len; i++)
        if (need_sorting || a[i] > b[i])
          {
            need_sorting = 1;
            tmp = a[i];
            a[i] = b[i];
            b[i] = tmp;
          }
        else if (a[i] < b[i])
            return;
}

/*
** Reject solutions with duplicate inputs; put valid ones in Zcash's canonical order.
*/
static int verify_sol(uint32_t *inputs)
{
    uint8_t seen[(1 << (PREFIX + 1)) / 8];
    memset(seen, 0, sizeof (seen));
    for (uint32_t i = 0; i < (1 << PARAM_K); i++)
      {
        if (inputs[i] / 8 >= sizeof (seen))
            return 0;
        uint8_t before = seen[inputs[i] / 8];
        seen[inputs[i] / 8] |= 1 << (inputs[i] & 7);
        if (before == seen[inputs[i] / 8])
            return 0;
      }
    for (uint32_t level = 0; level < PARAM_K; level++)
        for (uint32_t i = 0; i < (1 << PARAM_K); i += (2 << level))
            sort_pair(&inputs[i], 1 << level);
    return 1;
}

static int32_t cmp_target_256(const uint8_t *a, const uint8_t *b)
{
    for (int32_t i = SHA256_TARGET_LEN - 1; i >= 0; i--)
        if (a[i] != b[i])
            return (int32_t)a[i] - b[i];
    return 0;
}

static FILE *verify_out;

static void report(Slot *s, uint32_t *inputs);

/*
** v2: expand a tree reference of round r back to its input indices.
*/
static void v2_expand(const uint32_t *ent, unsigned r, uint32_t bucket, uint32_t slot,
        uint32_t *out, unsigned *pos)
{
    uint32_t t = ent[v2_ent_off[r] / 4 + (((size_t)(bucket / tile) * V2_NSLOTS + slot) * tile + bucket % tile) * v2_e_words[r] +
            v2_e_words[r] - 1];     // the tree word ends each record
    if (!r)
      {
        out[(*pos)++] = t;
        return;
      }
    v2_expand(ent, r - 1, t >> 20, (t >> 10) & 0x3ff, out, pos);
    v2_expand(ent, r - 1, t >> 20, t & 0x3ff, out, pos);
}

/*
** Check the solutions of a finished run, and report them.
*/
static void collect(Slot *s)
{
    [s.cb waitUntilCompleted];
    if (s.cb.status == MTLCommandBufferStatusError)
        fatal("GPU run failed: %s\n", s.cb.error.localizedDescription.UTF8String);
    total_runs++;
    if (use_v2)
      {
        cands_t *c = (cands_t *)s.cands.contents;
        uint32_t n = c->n;
        if (n > V2_MAX_CANDS)
          {
            total_dropped += n - V2_MAX_CANDS;
            n = V2_MAX_CANDS;
          }
        const uint32_t *trees = (const uint32_t *)s.ent.contents;
        for (uint32_t i = 0; i < n; i++)
          {
            uint32_t inputs[1 << PARAM_K];
            unsigned pos = 0;
            uint32_t b = c->cand[i] >> 20;
            v2_expand(trees, PARAM_K - 1, b, (c->cand[i] >> 10) & 0x3ff, inputs, &pos);
            v2_expand(trees, PARAM_K - 1, b, c->cand[i] & 0x3ff, inputs, &pos);
            report(s, inputs);
          }
        return;
      }
    sols_t *sols = (sols_t *)s.sols.contents;
    uint32_t nr = sols->nr;
    if (nr > MAX_SOLS)
      {
        total_dropped += nr - MAX_SOLS;
        nr = MAX_SOLS;
      }
    for (uint32_t i = 0; i < nr; i++)
        report(s, sols->values[i]);
}

/*
** Verify one candidate solution (duplicates, canonical order) and output it.
*/
static void report(Slot *s, uint32_t *inputs)
{
      {
        if (!verify_sol(inputs))
            return;
        total_sols++;
        uint8_t buffer[ZCASH_BLOCK_HEADER_LEN + ZCASH_SOLSIZE_LEN + ZCASH_SOL_LEN];
        memcpy(buffer, s->header, ZCASH_BLOCK_HEADER_LEN);
        memcpy(buffer + ZCASH_BLOCK_HEADER_LEN, "\xfd\x40\x05", ZCASH_SOLSIZE_LEN);
        store_encoded_sol(buffer + ZCASH_BLOCK_HEADER_LEN + ZCASH_SOLSIZE_LEN, inputs,
                1 << PARAM_K);
        uint8_t *sol = buffer + ZCASH_BLOCK_HEADER_LEN + ZCASH_SOLSIZE_LEN;
        if (verify_out)
          {
            fprintf(verify_out, "%s ", s_hexdump(s->header, 108));
            fprintf(verify_out, "%s ", s_hexdump(s->header + 108, ZCASH_NONCE_LEN));
            fprintf(verify_out, "%s\n", s_hexdump(sol, ZCASH_SOL_LEN));
          }
        if (!mining)
            return;
        uint8_t hash0[SHA256_DIGEST_SIZE], hash1[SHA256_DIGEST_SIZE];
        Sha256_Onestep(buffer, sizeof (buffer), hash0);
        Sha256_Onestep(hash0, sizeof (hash0), hash1);
        if (cmp_target_256(s->target, hash1) < 0)
            return;     // above the target
        total_shares++;
        const uint8_t *p = s->header + ZCASH_BLOCK_OFFSET_NTIME;
        printf("sol: %s %02x%02x%02x%02x ", s->job_id, p[0], p[1], p[2], p[3]);
        printf("%s ", s_hexdump(s->header + ZCASH_BLOCK_HEADER_LEN - ZCASH_NONCE_LEN +
                    s->fixed_nonce_bytes, ZCASH_NONCE_LEN - s->fixed_nonce_bytes));
        printf("%s%s\n", ZCASH_SOLSIZE_HEX, s_hexdump(sol, ZCASH_SOL_LEN));
        fflush(stdout);
      }
}

/*
** The current header; each run gets the next nonce.
*/
static uint8_t cur_header[ZCASH_BLOCK_HEADER_LEN];
static char cur_job_id[256];
static uint8_t cur_target[SHA256_TARGET_LEN];
static size_t cur_fixed_nonce_bytes;

static void next_nonce(Slot *s)
{
    uint8_t *nonce = cur_header + ZCASH_BLOCK_HEADER_LEN - ZCASH_NONCE_LEN;
    // increment nonce bytes 17-19 (byte 20 and above must stay zero), like SILENTARMY
    uint32_t ctr;
    memcpy(&ctr, nonce + 17, 4);
    ctr++;
    memcpy(nonce + 17, &ctr, 4);
    memset(nonce + 20, 0, 4);
    memcpy(s->header, cur_header, ZCASH_BLOCK_HEADER_LEN);
    memcpy(s->job_id, cur_job_id, sizeof (cur_job_id));
    memcpy(s->target, cur_target, SHA256_TARGET_LEN);
    s->fixed_nonce_bytes = cur_fixed_nonce_bytes;
}

static void set_blocking_mode(int fd, int block)
{
    int f;
    if (-1 == (f = fcntl(fd, F_GETFL)))
        fatal("fcntl F_GETFL: %s\n", strerror(errno));
    if (-1 == fcntl(fd, F_SETFL, block ? (f & ~O_NONBLOCK) : (f | O_NONBLOCK)))
        fatal("fcntl F_SETFL: %s\n", strerror(errno));
}

static void *last_newline(const char *s, size_t n)
{
    const char *p = s + n;
    while (n--)
        if (*--p == '\n')
            return (void *)p;
    return NULL;
}

/*
** Read the newest complete line from stdin (older queued lines are skipped).
** Return 1 iff a line was read.
*/
static int read_last_line(char *buf, size_t len, int block)
{
    size_t pos = 0;
    set_blocking_mode(0, block);
    while (1)
      {
        ssize_t n = read(0, buf + pos, len - pos - 1);
        if (n == -1 && errno == EINTR)
            continue;
        else if (n == -1 && (errno == EAGAIN || errno == EWOULDBLOCK))
          {
            if (!pos)
                return 0;
            set_blocking_mode(0, 1);    // a partial line: read the rest of it
            continue;
          }
        else if (n == -1)
            fatal("read stdin: %s\n", strerror(errno));
        else if (!n)
            fatal("EOF on stdin\n");
        pos += n;
        if (buf[pos - 1] == '\n')
            break;
        if (pos >= len - 1)
          {
            // keep only what follows the last complete line
            char *nl = last_newline(buf, pos);
            size_t keep = nl ? pos - (nl + 1 - buf) : 0;
            memmove(buf, buf + pos - keep, keep);
            pos = keep;
          }
      }
    char *start = last_newline(buf, pos - 1);
    if (start)
      {
        start++;
        pos -= start - buf;
        memmove(buf, start, pos);
      }
    buf[pos - 1] = 0;
    return 1;
}

/*
** Parse "<target> <job_id> <header> <nonce_leftpart>" into the current job, and randomize the
** rest of the nonce (except the last N_ZERO_BYTES bytes, which must be zero).
*/
static void parse_job(const char *str)
{
    uint32_t str_i = 0, i;
    for (i = 0; i < SHA256_TARGET_LEN; i++, str_i += 2)
        cur_target[i] = hex2val(str, str_i) * 16 + hex2val(str, str_i + 1);
    if (str[str_i++] != ' ')
        fatal("bad job line (target)\n");
    for (i = 0; i < sizeof (cur_job_id) - 1 && str[str_i] && str[str_i] != ' '; i++, str_i++)
        cur_job_id[i] = str[str_i];
    cur_job_id[i] = 0;
    if (str[str_i++] != ' ')
        fatal("bad job line (job id)\n");
    for (i = 0; i < ZCASH_BLOCK_HEADER_LEN && str[str_i] && str[str_i] != ' '; i++, str_i += 2)
        cur_header[i] = hex2val(str, str_i) * 16 + hex2val(str, str_i + 1);
    if (str[str_i++] != ' ')
        fatal("bad job line (header)\n");
    cur_fixed_nonce_bytes = 0;
    while (i < ZCASH_BLOCK_HEADER_LEN && str[str_i])
      {
        cur_header[i] = hex2val(str, str_i) * 16 + hex2val(str, str_i + 1);
        i++;
        str_i += 2;
        cur_fixed_nonce_bytes++;
      }
    if (i > ZCASH_BLOCK_HEADER_LEN - N_ZERO_BYTES)
        fatal("too many fixed nonce bytes\n");
    randomize(cur_header + i, ZCASH_BLOCK_HEADER_LEN - N_ZERO_BYTES - i);
    memset(cur_header + ZCASH_BLOCK_HEADER_LEN - N_ZERO_BYTES, 0, N_ZERO_BYTES);
}

static char line[1 << 20];

static void mining_mode(NSArray<Slot *> *slots)
{
    puts("METAL-EQ mining mode ready");
    fflush(stdout);
    read_last_line(line, sizeof (line), 1);
    parse_job(line);
    for (Slot *s in slots)
      {
        next_nonce(s);
        submit(s);
      }
    uint64_t t0 = 0;
    for (NSUInteger i = 0;; i = (i + 1) % slots.count)
      {
        @autoreleasepool
          {
            Slot *s = slots[i];
            collect(s);
            if (read_last_line(line, sizeof (line), 0))
                parse_job(line);
            next_nonce(s);
            submit(s);
            uint64_t t1 = now_us();
            if (t1 > t0 + 500000)
              {
                t0 = t1;
                printf("status: %" PRIu64 " %" PRIu64 "\n", total_sols, total_shares);
                fflush(stdout);
              }
          }
      }
}

/*
** --overlap benchmark: slot i runs its rounds while slot i+1's round 0 is computed alongside.
*/
static void bench_overlapped(NSArray<Slot *> *slots, uint64_t nonces)
{
    unsigned K = (unsigned)slots.count;
    memset(cur_header, 0, sizeof (cur_header));
    randomize(cur_header + 108, 17);
    memset(cur_target, 0xff, sizeof (cur_target));
    uint64_t t0 = now_us();
    // the first run's round 0 on its own
    Slot *first = slots[0];
    next_nonce(first);
    set_blake(first);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    v2_reset(cb, first);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    v2_encode_pass(enc, first, 0);
    [enc endEncoding];
    [cb commit];
    for (uint64_t i = 0; i < nonces; i++)
      {
        @autoreleasepool
          {
            Slot *s = slots[i % K];
            Slot *t = (i + 1 < nonces) ? slots[(i + 1) % K] : nil;
            if (t && i + 1 >= K)
                collect(t);             // t's previous run (i + 1 - K) is done: free its buffers
            if (t)
              {
                next_nonce(t);
                set_blake(t);
              }
            submit_overlapped(s, t);
          }
      }
    for (uint64_t i = nonces > K ? nonces - K : 0; i < nonces; i++)   // runs not collected in the loop
        collect(slots[i % K]);
    double secs = (now_us() - t0) / 1e6;
    printf("%" PRIu64 " runs, %" PRIu64 " solutions in %.2f s: %.1f Sol/s, %.3f sols/run, "
            "%" PRIu64 " dropped\n", total_runs, total_sols, secs, total_sols / secs,
            (double)total_sols / total_runs, total_dropped);
}

static void bench_mode(NSArray<Slot *> *slots, uint64_t nonces)
{
    // a fixed header: zero prefix, random nonce bytes 0..16, counter at 17..19, zeros after
    memset(cur_header, 0, sizeof (cur_header));
    randomize(cur_header + 108, 17);
    memset(cur_target, 0xff, sizeof (cur_target));
    uint64_t submitted = 0, done = 0;
    uint64_t t0 = now_us();
    for (Slot *s in slots)
        if (submitted < nonces)
          {
            next_nonce(s);
            submit(s);
            submitted++;
          }
    for (NSUInteger i = 0; done < nonces; i = (i + 1) % slots.count)
      {
        @autoreleasepool
          {
            Slot *s = slots[i];
            collect(s);
            done++;
            if (submitted < nonces)
              {
                next_nonce(s);
                submit(s);
                submitted++;
              }
          }
      }
    double secs = (now_us() - t0) / 1e6;
    if (cpu_sg)
      {
        double job_ms, waited_ms;
        uint64_t jobs;
        cpuhash_stats(&job_ms, &waited_ms, &jobs);
        printf("cpu round 0: %" PRIu64 " jobs, %.2f ms each, host waited %.2f ms per job\n",
               jobs, job_ms, waited_ms);
      }
    printf("%" PRIu64 " runs, %" PRIu64 " solutions in %.3f s: %.1f Sol/s, %.3f sols/run, "
            "%" PRIu64 " dropped\n", total_runs, total_sols, secs, total_sols / secs,
            (double)total_sols / total_runs, total_dropped);
}

/*
** Time each GPU pass of a run separately (each pass in its own command buffer).
*/
static void profile_mode(Slot *s, unsigned runs)
{
    double t_pass[PARAM_K + 1] = { 0 };
    memset(cur_header, 0, sizeof (cur_header));
    randomize(cur_header + 108, 17);
    if (use_v2)
      {
        for (unsigned run = 0; run < runs; run++)
          {
            @autoreleasepool
              {
                next_nonce(s);
                set_blake(s);
                for (unsigned pass = 0; pass <= PARAM_K; pass++)
                  {
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    if (!pass)
                        v2_reset(cb, s);
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    v2_encode_pass(enc, s, pass);
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    t_pass[pass] += cb.GPUEndTime - cb.GPUStartTime;
                  }
                s.cb = nil;
                total_runs++;
                cands_t *c = (cands_t *)s.cands.contents;
                const uint32_t *nsl = (const uint32_t *)s.nsl.contents;
                if (verbose && run == 0)
                    for (unsigned r = 0; r < PARAM_K; r++)
                      {
                        uint64_t sum = 0, max = 0;
                        for (unsigned b = 0; b < V2_NBUCKETS; b++)
                          {
                            sum += nsl[r * V2_NBUCKETS + b];
                            if (nsl[r * V2_NBUCKETS + b] > max)
                                max = nsl[r * V2_NBUCKETS + b];
                          }
                        fprintf(stderr, "round %u: %llu entries, fullest bucket %llu\n", r,
                                (unsigned long long)sum, (unsigned long long)max);
                      }
                if (verbose)
                    fprintf(stderr, "run %u: %u candidates\n", run, c->n);
              }
          }
        double total = 0;
        for (unsigned pass = 0; pass <= PARAM_K; pass++)
            total += t_pass[pass];
        for (unsigned pass = 0; pass <= PARAM_K; pass++)
            printf("%-8s %6.2f ms  %4.1f%%\n", pass < PARAM_K ?
                    [NSString stringWithFormat:@"round%u", pass].UTF8String : "final",
                    1e3 * t_pass[pass] / runs, 100 * t_pass[pass] / total);
        printf("total    %6.2f ms per run of GPU time\n", 1e3 * total / runs);
        return;
      }
    id<MTLBuffer> ht[2] = { s.ht0, s.ht1 };
    id<MTLBuffer> cnt[2] = { s.cnt0, s.cnt1 };
    for (unsigned run = 0; run < runs; run++)
      {
        @autoreleasepool
          {
            next_nonce(s);
            blake2b_state_t blake;
            zcash_blake2b_init(&blake, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
            zcash_blake2b_update(&blake, s->header, 128, 0);
            memcpy(s.blake.contents, blake.h, sizeof (blake.h));
            for (unsigned pass = 0; pass <= PARAM_K; pass++)
              {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                if (pass < PARAM_K)
                  {
                    [enc setComputePipelineState:p_init];
                    [enc setBuffer:cnt[pass % 2] offset:0 atIndex:0];
                    [enc dispatchThreads:MTLSizeMake(NR_ROWS / ROWS_PER_UINT, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc setComputePipelineState:p_round[pass]];
                    if (!pass)
                      {
                        [enc setBuffer:s.blake offset:0 atIndex:0];
                        [enc setBuffer:ht[0] offset:0 atIndex:1];
                        [enc setBuffer:cnt[0] offset:0 atIndex:2];
                        [enc dispatchThreads:MTLSizeMake(round0_threads, 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                      }
                    else
                      {
                        [enc setBuffer:ht[(pass - 1) % 2] offset:0 atIndex:0];
                        [enc setBuffer:ht[pass % 2] offset:0 atIndex:1];
                        [enc setBuffer:cnt[(pass - 1) % 2] offset:0 atIndex:2];
                        [enc setBuffer:cnt[pass % 2] offset:0 atIndex:3];
                        if (pass == PARAM_K - 1)
                            [enc setBuffer:s.sols offset:0 atIndex:4];
                        [enc dispatchThreads:MTLSizeMake(NR_ROWS, 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
                      }
                  }
                else
                  {
                    [enc setComputePipelineState:p_sols];
                    [enc setBuffer:ht[0] offset:0 atIndex:0];
                    [enc setBuffer:ht[1] offset:0 atIndex:1];
                    [enc setBuffer:s.sols offset:0 atIndex:2];
                    [enc setBuffer:cnt[(PARAM_K - 1) % 2] offset:0 atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(NR_ROWS, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                  }
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                t_pass[pass] += cb.GPUEndTime - cb.GPUStartTime;
              }
          }
      }
    double total = 0;
    for (unsigned pass = 0; pass <= PARAM_K; pass++)
        total += t_pass[pass];
    for (unsigned pass = 0; pass <= PARAM_K; pass++)
        printf("%-8s %6.2f ms  %4.1f%%\n", pass < PARAM_K ?
                [NSString stringWithFormat:@"round%u", pass].UTF8String : "sols",
                1e3 * t_pass[pass] / runs, 100 * t_pass[pass] / total);
    printf("total    %6.2f ms per run of GPU time\n", 1e3 * total / runs);
}

/*
** v2: time each pass at full GPU clocks. Builds valid inputs with one full run, then repeats each
** pass K times back-to-back in one command buffer (resetting its output counters each time).
*/
static void passbench_mode(Slot *s, unsigned reps)
{
    const size_t nsl_round = (size_t)V2_NBUCKETS * 4;
    memset(cur_header, 0, sizeof (cur_header));
    randomize(cur_header + 108, 17);
    next_nonce(s);
    set_blake(s);
    double total = 0;
    for (unsigned pass = 0; pass <= PARAM_K; pass++)
      {
        @autoreleasepool
          {
            // fresh valid inputs for this pass: a full run
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            v2_reset(cb, s);
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (unsigned p = 0; p <= PARAM_K; p++)
                v2_encode_pass(enc, s, p);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            // now repeat just this pass
            cb = [queue commandBuffer];
            for (unsigned k = 0; k < reps; k++)
              {
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                if (pass < PARAM_K)
                    [blit fillBuffer:s.nsl range:NSMakeRange(pass * nsl_round, nsl_round) value:0];
                else
                    [blit fillBuffer:s.cands range:NSMakeRange(0, 4) value:0];
                [blit endEncoding];
                enc = [cb computeCommandEncoder];
                v2_encode_pass(enc, s, pass);
                [enc endEncoding];
              }
            [cb commit];
            [cb waitUntilCompleted];
            double t = (cb.GPUEndTime - cb.GPUStartTime) / reps;
            total += t;
            printf("%-8s %6.3f ms\n", pass < PARAM_K ?
                    [NSString stringWithFormat:@"round%u", pass].UTF8String : "final", 1e3 * t);
          }
      }
    printf("sum      %6.3f ms per run (passes back-to-back, full clocks)\n", 1e3 * total);
}

/*
** Experiment: does the GPU overlap two independent dispatches in a concurrent compute encoder?
** Times round 1 of run A alone, round 0 of run B alone, and both together without a barrier.
*/
static void overlap_test(Slot *A, Slot *B, unsigned reps)
{
    const size_t nsl_round = (size_t)V2_NBUCKETS * 4;
    memset(cur_header, 0, sizeof (cur_header));
    randomize(cur_header + 108, 17);
    next_nonce(A);
    set_blake(A);
    next_nonce(B);
    set_blake(B);
    // valid inputs for A's round 1: run A fully once
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    v2_reset(cb, A);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (unsigned p = 0; p <= PARAM_K; p++)
        v2_encode_pass(enc, A, p);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const char *what[3] = { "round 1 (A) alone", "round 0 (B) alone", "both, concurrent" };
    for (int mode = 0; mode < 3; mode++)
      {
        @autoreleasepool
          {
            cb = [queue commandBuffer];
            for (unsigned k = 0; k < reps; k++)
              {
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                [blit fillBuffer:A.nsl range:NSMakeRange(1 * nsl_round, nsl_round) value:0];
                [blit fillBuffer:B.nsl range:NSMakeRange(0, nsl_round) value:0];
                [blit endEncoding];
                enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                if (mode != 1)
                    v2_encode_pass(enc, A, 1);
                if (mode != 0)
                    v2_encode_pass(enc, B, 0);
                [enc endEncoding];
              }
            [cb commit];
            [cb waitUntilCompleted];
            printf("%-20s %6.3f ms\n", what[mode], 1e3 * (cb.GPUEndTime - cb.GPUStartTime) / reps);
          }
      }
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [--mining | --nonces N [--verify-out FILE] | --profile RUNS] [--bpt N] [--cpu-sg N]\n"
            "          [--cpu-threads N] [--lean 0|5|6] [--lean-from R] [--inflight K] [--v1] [-v]\n", prog);
    exit(1);
}

int main(int argc, char **argv)
{
    @autoreleasepool
      {
        uint64_t nonces = 0;
        unsigned profile_runs = 0, passbench_reps = 0, overlap_reps = 0;
        static struct option opts[] = {
            {"mining", no_argument, 0, 'm'},
            {"nonces", required_argument, 0, 'n'},
            {"verify-out", required_argument, 0, 'o'},
            {"inflight", required_argument, 0, 'k'},
            {"tg", required_argument, 0, 't'},
            {"coll", required_argument, 0, 'c'},
            {"round0-threads", required_argument, 0, 'r'},
            {"use", required_argument, 0, 'u'},     // accepted for SILENTARMY compatibility
            {"profile", required_argument, 0, 'p'},
            {"diag", required_argument, 0, 'd'},
            {"tile", required_argument, 0, 'y'},
            {"bpt", required_argument, 0, 'b'},
            {"v2", no_argument, 0, '2'},
            {"v1", no_argument, 0, '1'},
            {"tg2", required_argument, 0, 'G'},
            {"tg0", required_argument, 0, 'Z'},
            {"pairs", required_argument, 0, 'P'},
            {"passbench", required_argument, 0, 'B'},
            {"twoq", no_argument, 0, 'Q'},
            {"lean", required_argument, 0, 'L'},
            {"lean-from", required_argument, 0, 'F'},
            {"overlaptest", required_argument, 0, 'O'},
            {"overlap", no_argument, 0, 'V'},
            {"overlap-chunks", required_argument, 0, 'H'},
            {"cpu-sg", required_argument, 0, 'C'},
            {"cpu-threads", required_argument, 0, 'T'},
            {0, 0, 0, 0},
        };
        int c;
        while ((c = getopt_long(argc, argv, "v", opts, NULL)) != -1)
            switch (c)
              {
              case 'm': mining = 1; break;
              case 'n': nonces = strtoull(optarg, NULL, 0); break;
              case 'o':
                if (!(verify_out = fopen(optarg, "w")))
                    fatal("%s: %s\n", optarg, strerror(errno));
                break;
              case 'k': inflight = (unsigned)strtoul(optarg, NULL, 0); break;
              case 't': tg_size = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'c': coll_per_th = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'r': round0_threads = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'u': break;
              case '2': use_v2 = 1; break;
              case '1': use_v2 = 0; break;
              case 'G': v2_tg = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'Z': v2_tg0 = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'P': v2_max_pairs = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'Q': two_queues = 1; break;
              case 'L': lean = (int)strtol(optarg, NULL, 0); break;
              case 'F': lean_from = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'V': overlap = 1; break;
              case 'C': cpu_sg = (int)strtol(optarg, NULL, 0); break;
              case 'T': cpu_threads = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'H': overlap_chunks = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'O': overlap_reps = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'B': passbench_reps = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'd': diag_mode = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'y': tile = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'b': bpt = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'p': profile_runs = (unsigned)strtoul(optarg, NULL, 0); break;
              case 'v': verbose++; break;
              default: usage(argv[0]);
              }
        if (!mining && !nonces && !profile_runs && !passbench_reps && !overlap_reps)
            usage(argv[0]);
        if (!inflight || inflight > 32 || !tg_size || NR_ROWS % tg_size ||
                !round0_threads || NR_INPUTS % round0_threads)
            fatal("bad --inflight/--tg/--round0-threads value\n");
        if (!bpt || (bpt & (bpt - 1)) || V2_NBUCKETS % bpt)
            fatal("--bpt must be a power of 2 dividing %u\n", V2_NBUCKETS);
        if (cpu_sg < 0)
            cpu_sg = use_v2 && !overlap && !two_queues && !profile_runs && !passbench_reps && !overlap_reps;
        if (cpu_sg && (!use_v2 || cpu_sg > 8 || v2_tg0 != 256 || overlap || two_queues || !cpu_threads ||
                       profile_runs || passbench_reps || overlap_reps))
            fatal("--cpu-sg needs v2, at most 8, --tg0 256, --cpu-threads > 0, and no "
                  "--overlap/--twoq/--profile/--passbench/--overlaptest\n");
        if (cpu_sg)
            cpuhash_init(cpu_threads);
        init_metal();
        NSMutableArray<Slot *> *slots = [NSMutableArray array];
        for (unsigned i = 0; i < inflight; i++)
            [slots addObject:new_slot()];
        if (overlap_reps)
            overlap_test(slots[0], slots[1], overlap_reps);
        else if (passbench_reps)
            passbench_mode(slots[0], passbench_reps);
        else if (profile_runs)
            profile_mode(slots[0], profile_runs);
        else if (mining)
            mining_mode(slots);
        else if (overlap && use_v2)
            bench_overlapped(slots, nonces);
        else
            bench_mode(slots, nonces);
        if (verify_out)
            fclose(verify_out);
      }
    return 0;
}
