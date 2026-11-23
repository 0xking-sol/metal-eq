#pragma once
#include <stdint.h>

// Start the worker threads (once).
void cpuhash_init(unsigned threads);

// Compute per_tg * ntg round-0 BLAKE2b outputs, 7 words each, into out: for each 256-input
// round-0 threadgroup t < ntg, inputs t * 256 + [0, per_tg). midstate = the state after the
// header's first 128 bytes; last16 = header bytes 128..139 + 4 zero bytes. Blocks until done.
void cpuhash_run(const uint64_t midstate[8], const uint8_t last16[16], unsigned per_tg,
                 unsigned ntg, uint64_t *out);

// The same in the background: cpuhash_start returns at once (after any previous job is done),
// cpuhash_wait blocks until the job is done.
void cpuhash_start(const uint64_t midstate[8], const uint8_t last16[16], unsigned per_tg,
                   unsigned ntg, uint64_t *out);
void cpuhash_wait(void);

// Per job: the average time the workers took, and the average time callers blocked waiting.
void cpuhash_stats(double *job_ms, double *waited_ms, uint64_t *jobs);
