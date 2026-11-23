// --cpu-sg: the CPU computes part of round 0's BLAKE2b hashes; the GPU loads them instead of
// computing them. A small pool of worker threads takes chunks of the job with an atomic counter
// (the cores run at different clocks, so static slices would leave the fast ones idle).
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>
#include "blake_neon.h"
#include "cpuhash.h"

#define CHUNK 768                       // hashes per claim (a multiple of 3 for hash_hybrid3)

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_start = PTHREAD_COND_INITIALIZER, cv_done = PTHREAD_COND_INITIALIZER;
static uint64_t gen;                    // job number, bumped to start the workers
static unsigned nworkers, busy;         // workers, and those still on the current job
static atomic_uint next_j;              // next unclaimed hash of the current job
static struct { BnMid md; unsigned per_tg; uint32_t total; uint64_t *out; } job;
static uint64_t t_start, t_jobs_us, t_waited_us, njobs;   // stats: job time, time callers blocked

static uint64_t now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000 + t.tv_nsec / 1000;
}

// hash j of a job is input (j / per_tg) * 256 + j % per_tg: the first per_tg inputs of each
// 256-input round-0 threadgroup
static inline uint32_t input_of(uint32_t j)
{
    return (j / job.per_tg) * 256 + j % job.per_tg;
}

static void *worker(void *arg)
{
    (void)arg;
    uint64_t seen = 0;
    for (;;)
      {
        pthread_mutex_lock(&mu);
        while (gen == seen)
            pthread_cond_wait(&cv_start, &mu);
        seen = gen;
        pthread_mutex_unlock(&mu);
        for (;;)
          {
            uint32_t j = atomic_fetch_add_explicit(&next_j, CHUNK, memory_order_relaxed);
            if (j >= job.total)
                break;
            uint32_t end = j + CHUNK < job.total ? j + CHUNK : job.total;
            uint64_t *o = job.out + (size_t)j * 7;
            for (; j + 3 <= end; j += 3, o += 21)
                hash_hybrid3(&job.md, input_of(j), input_of(j + 1), input_of(j + 2), o, o + 7, o + 14);
            for (; j < end; j++, o += 7)
                hash_scalar(&job.md, input_of(j), o);
          }
        pthread_mutex_lock(&mu);
        if (--busy == 0)
          {
            t_jobs_us += now_us() - t_start;
            njobs++;
            pthread_cond_signal(&cv_done);
          }
        pthread_mutex_unlock(&mu);
      }
    return NULL;
}

void cpuhash_init(unsigned threads)
{
    nworkers = threads;
    for (unsigned t = 0; t < threads; t++)
      {
        pthread_t th;
        pthread_create(&th, NULL, worker, NULL);
        pthread_detach(th);
      }
}

void cpuhash_wait(void)
{
    uint64_t t0 = now_us();
    pthread_mutex_lock(&mu);
    while (busy)
        pthread_cond_wait(&cv_done, &mu);
    pthread_mutex_unlock(&mu);
    t_waited_us += now_us() - t0;
}

void cpuhash_stats(double *job_ms, double *waited_ms, uint64_t *jobs)
{
    *jobs = njobs;
    *job_ms = njobs ? t_jobs_us / 1e3 / njobs : 0;
    *waited_ms = njobs ? t_waited_us / 1e3 / njobs : 0;
}

void cpuhash_start(const uint64_t midstate[8], const uint8_t last16[16], unsigned per_tg,
                   unsigned ntg, uint64_t *out)
{
    cpuhash_wait();                     // one job at a time
    memcpy(job.md.h, midstate, sizeof job.md.h);
    memcpy(&job.md.m0, last16, 8);
    uint32_t lo;
    memcpy(&lo, last16 + 8, 4);
    job.md.m1lo = lo;
    job.per_tg = per_tg;
    job.total = per_tg * ntg;
    job.out = out;
    atomic_store_explicit(&next_j, 0, memory_order_relaxed);
    pthread_mutex_lock(&mu);
    t_start = now_us();
    busy = nworkers;
    gen++;
    pthread_cond_broadcast(&cv_start);
    pthread_mutex_unlock(&mu);
}

void cpuhash_run(const uint64_t midstate[8], const uint8_t last16[16], unsigned per_tg,
                 unsigned ntg, uint64_t *out)
{
    cpuhash_start(midstate, last16, per_tg, ntg, out);
    cpuhash_wait();
}
