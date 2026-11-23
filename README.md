# metal-eq

Equihash (200,9) solver for Apple Silicon GPUs, written in Metal. Built to mine Zcash Crosslink on Macs, and it comes with a small zebrad patch so the node's internal miner can use it.

As far as i know its the first Equihash 200,9 solver for Metal. It started as a port of [SILENTARMY](https://github.com/mbevand/silentarmy) (Marc Bevand, MIT) and then moved to its own bucketed design.

quick start: `./build.sh`, then `./tune.sh` to find the best flags for your Mac, then see "mining with zebrad" below.

## speed

measured while mining (zebrad's own solution counter, 10 min windows), with the flags from the next section:

| chip | GPU | flags | Sol/s |
|---|---|---|---|
| M5 Pro | 20-core | `--cpu-sg 2` | ~475 |
| M3 Max | 40-core | `--bpt 4 --lean 0 --cpu-sg 0` | ~580-620 (depends on power and heat, see tips) |

for reference, silentarmy through Apple's OpenCL did ~136 Sol/s on the same M5 Pro, with 16 processes and 13 GB of GPU memory. metal-eq is one process and well under 2 GB. zebrad's own CPU miner (what the official Mac build mines with) tops out around 38 Sol/s on that chip with all 16 threads busy, ~7 with one thread.

every solution gets checked again inside zebrad (Equihash validity, then the real block hash against the target) before it can become a block. no invalid solutions so far.

## safety

- source only. no prebuilt binaries and nothing gets downloaded at build or run time. you build it yourself with Apple's compiler (`build.sh` is 3 commands), so you can read it all first
- no network access. metal-eq reads jobs on stdin and writes solutions on stdout, and the binary only links Apple's system frameworks (check with `otool -L metal-eq` and `nm -u metal-eq`)
- it never sees your wallet, keys or RPC. all zebrad hands it is the block header to mine (public data), and all it sends back is a nonce and a solution. it reads no files or env vars, and the only file it writes is the one you name with `--verify-out`
- zebrad re-checks every solution before it can become a block, so even a broken solver cant get you a bad block. worst case is wasted GPU time
- no sudo, no installer, no launch agents. it runs as your user
- the zebrad patch only touches the miner (3 files, readable diff in `zebrad/`), no consensus, wallet or networking code. GPU mining stays off unless you set `SILENTARMY_SOLVERS`
- no dev fee and no telemetry
- hardware wise its an ordinary Metal compute job like a game or a local AI model: no overclocking, drivers or kernel extensions, and macOS throttles it if things get hot. laptops get warm and loud, see tips
- check its output yourself: `./metal-eq --nonces 50 --verify-out f.txt && python3 tools/eqcheck.py f.txt` (independent checker, only python's standard library)

## settings per machine

the best flags depend on the chip. quickest way to find yours: plug in and run `./tune.sh` (a minute or two). it benchmarks the main combinations back to back, re-checks the top 3 and prints the winner plus a wrapper line to mine with it. if a metal-eq is already mining it gets paused during the tune and resumed after.

these are the two setups we actually mine with.

**M5 Pro (20-core GPU): `--cpu-sg 2`**
- leave `--bpt` at 1 (the default). an empty round only costs ~0.03 ms to launch on the M5 Pro, so grouping buckets just takes away parallelism: `--bpt 2` was -2.7%, `--bpt 4` -4.4%
- `--cpu-sg 2` (the CPU hashes 2/8 of round 0) was +2.3% over the default 1 in a bench and ~+0.8% live, once zebrad's own CPU solver was turned off (`internal_miner_threads = 0`, see below)

**M3 Max (40-core GPU): `--bpt 4 --lean 0 --cpu-sg 0`**
- `--bpt 4`: one threadgroup works through 4 buckets in turn. the M3 pays ~0.075 ms per round just to launch 4096 x 256 threads (0.6-0.7 ms of a ~3.9 ms run). a single run profiles about the same either way, the gain shows up with 3 runs in flight where 4x fewer threadgroups overlap better: +7.6% to +11% in a bench, +6-8% live
- `--lean 0`: `--bpt` only changes the plain v2 kernels, and by default rounds 5-8 run the lean ones (always one bucket per threadgroup). with `--lean 0` it covers every round. `--bpt 4` gained +6.1% with the lean kernels on, more with them off
- `--cpu-sg 0`. in a bench `--cpu-sg 2` was +4% for ~16 W more at the wall, but live it was slower (~572 vs ~620 Sol/s): the CPU share falls behind and the GPU waits and clocks down. that was with the power supply already maxed out, we havent measured 2 with power to spare

other chips arent tested, and with one chip each we cant tell if the bpt difference is Pro vs Max or M3 vs M5, so let `./tune.sh` decide. by hand: try `--bpt 1/2/4` (with `--lean 0` and with the default 6) and `--cpu-sg 0/1/2` with `./metal-eq --nonces 300` and compare runs/s. pause your miner while you do (`pkill -STOP metal-eq`, then `pkill -CONT metal-eq`), otherwise both share the GPU. if you tune a chip we dont list, a PR with the numbers is very welcome

## versions

how it got here. from v3 on each step was an A/B against the previous version in one session (runs/s drift +-10% between sessions), on the M5 Pro except bpt (M3 Max):

| version | change | result |
|---|---|---|
| v1 | straight Metal port of silentarmy, 2 runs in flight | ~150 Sol/s on 1.6 GB |
| v2 | own bucketed design: 4096 buckets, one 256-thread group per bucket in on-chip memory, appends instead of scattered writes, and pairs that share a subtree get pruned in every round | ~205 Sol/s |
| v3 | slimmer on-chip footprint (640 slots per bucket) and entries packed at each round's size | ~340-410 Sol/s |
| v3.1 | 1024-pair buffer. 512 was the mean, so half the buckets fell back to the slow path | ~420 Sol/s |
| v4 | the tree word lives inside each record, so its one write and one read per entry | +1.4% |
| v4.1 | "lean" kernels for rounds 5 to 8: counting sort, binary-searched pairs, data cached on chip | +2% |
| v4.2 | split-record packing: whole data words + tree word, the last 16 bits or less in a small side array (172 -> 162 bytes per entry path, all accesses aligned) | +5.2%, ~460 Sol/s |
| v4.3 | the CPU computes 1/8 of round 0 with NEON BLAKE2b (`--cpu-sg`) while the GPU does the rest | +3.8%, ~475 Sol/s |
| bpt | `--bpt N`: one threadgroup per N buckets in the plain v2 kernels (rounds 1-4 and the final round, all rounds with `--lean 0`). default 1 = v4.3 | +7.6% to +11% bench, +6-8% live on the M3 Max (`--bpt 4 --lean 0`), worse on the M5 Pro |

the M5 Pro still runs the v4.3 build (with `--cpu-sg 2`), the M3 Max runs this code with `--bpt 4`. this repo is the bpt version: at the default `--bpt 1` it finds the same solutions as v4.3 on the same nonces (A/B on the M5 Pro: 2846 vs 2847, 2845 identical) at the same speed (269 vs 264 runs/s).

things that didnt help (A/B): computing the next run's round 0 inside rounds 1-8 (-35%, register pressure), padded bucket counters (-6%), a slot-major record layout (`--tile`, -17%), 2 processes or 4 runs in flight, other threadgroup sizes.

every version was checked before it went live with zebrad's own Equihash verifier, on bench output and over the mining protocol. from the bpt version on also with `tools/eqcheck.py`, an independent checker.

## how it works

- Wagner's algorithm with buckets, same idea as Tromp's and djezo's CUDA solvers: 4096 buckets, one 256-thread group per bucket working in on-chip memory, appends instead of scattered writes
- prunes pairs that share a subtree in every round, so the last round only has a handful of candidates
- records are packed per round (data words + tree word in one record, leftover bits in small side arrays), since the later rounds are memory bound
- "lean" kernels for rounds 5 to 8 (counting sort, binary-searched pairs, data cached on chip)
- the CPU computes part of round 0's BLAKE2b with NEON (`--cpu-sg`) while the GPU does the rest
- optionally one threadgroup handles several buckets (`--bpt`), which pays off where launching a round is expensive
- 3 Equihash runs in flight so the GPU never waits on the host

## build

needs the Xcode command line tools (`xcode-select --install`). the shaders compile at runtime.

```
./build.sh
```

if linking fails with `unknown architecture arm64e.x1...`, your Xcode is older than your command line tools. build with `DEVELOPER_DIR=/Library/Developer/CommandLineTools ./build.sh` or update Xcode.

## run

```
./metal-eq --nonces 300                    # benchmark on a fixed header
./metal-eq --nonces 50 --verify-out f.txt  # write the solutions out so you can check them
./metal-eq --profile 20                    # GPU time per pass
./metal-eq --mining                        # silentarmy's stdin/stdout mining protocol (what zebrad uses)
```

compare runs/s between settings (runs divided by seconds, both on the last line), not Sol/s. Sol/s is noisy because the number of solutions per run varies (~1.9 on average).

check solutions with the independent verifier (BLAKE2b from python's hashlib plus the full Equihash tree rules):

```
./metal-eq --nonces 50 --verify-out f.txt && python3 tools/eqcheck.py f.txt
```

useful flags:
- `--bpt N`: buckets per threadgroup in the plain v2 kernels (rounds 1-4 and the final round, every round with `--lean 0`), a power of 2 (default 1). 4 with `--lean 0` on the M3 Max
- `--cpu-sg N`: the CPU hashes N/8 of round 0 (default 1, 0 = off, and `--profile` runs without it). 2 is best on the M5 Pro, 0 on the M3 Max. 3+ is too much for the CPU
- `--cpu-threads N`: CPU hashing threads (default 6)
- `--lean N`: 6 = lean kernels with the on-chip cache (default), 5 = without it, 0 = off. use 0 with `--bpt` above 1
- `--lean-from R`: use the lean kernels from round R on (default 5)
- `--inflight K`: Equihash runs in flight (default 3)
- `--v1`: the original straight port of silentarmy

## mining with zebrad (Crosslink)

1. get the Crosslink source at the latest release (v18 right now): `git clone https://github.com/ShieldedLabs/crosslink_monolith && cd crosslink_monolith && git checkout v18`
2. apply the patch: `git apply /path/to/metal-eq/zebrad/zebrad-gpu-solvers-v18.patch`
3. build zebrad as usual (`cd zebra-crosslink && cargo build --release --bin zebrad`). needs Rust via rustup (the repo pins its own toolchain and rustup fetches it) and protoc, eg `brew install protobuf`
4. start it with these env vars:
   - `SILENTARMY_BIN=/path/to/metal-eq`
   - `SILENTARMY_SOLVERS=1` (one process is enough, it keeps the whole GPU busy)

   eg `SILENTARMY_BIN=/path/to/metal-eq SILENTARMY_SOLVERS=1 ./target/release/zebrad start`
5. set `internal_miner_threads = 0` under `[mining]` in your zebrad.toml. with the patch, 0 really means no CPU solvers once GPU solvers are on. thats better on a Mac: a CPU solver thread only adds ~1 Sol/s, and while the GPU mines the CPU and GPU share one power budget (on the M5 Pro the CPU is held at ~5 W), so those watts do more in the GPU. blocks pay the node's own wallet unless you set `miner_address` under `[mining]`

to mine with flags, point `SILENTARMY_BIN` at a small wrapper script (and `chmod +x` it), eg for the M3 Max:

```
#!/bin/sh
exec /path/to/metal-eq --bpt 4 --lean 0 --cpu-sg 0 "$@"
```

what the patch does:
- runs the GPU solver as a child process per solver id, speaking silentarmy's protocol, and restarts it if it dies
- re-checks every solution against the real block before it can become one: Equihash validity first, then the block hash against the target, the same way zebrad checks its own (the solver only sees the header, not the BFT fat pointer)
- stops the GPU solver when theres nothing to mine (eg you switch MINING off in zebrad's window) and when zebrad shuts down. the next template starts it again
- logs stats every 10 min (`GPU solvers: Equihash ... solutions_per_second=... invalid=... blocks_found=...`)
- cuts every solver's nap after a cancelled template from 2 s to 50 ms (Crosslink templates change every few seconds, and during the nap the GPU keeps mining the old one). its the one change that also applies without `SILENTARMY_SOLVERS`

dont name your own env vars `ZEBRA_*`, zebra reads all of those as config and refuses to start on unknown ones.

Crosslink puts out new releases often and old ones stop working after a while, so rebuild on each new tag. the patch only touches zebrad's miner. this one is for v18: v17 changed which hash the block target applies to, so the older versions of the patch dont fit v17+. if it doesnt apply on a newer tag open an issue.

## tips

- MacBooks: keep it plugged in and turn on High Power mode (System Settings > Battery > Energy Mode) if yours has it, otherwise the GPU clocks down ~18% after ~45 min. High Power gets most of that back. the fans get loud
- a laptop with a Max chip can draw more while mining than a smaller charger delivers, and then the battery slowly drains. use the biggest charger made for it
- keep the Mac awake while it mines, eg `caffeinate -is` in another terminal. a sleeping Mac drops its peers and stops mining
- other GPU heavy apps (local LLMs etc) slow down a lot while it mines. pause it with `pkill -STOP metal-eq` and resume with `pkill -CONT metal-eq`

## credits

- SILENTARMY by Marc Bevand (MIT): metal-eq started as a port of it, incl. the BLAKE2b helper and the mining protocol. sha256.c is Igor Pavlov's public domain code, via silentarmy
- the bucketed design follows the ideas in John Tromp's equihash solvers and djezo's CUDA solver

## license

MIT, see LICENSE (it also carries SILENTARMY's MIT notice). the patch in `zebrad/` changes Crosslink's code: our changes are MIT or Apache-2.0 like Zebra, the upstream code it touches keeps its own terms (see LICENSE).
