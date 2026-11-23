#!/usr/bin/env python3
"""Independent Equihash (200,9) checker for metal-eq --verify-out files (one solution per line:
header hex (108 B), nonce hex (32 B), minimal-encoded solution hex (1344 B)).

For each solution: unpack the 512 21-bit indices, hash each index with BLAKE2b
(personalization "ZcashPoW" || n || k, 50-byte digest = 2 hashes of 200 bits), then walk the
binary tree level by level: each pair must collide on the next 20-bit digit, keep the left
subtree's first index below the right's, and share no index; the root must XOR to zero.
Usage: eqcheck.py FILE...      Prints valid/invalid counts and the set of solutions (for diffs)."""
import hashlib, struct, sys

N, K = 200, 9
PERSON = b"ZcashPoW" + struct.pack("<II", N, K)


def indices(sol):
    v = int.from_bytes(sol, "big")
    bits = len(sol) * 8
    return [(v >> (bits - 21 * (i + 1))) & ((1 << 21) - 1) for i in range(512)]


def leaf(prefix, i, cache):
    j = i // 2
    if j not in cache:
        cache[j] = hashlib.blake2b(prefix + struct.pack("<I", j), digest_size=50, person=PERSON).digest()
    h = cache[j][(i % 2) * 25:(i % 2) * 25 + 25]
    return int.from_bytes(h, "big")         # 200 bits, digit 0 = the top 20 bits


def valid(header, nonce, sol):
    if len(sol) != 1344:
        return False
    idx = indices(sol)
    prefix = header + nonce
    cache = {}
    level = [(leaf(prefix, i, cache), [i]) for i in idx]
    for r in range(K):
        shift = N - 20 * (r + 1)                # digit r occupies bits [shift, shift + 20)
        nxt = []
        for a, b in zip(level[0::2], level[1::2]):
            x = a[0] ^ b[0]
            if (x >> shift) & 0xFFFFF:
                return False                    # no collision on digit r
            if a[1][0] >= b[1][0]:
                return False                    # ordering rule
            if set(a[1]) & set(b[1]):
                return False                    # duplicate index
            nxt.append((x, a[1] + b[1]))
        level = nxt
    return level[0][0] == 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit("usage: eqcheck.py FILE...   (FILE from metal-eq --verify-out; exits 1 if any solution is invalid)")
    any_bad = False
    for path in sys.argv[1:]:
        ok = bad = 0
        seen = set()
        for line in open(path):
            parts = line.split()
            if len(parts) != 3:
                continue
            h, n, s = (bytes.fromhex(p) for p in parts)
            if valid(h, n, s):
                ok += 1
                seen.add((n.hex(), s.hex()[:32]))
            else:
                bad += 1
        print(f"{path}: {ok} valid, {bad} invalid, {len(seen)} distinct")
        any_bad = any_bad or bad > 0
    sys.exit(1 if any_bad else 0)


if __name__ == "__main__":
    main()
