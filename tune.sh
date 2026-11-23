#!/bin/bash
# Find the fastest metal-eq flags for this Mac. Takes a minute or two.
#
#   ./tune.sh            # 300 runs per setting
#   ./tune.sh 600        # more runs per setting, steadier numbers
#
# It benchmarks a few --bpt / --lean / --cpu-sg combinations back to back and ranks them by runs/s
# (not Sol/s, which is noisy). A metal-eq that is already mining gets paused (SIGSTOP) while this
# runs and resumed when it ends, otherwise both would share the GPU. Plug your Mac in first.
set -uo pipefail
cd "$(dirname "$0")"
N=${1:-300}
[ -x ./metal-eq ] || { echo "build first: ./build.sh"; exit 1; }

paused=$(pgrep '^metal-eq' | tr '\n' ' ' | sed 's/ *$//')  # also catches a renamed copy, eg metal-eq.real
resume() { [ -n "$paused" ] && kill -CONT $paused 2>/dev/null && echo "resumed the paused miner"; }
trap resume EXIT
trap 'exit 130' INT TERM
if [ -n "$paused" ]; then
    echo "pausing the running metal-eq (pid $paused) until the tune ends"
    kill -STOP $paused
fi

# prints runs/s for one setting ("<runs> runs, <sols> solutions in <secs> s: ..." is the last line)
bench() {
    ./metal-eq --nonces "$N" "$@" 2>/dev/null | awk '/ runs, .* solutions in / {
        for (i = 1; i <= NF; i++) if ($i == "in") { printf "%.1f\n", $1 / $(i + 1); exit } }'
}

settings=(
    "--bpt 1"
    "--bpt 1 --lean 0"
    "--bpt 2 --lean 0"
    "--bpt 4 --lean 0"
)
results=()
echo "warming up the GPU..."
bench >/dev/null
for s in "${settings[@]}"; do
    for sg in 0 1 2; do
        flags="$s --cpu-sg $sg"
        r=$(bench $flags)
        [ -n "$r" ] || { echo "  $flags: failed"; continue; }
        printf "  %-28s %7s runs/s\n" "$flags" "$r"
        results+=("$r|$flags")
    done
done
[ ${#results[@]} -gt 0 ] || { echo "no setting ran, try ./metal-eq --nonces 50 by hand"; exit 1; }

# re-run the top 3 once more, in reverse order, and rank by the mean of both rounds
echo "re-checking the top 3..."
top=$(printf '%s\n' "${results[@]}" | sort -t'|' -k1,1 -rn | head -n 3)
final=()
while IFS='|' read -r r1 flags; do
    r2=$(bench $flags)
    [ -n "$r2" ] || r2=$r1
    final+=("$(awk -v a="$r1" -v b="$r2" 'BEGIN { printf "%.1f", (a + b) / 2 }')|$flags")
done < <(printf '%s\n' "$top" | tail -r)

echo
echo "best settings on this Mac (mean runs/s of two rounds):"
printf '%s\n' "${final[@]}" | sort -t'|' -k1,1 -rn | awk -F'|' '{ printf "  %7s  %s\n", $1, $2 }'
best=$(printf '%s\n' "${final[@]}" | sort -t'|' -k1,1 -rn | head -n 1 | cut -d'|' -f2)
echo
echo "to mine with it, point SILENTARMY_BIN at a wrapper like:"
echo "  #!/bin/sh"
echo "  exec $(pwd)/metal-eq $best \"\$@\""
echo "differences under ~2% are noise; run ./tune.sh 600 to be sure."
