#!/bin/bash
# env.sh — site settings and helpers for the checkpoint-read experiment on
# anjuna2. Source it; do not execute it.
#
# anjuna2 has no batch scheduler, so these are plain shell scripts driven by
# torchrun. That removes the one thing Slurm was giving us for free -- a fresh
# allocation per job, which guaranteed cold page cache -- so cache control here
# is explicit and is the single easiest way to get meaningless numbers. See
# drop_caches() below.

# --- site ---------------------------------------------------------------
: "${BEEGFS_ROOT:=/mnt/beegfs/$USER/ckpt_experiment}"
: "${PYTHON:=python3}"
: "${TORCHRUN:=torchrun}"
: "${WORKLOAD_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# OSS hosts whose page cache must also be dropped. Dropping caches on anjuna2
# alone is NOT enough: colva1-4 have their own RAM and will serve a checkpoint
# straight back out of it. Needs passwordless ssh + sudo, or run it by hand.
: "${OSS_HOSTS:=}"          # e.g. "colva1 colva2 colva3 colva4"

# Root helper that performs the drop. A NOPASSWD sudoers rule on `sh -c ...` is
# passwordless root and is also brittle -- sudoers matches the command line
# literally, so any whitespace change here silently reintroduces a password
# prompt, which `sudo -n` turns into a failed drop. A fixed wrapper path is one
# stable thing to authorise instead. Install on anjuna2 AND every OSS:
#   printf '#!/bin/sh\nsync; echo 3 > /proc/sys/vm/drop_caches\n' \
#     | sudo tee /usr/local/sbin/drop-caches >/dev/null
#   sudo chmod 755 /usr/local/sbin/drop-caches
#   echo "$USER ALL=(root) NOPASSWD: /usr/local/sbin/drop-caches" \
#     | sudo tee /etc/sudoers.d/dropcaches >/dev/null && sudo chmod 440 /etc/sudoers.d/dropcaches
# Hosts without the wrapper fall back to the old inline form.
: "${DROP_CACHES_CMD:=/usr/local/sbin/drop-caches}"

# --- BeeGFS placement ---------------------------------------------------
# Which storage pool the checkpoint lands in. Unset means the parent directory's
# pool, which is fine for a baseline but is not reproducible -- pool membership
# changes. A media comparison needs a pool that is single-media; check with
# `beegfs-ctl --liststoragepools` before assuming pool 3 is HDD and 6 is SSD.
: "${POOL_ID:=}"            # 3 = hdd, 6 = ssd (check: beegfs-ctl --liststoragepools)
: "${STRIPE_COUNT:=4}"
: "${CHUNKSIZE:=512k}"
# Optional JSON mapping each .distcp to its ordered BeeGFS target IDs. Without
# it, parse_darshan.py reports stripe *slots* (0..STRIPE_COUNT-1) relative to
# each file's own target list, which cannot be compared across files -- BeeGFS
# gives every file a different starting target. Build it with:
#   run/make_target_map.sh "$CKPT_DIR" > targets.json
: "${TARGET_MAP:=}"

# --- model / parallelism ------------------------------------------------
: "${MODEL:=llama31_70b}"
: "${SAVE_CONFIG:=TP8/PP4/DP1}"   # writer: training-shaped, 32 ranks
: "${LOAD_CONFIG:=TP4}"           # reader: inference-shaped
: "${OPTIMIZER:=adam}"
: "${LAYOUT:=per-rank}"           # per-rank | per-rank-threads | per-shard
: "${THREAD_COUNT:=1}"
: "${FILL:=random}"               # real bytes: we have the space now
: "${DEVICE:=cpu}"                # no GPUs needed; DCP reads to CPU regardless
: "${RAM_BUDGET_GB:=}"            # blank = autodetect from free(1)

TP="$(printf '%s' "$LOAD_CONFIG" | sed -n 's/.*[Tt][Pp]\([0-9]\+\).*/\1/p')"
: "${TP:=1}"
: "${NPROC:=$TP}"

: "${TAG_BASE:=${MODEL}_${OPTIMIZER}_${LAYOUT}${POOL_ID:+_pool$POOL_ID}}"
: "${CKPT_DIR:=$BEEGFS_ROOT/$TAG_BASE}"
: "${RESULTS_DIR:=$BEEGFS_ROOT/results}"
: "${DARSHAN_LOGS:=$BEEGFS_ROOT/darshan}"

mkdir -p "$RESULTS_DIR" "$DARSHAN_LOGS"

# --- helpers ------------------------------------------------------------

# Largest --only-layers that fits in RAM. Reads the real tensor inventory
# rather than a hardcoded table, so it stays correct if the model changes.
autosize_layers() {
    local read_set="${1:-eval}" budget="$RAM_BUDGET_GB"
    if [[ -z "$budget" ]]; then
        # 70% of *available*, not total: leave room for the page cache the
        # read itself will churn through.
        budget=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024*0.7}' /proc/meminfo)
    fi
    "$PYTHON" "$WORKLOAD_DIR/capacity.py" --model "$MODEL" --read-set "$read_set" \
        --optimizer "$OPTIMIZER" --load-config "$LOAD_CONFIG" \
        --ram-gb "$budget" --quiet
}

# Pin a directory to a storage pool and stripe pattern. Must run on the
# directory BEFORE any files are created in it -- BeeGFS applies the pattern at
# file creation time and will not retroactively move anything.
set_stripe() {
    local dir="$1"
    mkdir -p "$dir"
    if ! command -v beegfs-ctl >/dev/null 2>&1; then
        echo "  (beegfs-ctl not found; leaving default stripe pattern)" >&2
        return 0
    fi
    local args=(--setpattern --chunksize="$CHUNKSIZE" --numtargets="$STRIPE_COUNT")
    [[ -n "$POOL_ID" ]] && args+=(--storagepoolid="$POOL_ID")
    echo "  beegfs-ctl ${args[*]} $dir"
    sudo beegfs-ctl "${args[@]}" "$dir" || \
        echo "  WARNING: setpattern failed; pattern is whatever the parent had" >&2
    sudo beegfs-ctl --getentryinfo "$dir" | sed 's/^/    /'
}

# Cold-cache a read. Without this the pool A/B measures RAM, not disks.
# The shell fragment run as root on each host. Prefers the authorised wrapper
# and falls back to the inline form where it is not installed, so a host that
# has not been set up yet still drops (interactively) rather than being skipped.
_drop_snippet="if [ -x $DROP_CACHES_CMD ]; then sudo -n $DROP_CACHES_CMD; \
else sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; fi"

# Reports the host's Cached figure before and after. `sudo -n` succeeding does
# not mean the cache went away -- report the delta so a drop that did not take
# is visible in the run log instead of being inferred later from odd timings.
_cached_kb() { awk '/^Cached:/ {print $2; exit}' /proc/meminfo; }

drop_caches() {
    sync
    local before after
    before=$(_cached_kb)
    if eval "$_drop_snippet" 2>/dev/null; then
        after=$(_cached_kb)
        echo "  dropped page cache on $(hostname): Cached ${before} -> ${after} kB"
    else
        echo "  WARNING: could not drop cache on $(hostname) (needs sudo;" >&2
        echo "           install $DROP_CACHES_CMD and its sudoers rule)" >&2
    fi
    if [[ -z "$OSS_HOSTS" ]]; then
        echo "  WARNING: OSS_HOSTS is empty -- server-side page cache on the" >&2
        echo "           OSSs was NOT dropped. Timings may reflect their RAM," >&2
        echo "           not their disks. Set OSS_HOSTS='colva1 colva2 ...'." >&2
    else
        for h in $OSS_HOSTS; do
            out=$(ssh -o BatchMode=yes "$h" \
                "b=\$(awk '/^Cached:/{print \$2;exit}' /proc/meminfo); sync; \
                 $_drop_snippet || exit 1; \
                 a=\$(awk '/^Cached:/{print \$2;exit}' /proc/meminfo); \
                 echo \"\$b \$a\"" 2>/dev/null) \
                && echo "  dropped page cache on $h: Cached $(echo "$out" | \
                       awk '{print $1" -> "$2}') kB" \
                || echo "  WARNING: cache drop failed on $h" >&2
        done
    fi
}

# Darshan. gloo is not MPI, so DARSHAN_ENABLE_NONMPI is mandatory -- without it
# Darshan produces no log at all and fails silently. DXT is required because
# aggregate POSIX counters cannot show per-offset reuse, which is the whole
# read-amplification result.
setup_darshan() {
    # A non-MPI build is searched for FIRST. gloo is not MPI, and a Darshan
    # built with MPI support silently instruments nothing for a non-MPI
    # program -- it produces an empty log directory, not an error. Confirmed
    # on anjuna2: /usr/local/lib/libdarshan.so has 117 MPI symbols and emits
    # no log for a plain python read.
    if [[ -z "${DARSHAN_RUNTIME_LIB:-}" ]]; then
        for c in "$HOME"/darshan-nompi/lib/libdarshan.so \
                 /opt/darshan-nompi/lib/libdarshan.so \
                 /usr/lib/libdarshan.so /usr/lib64/libdarshan.so \
                 /usr/local/lib/libdarshan.so "$HOME"/darshan*/lib/libdarshan.so \
                 /opt/darshan*/lib/libdarshan.so; do
            [[ -f "$c" ]] && { DARSHAN_RUNTIME_LIB="$c"; break; }
        done
    fi
    if [[ -z "${DARSHAN_RUNTIME_LIB:-}" || ! -f "${DARSHAN_RUNTIME_LIB:-}" ]]; then
        echo "  Darshan runtime not found; falling back to strace." >&2
        echo "  Locate it with: find / -name 'libdarshan.so*' 2>/dev/null" >&2
        echo "  then re-run with DARSHAN_RUNTIME_LIB=/path/to/libdarshan.so" >&2
        return 1
    fi
    # Refuse an MPI build rather than collect nothing.
    if command -v nm >/dev/null 2>&1; then
        local nmpi
        nmpi=$(nm -D "$DARSHAN_RUNTIME_LIB" 2>/dev/null | grep -c 'MPI_' || true)
        if [[ "${nmpi:-0}" -gt 0 ]]; then
            echo "  $DARSHAN_RUNTIME_LIB is an MPI build ($nmpi MPI symbols)." >&2
            echo "  It will instrument NOTHING for a gloo job and will fail" >&2
            echo "  silently with an empty log dir. Build a non-MPI Darshan:" >&2
            echo "    cd ~/darshan-3.4.6/darshan-runtime && ./configure \\" >&2
            echo "      --prefix=\$HOME/darshan-nompi --without-mpi \\" >&2
            echo "      --with-log-path-by-env=DARSHAN_LOG_DIR_PATH \\" >&2
            echo "      --with-jobid-env=NONE CC=gcc && make -j && make install" >&2
            echo "  then re-run; it is picked up automatically." >&2
            return 1
        fi
    fi
    # Deliberately does NOT export LD_PRELOAD. Exporting it instruments every
    # later command in the calling script -- mkdir, find, even the log parser --
    # and their logs land in the same directory and get collected as if they
    # were the workload. (Observed: a run whose Darshan "exe" was `mkdir`, with
    # 676 files and 0.04 GB against a real 16.06 GB read.) The caller builds
    # DARSHAN_ENV and applies it to the workload command alone.
    mkdir -p "$DARSHAN_LOGS"
    # Darshan allocates file records in open() order from a fixed per-module
    # buffer (DARSHAN_MODMEM, 2 MiB by default). A python process opens
    # thousands of .py/.pyc files before it does any real work, and torch here
    # lives under $HOME/.local, which is NOT in Darshan's compiled-in exclude
    # list (/usr/, /lib/, /etc/, ...). The buffer filled with import records and
    # the checkpoint reads were never recorded at all -- the log said
    # "POSIX module contains incomplete data" and our report said 0 bytes
    # against a real 16 GB read.
    #
    # DARSHAN_EXCLUDE_DIRS REPLACES the built-in list, so the defaults are
    # repeated here and the interpreter's directories added.
    : "${DARSHAN_MODMEM:=64}"          # MiB per module
    : "${DARSHAN_EXCLUDE_DIRS:=/etc/,/dev/,/usr/,/bin/,/boot/,/lib/,/lib64/,/opt/,/sbin/,/sys/,/proc/,/var/,/tmp/,$HOME/.local/,$HOME/.cache/,$HOME/darshan-nompi/}"
    DARSHAN_ENV=(env "LD_PRELOAD=$DARSHAN_RUNTIME_LIB"
                     "DARSHAN_ENABLE_NONMPI=1"
                     "DARSHAN_LOG_DIR_PATH=$DARSHAN_LOGS"
                     "DARSHAN_MODMEM=$DARSHAN_MODMEM"
                     "DARSHAN_EXCLUDE_DIRS=$DARSHAN_EXCLUDE_DIRS"
                     "DXT_ENABLE_IO_TRACE=1")
    echo "  Darshan: $DARSHAN_RUNTIME_LIB -> $DARSHAN_LOGS"
    echo "  (preloaded onto the workload only, not this shell)"
    echo "  MODMEM ${DARSHAN_MODMEM} MiB/module, excluding the interpreter's dirs"
    return 0
}

banner() { echo; echo "=== $* ==="; }
