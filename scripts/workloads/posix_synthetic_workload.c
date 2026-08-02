/*
 * posix_synthetic_workload.c
 *
 * Simulates configurable POSIX I/O workloads for Darshan instrumentation.
 * Uses raw POSIX syscalls (open/read/write/lseek/fsync) — Darshan records
 * these under the POSIX module. Does not simulate MPI-IO or STDIO.
 *
 * All parameters are passed as CLI arguments by run_workloads.py.
 *
 * Usage:
 *   ./posix_synthetic_workload <profile_name> <read_ratio> <access_pattern>
 *                              <stride_size> <op_size> <num_ops> <num_files>
 *                              <num_phases> <fsync_interval> <work_dir> <mode>
 *                              [block_size] [nd_dims]
 *
 * access_pattern: 0 = sequential, 1 = random, 2 = strided, 3 = nd_strided
 *
 * block_size / nd_dims are optional and only apply to nd_strided (pattern 3):
 *   block_size — contiguous bytes touched per innermost run. MUST be smaller
 *                than stride_size, otherwise there is no gap between rows and
 *                the pattern degenerates into plain sequential I/O (see the
 *                nd_strided geometry notes below). Defaults to stride_size/2.
 *   nd_dims    — number of dimensions in the strided selection (2..5).
 *                Defaults to 2. Each dimension contributes one additional
 *                distinct offset-delta, so keep this <= 4 if the pattern must
 *                stay visible in Darshan's four POSIX_STRIDE*_STRIDE slots.
 *
 * mode:
 *   0 = SETUP   — write files to disk without Darshan attached.
 *                 Only needed for pure-read profiles (read_ratio == 1.0).
 *                 Mixed profiles create their own files in workload mode.
 *                 No reads are performed. No cleanup.
 *   1 = WORKLOAD — the measured run. Darshan is attached.
 *                 For pure-read profiles: skips all writes, reads from
 *                 files created during setup. Cleans up files on exit.
 *                 For pure-write/mixed profiles: writes and reads normally.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <errno.h>
#include <mpi.h>

/* Run mode constants */
#define MODE_SETUP 0
#define MODE_WORKLOAD 1

/* Access pattern constants */
#define PATTERN_SEQUENTIAL 0
#define PATTERN_RANDOM 1
#define PATTERN_STRIDED 2
#define PATTERN_ND_STRIDED 3

/* Max dimensions for nd_strided. Matches the ceiling Darshan itself tracks in
 * its HDF5/PnetCDF modules (H5D_MAX_NDIMS / PNETCDF_VAR_MAX_NDIMS == 5). */
#define ND_MAX_DIMS 5

/* -------------------------------------------------------------------------
 * Profile — populated from CLI args
 * ---------------------------------------------------------------------- */
typedef struct
{
    char profile_name[256];
    double read_ratio;
    int access_pattern;
    long stride_size;
    long op_size;
    long num_ops;
    int num_files;
    int num_phases;
    int fsync_interval;
    char work_dir[1024];
    int mode;        /* MODE_SETUP or MODE_WORKLOAD */
    long block_size; /* nd_strided: contiguous bytes per innermost run */
    int nd_dims;     /* nd_strided: dimensions in the selection (2..ND_MAX_DIMS) */
} Profile;

/* -------------------------------------------------------------------------
 * nd_strided geometry — precomputed once per file.
 *
 * Models an N-dimensional sub-block selection over the file, the same shape
 * an HDF5 hyperslab or PnetCDF start/count/stride selection produces:
 *
 *   dim 0 (innermost) — a contiguous run of block_size bytes, walked in
 *                       op_size steps.
 *   dim 1             — jump to the next row by stride_size, leaving a real
 *                       (stride_size - block_size) byte hole behind.
 *   dim k >= 2        — jump past the entire span of the dims below, scaled
 *                       by the same sparsity ratio so a hole remains at every
 *                       level.
 *
 * This is the part the previous implementation got wrong: it derived the row
 * width from the row pitch (cols = stride_size / op_size), which forces
 * block_size == stride_size and closes the hole entirely. The offsets then
 * collapse algebraically to `idx * op_size` — literally sequential I/O, not
 * merely something that resembles it — so the pattern was indistinguishable
 * from a contiguous profile no matter which tool observed it. Keeping the
 * width (block_size) independent of the pitch (stride_size) is what makes the
 * access genuinely strided at any file size.
 * ---------------------------------------------------------------------- */
typedef struct
{
    int ndims;
    long extent[ND_MAX_DIMS]; /* positions per dimension, index 0 = innermost */
    long pitch[ND_MAX_DIMS];  /* byte stride per dimension */
    long total;               /* product of all extents (odometer wrap point) */
} NdGeometry;

/* -------------------------------------------------------------------------
 * Utility: allocate and fill an O_DIRECT-compatible buffer.
 * O_DIRECT requires buffers aligned to the filesystem block size (4096).
 * Uses posix_memalign to guarantee 4096-byte alignment.
 * ---------------------------------------------------------------------- */
static char *make_buffer(long size)
{
    void *buf = NULL;
    /* Align to 4096 — required for O_DIRECT */
    if (posix_memalign(&buf, 4096, (size_t)size) != 0)
    {
        fprintf(stderr, "posix_memalign failed for buffer of size %ld\n", size);
        exit(1);
    }
    char *cbuf = (char *)buf;
    for (long i = 0; i < size; i++)
        cbuf[i] = (char)(i & 0xFF);
    return cbuf;
}

/* -------------------------------------------------------------------------
 * Utility: retry write until all 'size' bytes are written (inlined for speed).
 * Returns 0 on success, -1 on error (errno set by underlying write).
 * ---------------------------------------------------------------------- */
static inline int full_write(int fd, const char *buf, long size)
{
    long remaining = size;
    while (remaining > 0)
    {
        ssize_t n = write(fd, buf + (size - remaining), (size_t)remaining);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return -1;
        }
        remaining -= n;
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Utility: retry read until all 'size' bytes are read or EOF (inlined for speed).
 * Returns bytes actually read (< size only on EOF), -1 on error.
 * ---------------------------------------------------------------------- */
static inline long full_read(int fd, char *buf, long size)
{
    long total = 0;
    while (total < size)
    {
        ssize_t n = read(fd, buf + total, (size_t)(size - total));
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (n == 0)
            break; /* EOF */
        total += n;
    }
    return total;
}

/* -------------------------------------------------------------------------
 * Fast inline LCG random number generator (faster than rand_r)
 * ---------------------------------------------------------------------- */
static inline unsigned int fast_rand(unsigned int *seed)
{
    *seed = (*seed * 1103515245U + 12345U) & 0x7fffffffU;
    return *seed;
}

/* -------------------------------------------------------------------------
 * Utility: random aligned block offset within [0, file_size - op_size].
 * Guarantees offset + op_size <= file_size.
 * ---------------------------------------------------------------------- */
static inline long random_offset(long file_size, long op_size, unsigned int *seed)
{
    /* Number of complete blocks that fit, starting from offset 0 */
    long num_blocks = (file_size - op_size) / op_size + 1;
    if (num_blocks <= 0)
        return 0;
    long block = (long)(fast_rand(seed) % (unsigned long)num_blocks);
    return block * op_size;
}

/* -------------------------------------------------------------------------
 * Utility: integer nth root — largest r with r^n <= v. Used to spread the
 * available file extent evenly across nd_strided's outer dimensions.
 * Avoids pow()/-lm so the existing plain `gcc -O3` compile line still works.
 * ---------------------------------------------------------------------- */
static long int_nth_root(long v, int n)
{
    if (n <= 1)
        return v;
    if (v < 1)
        return 1;

    long r = 1;
    while (1)
    {
        long next = r + 1;
        long pw = 1;
        int over = 0;
        for (int i = 0; i < n; i++)
        {
            if (pw > v / next) /* pw * next would exceed v */
            {
                over = 1;
                break;
            }
            pw *= next;
        }
        if (over || pw > v)
            break;
        r = next;
    }
    return r;
}

/* -------------------------------------------------------------------------
 * Build the nd_strided geometry for a file of the given size.
 *
 * The selection is sized to span as much of the file as the dimension count
 * allows, with every dimension keeping a real gap between consecutive
 * positions. Yields exactly `ndims` distinct offset-deltas, which is what
 * makes the pattern separable from contiguous I/O in the POSIX stride
 * counters at any file size.
 * ---------------------------------------------------------------------- */
static void nd_geometry_init(NdGeometry *g, const Profile *p, long file_size)
{
    int nd = p->nd_dims;
    if (nd < 2)
        nd = 2;
    if (nd > ND_MAX_DIMS)
        nd = ND_MAX_DIMS;

    long block = p->block_size;
    if (block < p->op_size)
        block = p->op_size;
    if (block > p->stride_size)
        block = p->stride_size; /* validated away in main(), clamped defensively */

    /* Innermost: contiguous run of `block` bytes in op_size steps. */
    g->pitch[0] = p->op_size;
    g->extent[0] = block / p->op_size;
    if (g->extent[0] < 1)
        g->extent[0] = 1;

    /* Sparsity ratio between levels — how much bigger a pitch is than the
     * span it jumps over. Derived from the caller's own block/stride choice
     * so the hole size stays consistent at every level. */
    long ratio = p->stride_size / block;
    if (ratio < 2)
        ratio = 2;

    /* Target positions per outer dimension: spread the file's expansion
     * factor (file_size / block) evenly across the nd-1 outer dims. */
    long expansion = file_size / (block > 0 ? block : 1);
    if (expansion < 1)
        expansion = 1;
    long target = int_nth_root(expansion, nd - 1);
    if (target < 2)
        target = 2;

    long reach = g->extent[0] * g->pitch[0]; /* bytes spanned by dims 0..k */

    for (int k = 1; k < nd; k++)
    {
        g->pitch[k] = (k == 1) ? p->stride_size : reach * ratio;

        long max_pos = (g->pitch[k] > 0) ? file_size / g->pitch[k] : 1;
        if (max_pos < 1)
            max_pos = 1;

        g->extent[k] = (target < max_pos) ? target : max_pos;
        reach = g->extent[k] * g->pitch[k];
    }

    g->ndims = nd;
    g->total = 1;
    for (int k = 0; k < nd; k++)
        g->total *= g->extent[k];
    if (g->total < 1)
        g->total = 1;
}

/* -------------------------------------------------------------------------
 * Map a monotonically increasing op index onto an nd_strided file offset by
 * treating the index as an odometer over the selection's dimensions
 * (innermost dimension turning fastest). Wraps at g->total so num_ops may
 * exceed the size of the selection.
 * ---------------------------------------------------------------------- */
static inline long nd_offset(const NdGeometry *g, long idx)
{
    long rem = idx % g->total;
    long off = 0;

    for (int k = 0; k < g->ndims; k++)
    {
        off += (rem % g->extent[k]) * g->pitch[k];
        rem /= g->extent[k];
    }
    return off;
}

/* -------------------------------------------------------------------------
 * Utility: calculate next offset based on access pattern (inlined for speed)
 * ---------------------------------------------------------------------- */
static inline long calculate_offset(const Profile *p, long file_size,
                                    long *sequential_offset, long *stride_cursor,
                                    const NdGeometry *nd, unsigned int *seed)
{
    long offset;

    if (p->access_pattern == PATTERN_SEQUENTIAL)
    {
        offset = *sequential_offset;
        *sequential_offset += p->op_size;
        if (*sequential_offset + p->op_size > file_size)
            *sequential_offset = 0; /* wrap for reads */
    }
    else if (p->access_pattern == PATTERN_RANDOM)
    {
        offset = random_offset(file_size, p->op_size, seed);
    }
    else if (p->access_pattern == PATTERN_STRIDED)
    {
        long raw = (*stride_cursor * p->stride_size) %
                   (file_size > 0 ? file_size : p->stride_size);
        offset = (raw / p->op_size) * p->op_size;
        (*stride_cursor)++;
    }
    else /* PATTERN_ND_STRIDED */
    {
        long raw = nd_offset(nd, *stride_cursor) % (file_size > 0 ? file_size : p->op_size);
        offset = (raw / p->op_size) * p->op_size;
        (*stride_cursor)++;
    }

    return offset;
}

/* -------------------------------------------------------------------------
 * Single-file workload: write phase
 *
 * Parameters:
 *   fd             — open file descriptor
 *   p              — profile
 *   ops_in_phase   — number of write ops to perform this phase
 *   file_size      — total file size (fixed); random writes stay within this
 *   write_offset   — cursor for sequential writes (in/out)
 *   stride_cursor  — global op index for strided writes (in/out)
 *   global_op      — global op count for fsync_interval (in/out)
 *   buf            — pre-allocated write buffer of p->op_size bytes
 *   nd             — precomputed nd_strided geometry (unused by other patterns)
 *   seed           — PRNG state (in/out)
 * Returns 0 on success, -1 if a fatal I/O error occurred (file should be
 * abandoned — Darshan counters for it will be incomplete).
 * ---------------------------------------------------------------------- */
static int do_write_phase(int fd, const Profile *p, long ops_in_phase,
                          long file_size, long *write_offset,
                          long *stride_cursor, long *global_op,
                          char *buf, const NdGeometry *nd, unsigned int *seed)
{
    for (long i = 0; i < ops_in_phase; i++)
    {
        long offset = calculate_offset(p, file_size, write_offset, stride_cursor, nd, seed);

        if (lseek(fd, offset, SEEK_SET) < 0)
        {
            perror("lseek (write)");
            return -1;
        }
        if (full_write(fd, buf, p->op_size) < 0)
        {
            perror("write");
            return -1;
        }

        (*global_op)++;
        if (p->fsync_interval > 0 && *global_op % p->fsync_interval == 0)
            fsync(fd);
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Single-file workload: read phase
 *
 * Parameters:
 *   fd             — open file descriptor
 *   p              — profile
 *   ops_in_phase   — number of read ops to perform this phase
 *   file_size      — total file size; random/strided offsets stay within this
 *   read_offset    — cursor for sequential reads (in/out)
 *   stride_cursor  — global op index for strided reads (in/out)
 *   global_op      — global op count for fsync_interval (in/out)
 *   buf            — pre-allocated read buffer of p->op_size bytes
 *   nd             — precomputed nd_strided geometry (unused by other patterns)
 *   seed           — PRNG state (in/out)
 * Returns 0 on success, -1 if a fatal I/O error occurred.
 * ---------------------------------------------------------------------- */
static int do_read_phase(int fd, const Profile *p, long ops_in_phase,
                         long file_size, long *read_offset,
                         long *stride_cursor, long *global_op,
                         char *buf, const NdGeometry *nd, unsigned int *seed)
{
    for (long i = 0; i < ops_in_phase; i++)
    {
        long offset = calculate_offset(p, file_size, read_offset, stride_cursor, nd, seed);

        if (lseek(fd, offset, SEEK_SET) < 0)
        {
            perror("lseek (read)");
            return -1;
        }
        if (full_read(fd, buf, p->op_size) < 0)
        {
            perror("read");
            return -1;
        }

        (*global_op)++;
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Setup mode: write files to disk without Darshan attached.
 * Creates and fully populates each file. No reads. No cleanup.
 * ---------------------------------------------------------------------- */
static void run_setup(const Profile *p)
{
    long total_read_ops = (long)(p->num_ops * p->read_ratio);
    long total_write_ops = p->num_ops - total_read_ops;

    /* For pure-read profiles, setup still needs to write num_ops worth of data
     * so the file exists and is fully populated for the measured read run. */
    long setup_write_ops = (total_write_ops > 0) ? total_write_ops : p->num_ops;

    long ops_per_file = setup_write_ops / p->num_files;

    for (int f = 0; f < p->num_files; f++)
    {
        char filepath[2048];
        snprintf(filepath, sizeof(filepath), "%s/workload_%s_f%d",
                 p->work_dir, p->profile_name, f);

        long w_ops = (f == p->num_files - 1)
                         ? setup_write_ops - ops_per_file * (p->num_files - 1)
                         : ops_per_file;

        long file_size = w_ops * p->op_size;

        int fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
        if (fd < 0)
        {
            fprintf(stderr, "setup: open failed for %s: %s\n", filepath, strerror(errno));
            continue;
        }
        if (ftruncate(fd, file_size) < 0)
            perror("setup: ftruncate");

        /* Write sequentially regardless of profile access pattern —
         * we only care that data exists, not how it was written. */
        char *buf = make_buffer(p->op_size);
        for (long i = 0; i < w_ops; i++)
        {
            if (full_write(fd, buf, p->op_size) < 0)
            {
                perror("setup: write");
                break;
            }
        }
        free(buf);
        fsync(fd);
        close(fd);
    }
}

/* -------------------------------------------------------------------------
 * Run workload on a single file (measured — Darshan attached)
 * ---------------------------------------------------------------------- */
static void run_file_workload(const Profile *p, const char *filepath,
                              long total_write_ops, long total_read_ops)
{
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)getpid();

    /* File size = full dataset regardless of read/write split so that random
     * and strided reads always cover the intended data extent. */
    long file_size = (total_write_ops + total_read_ops) * p->op_size;
    if (p->mode == MODE_WORKLOAD && p->read_ratio >= 1.0)
    {
        /* Pure-read: file was pre-populated by setup with num_ops * op_size bytes */
        file_size = (total_read_ops > 0 ? total_read_ops : p->num_ops) * p->op_size;
    }

    /* Open flags — O_DIRECT bypasses page cache for true storage throughput */
    int flags;
    if (p->mode == MODE_WORKLOAD && p->read_ratio >= 1.0)
        flags = O_RDONLY | O_DIRECT;
    else
        flags = O_RDWR | O_CREAT | O_TRUNC | O_DIRECT;

    int fd = open(filepath, flags, 0644);
    if (fd < 0)
    {
        fprintf(stderr, "open failed for %s: %s\n", filepath, strerror(errno));
        return;
    }

    /* Pre-allocate only when we own the file creation */
    if (flags != O_RDONLY)
    {
        if (ftruncate(fd, file_size) < 0)
            perror("ftruncate");
    }

    /* Allocate buffer once for all phases */
    char *buf = make_buffer(p->op_size);

    /* nd_strided selection geometry — depends on file_size, so computed here
     * once per file rather than per op. */
    NdGeometry nd;
    nd_geometry_init(&nd, p, file_size);

    /* Count write vs read phases.
     * Phase ordering: W, R, W, R, ... (phase 0 = write).
     * Exception: pure-read workload mode → all phases are reads. */
    int write_phases = 0;
    int read_phases = 0;

    if (p->mode == MODE_WORKLOAD && p->read_ratio >= 1.0)
    {
        read_phases = p->num_phases;
    }
    else
    {
        for (int ph = 0; ph < p->num_phases; ph++)
        {
            if (ph % 2 == 0)
                write_phases++;
            else
                read_phases++;
        }
    }

    long write_ops_per_write_phase = (write_phases > 0) ? total_write_ops / write_phases : 0;
    long read_ops_per_read_phase = (read_phases > 0) ? total_read_ops / read_phases : 0;

    /* Cursors */
    long write_offset = 0;        /* sequential write cursor */
    long read_offset = 0;         /* sequential read cursor  */
    long write_stride_cursor = 0; /* global strided-write op index */
    long read_stride_cursor = 0;  /* global strided-read op index  */
    long global_op = 0;           /* global op count for fsync_interval */
    long write_phase_count = 0;
    long read_phase_count = 0;

    for (int ph = 0; ph < p->num_phases; ph++)
    {
        int is_write_phase = (p->mode == MODE_WORKLOAD && p->read_ratio >= 1.0)
                                 ? 0
                                 : (ph % 2 == 0);

        if (is_write_phase && write_phase_count < write_phases)
        {
            long ops = (write_phase_count == write_phases - 1)
                           ? (total_write_ops - write_ops_per_write_phase * (write_phases - 1))
                           : write_ops_per_write_phase;
            if (do_write_phase(fd, p, ops, file_size, &write_offset,
                               &write_stride_cursor, &global_op, buf, &nd, &seed) < 0)
            {
                fprintf(stderr, "[%s] write phase %ld failed — abandoning file\n",
                        p->profile_name, write_phase_count);
                break;
            }
            write_phase_count++;
        }
        else if (!is_write_phase && read_phase_count < read_phases)
        {
            long ops = (read_phase_count == read_phases - 1)
                           ? (total_read_ops - read_ops_per_read_phase * (read_phases - 1))
                           : read_ops_per_read_phase;
            if (do_read_phase(fd, p, ops, file_size, &read_offset,
                              &read_stride_cursor, &global_op, buf, &nd, &seed) < 0)
            {
                fprintf(stderr, "[%s] read phase %ld failed — abandoning file\n",
                        p->profile_name, read_phase_count);
                break;
            }
            read_phase_count++;
        }
    }

    free(buf);
    fsync(fd);
    close(fd);
}

/* -------------------------------------------------------------------------
 * Metadata heavy workload: create / stat / delete N files
 * ---------------------------------------------------------------------- */
static void run_metadata_workload(const Profile *p)
{
    char filepath[2048];
    char *buf = make_buffer(p->op_size);
    struct stat st;

    for (int i = 0; i < p->num_files; i++)
    {
        snprintf(filepath, sizeof(filepath), "%s/meta_%s_%d",
                 p->work_dir, p->profile_name, i);

        /* CREATE */
        int fd = open(filepath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0)
        {
            fprintf(stderr, "open (meta create) failed: %s\n", strerror(errno));
            continue;
        }
        if (full_write(fd, buf, p->op_size) < 0)
            perror("write (meta)");
        if (p->fsync_interval > 0)
            fsync(fd);
        close(fd);

        /* STAT */
        if (stat(filepath, &st) < 0)
            perror("stat (meta)");

        /* DELETE */
        if (unlink(filepath) < 0)
            perror("unlink (meta)");
    }

    free(buf);
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    /* Initialize MPI (required for Darshan to activate) */
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 12)
    {
        if (rank == 0)
        {
            fprintf(stderr,
                    "Usage: %s <profile_name> <read_ratio> <access_pattern (0|1|2|3)>\n"
                    "          <stride_size> <op_size> <num_ops> <num_files>\n"
                    "          <num_phases> <fsync_interval> <work_dir> <mode (0|1)>\n"
                    "          [block_size] [nd_dims]\n"
                    "  mode 0 = setup (write files only, no Darshan — pure-read profiles only)\n"
                    "  mode 1 = workload (measured run, Darshan attached)\n"
                    "  block_size — nd_strided only: contiguous bytes per innermost run;\n"
                    "               must be < stride_size. Default stride_size/2.\n"
                    "  nd_dims    — nd_strided only: dimensions (2..%d). Default 2.\n",
                    argv[0], ND_MAX_DIMS);
        }
        MPI_Finalize();
        return 1;
    }

    Profile p;
    strncpy(p.profile_name, argv[1], sizeof(p.profile_name) - 1);
    p.read_ratio = atof(argv[2]);
    p.access_pattern = atoi(argv[3]);
    p.stride_size = atol(argv[4]);
    p.op_size = atol(argv[5]);
    p.num_ops = atol(argv[6]);
    p.num_files = atoi(argv[7]);
    p.num_phases = atoi(argv[8]);
    p.fsync_interval = atoi(argv[9]);
    strncpy(p.work_dir, argv[10], sizeof(p.work_dir) - 1);
    p.mode = atoi(argv[11]);

    /* Optional nd_strided knobs — absent/0 means "use the default". */
    p.block_size = (argc > 12) ? atol(argv[12]) : 0;
    p.nd_dims = (argc > 13) ? atoi(argv[13]) : 2;

    /* Validate */
    if (p.op_size <= 0 || p.num_ops <= 0 || p.num_phases < 1)
    {
        if (rank == 0)
            fprintf(stderr, "Invalid parameters: op_size, num_ops must be >0; num_phases >= 1\n");
        MPI_Finalize();
        return 1;
    }
    if ((p.access_pattern == PATTERN_STRIDED || p.access_pattern == PATTERN_ND_STRIDED) && p.stride_size <= 0)
    {
        if (rank == 0)
            fprintf(stderr, "stride_size must be >0 for strided or nd_strided access pattern\n");
        MPI_Finalize();
        return 1;
    }
    if (p.access_pattern == PATTERN_ND_STRIDED)
    {
        if (p.nd_dims < 2 || p.nd_dims > ND_MAX_DIMS)
        {
            if (rank == 0)
                fprintf(stderr, "nd_dims must be between 2 and %d for nd_strided\n", ND_MAX_DIMS);
            MPI_Finalize();
            return 1;
        }
        /* Need room for a contiguous run strictly inside the pitch, otherwise
         * there is no hole and the pattern degenerates to sequential I/O. */
        if (p.stride_size <= p.op_size)
        {
            if (rank == 0)
                fprintf(stderr,
                        "nd_strided requires stride_size (%ld) > op_size (%ld) so a gap "
                        "can exist between rows\n",
                        p.stride_size, p.op_size);
            MPI_Finalize();
            return 1;
        }
        if (p.block_size <= 0)
            p.block_size = p.stride_size / 2; /* default: half-open rows */
        /* Align down to a whole number of ops. */
        p.block_size = (p.block_size / p.op_size) * p.op_size;
        if (p.block_size < p.op_size)
            p.block_size = p.op_size;
        if (p.block_size >= p.stride_size)
        {
            if (rank == 0)
                fprintf(stderr,
                        "nd_strided requires block_size (%ld) < stride_size (%ld); otherwise "
                        "rows abut and the access collapses into plain sequential I/O\n",
                        p.block_size, p.stride_size);
            MPI_Finalize();
            return 1;
        }
    }
    if (p.mode != MODE_SETUP && p.mode != MODE_WORKLOAD)
    {
        if (rank == 0)
            fprintf(stderr, "mode must be 0 (setup) or 1 (workload)\n");
        MPI_Finalize();
        return 1;
    }

    /* Ensure work directory exists */
    if (mkdir(p.work_dir, 0755) < 0 && errno != EEXIST)
    {
        if (rank == 0)
            fprintf(stderr, "mkdir failed for %s: %s\n", p.work_dir, strerror(errno));
        MPI_Finalize();
        return 1;
    }

    /* Metadata workload: always runs as a single measured pass — no setup needed */
    if (strcmp(p.profile_name, "metadata_heavy") == 0)
    {
        if (p.mode == MODE_SETUP)
        {
            MPI_Finalize();
            return 0;
        }
        run_metadata_workload(&p);
        MPI_Finalize();
        return 0;
    }

    /* Setup mode: write files without Darshan, then exit */
    if (p.mode == MODE_SETUP)
    {
        run_setup(&p);
        MPI_Finalize();
        return 0;
    }

    /* Workload mode: measured run */
    long total_read_ops = (long)(p.num_ops * p.read_ratio);
    long total_write_ops = p.num_ops - total_read_ops;

    /* Distribute ops across files */
    long ops_per_file = p.num_ops / p.num_files;
    long read_ops_per_file = (long)(ops_per_file * p.read_ratio);
    long write_ops_per_file = ops_per_file - read_ops_per_file;

    for (int f = 0; f < p.num_files; f++)
    {
        char filepath[2048];
        snprintf(filepath, sizeof(filepath), "%s/workload_%s_f%d",
                 p.work_dir, p.profile_name, f);

        /* Last file absorbs any remainder from integer division */
        long r_ops = (f == p.num_files - 1)
                         ? total_read_ops - read_ops_per_file * (p.num_files - 1)
                         : read_ops_per_file;
        long w_ops = (f == p.num_files - 1)
                         ? total_write_ops - write_ops_per_file * (p.num_files - 1)
                         : write_ops_per_file;

        run_file_workload(&p, filepath, w_ops, r_ops);
    }

    /* Cleanup: remove workload files after the measured run */
    for (int f = 0; f < p.num_files; f++)
    {
        char filepath[2048];
        snprintf(filepath, sizeof(filepath), "%s/workload_%s_f%d",
                 p.work_dir, p.profile_name, f);
        unlink(filepath);
    }

    MPI_Finalize();
    return 0;
}
