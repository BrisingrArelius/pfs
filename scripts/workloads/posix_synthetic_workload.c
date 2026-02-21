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
 *
 * access_pattern: 0 = sequential, 1 = random, 2 = strided
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
    int mode; /* MODE_SETUP or MODE_WORKLOAD */
} Profile;

/* -------------------------------------------------------------------------
 * Utility: allocate and fill a buffer with deterministic data
 * ---------------------------------------------------------------------- */
static char *make_buffer(long size)
{
    char *buf = malloc(size);
    if (!buf)
    {
        fprintf(stderr, "malloc failed for buffer of size %ld\n", size);
        exit(1);
    }
    for (long i = 0; i < size; i++)
        buf[i] = (char)(i & 0xFF);
    return buf;
}

/* -------------------------------------------------------------------------
 * Utility: retry write until all 'size' bytes are written.
 * Returns 0 on success, -1 on error (errno set by underlying write).
 * ---------------------------------------------------------------------- */
static int full_write(int fd, const char *buf, long size)
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
 * Utility: retry read until all 'size' bytes are read or EOF.
 * Returns bytes actually read (< size only on EOF), -1 on error.
 * ---------------------------------------------------------------------- */
static long full_read(int fd, char *buf, long size)
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
 * Utility: random aligned block offset within [0, file_size - op_size].
 * Guarantees offset + op_size <= file_size.
 * ---------------------------------------------------------------------- */
static long random_offset(long file_size, long op_size, unsigned int *seed)
{
    /* Number of complete blocks that fit, starting from offset 0 */
    long num_blocks = (file_size - op_size) / op_size + 1;
    if (num_blocks <= 0)
        return 0;
    long block = (long)(rand_r(seed) % (unsigned long)num_blocks);
    return block * op_size;
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
 *   seed           — PRNG state (in/out)
 * Returns 0 on success, -1 if a fatal I/O error occurred (file should be
 * abandoned — Darshan counters for it will be incomplete).
 * ---------------------------------------------------------------------- */
static int do_write_phase(int fd, const Profile *p, long ops_in_phase,
                          long file_size, long *write_offset,
                          long *stride_cursor, long *global_op,
                          char *buf, unsigned int *seed)
{
    for (long i = 0; i < ops_in_phase; i++)
    {
        long offset;

        if (p->access_pattern == PATTERN_SEQUENTIAL)
        {
            offset = *write_offset;
            *write_offset += p->op_size;
        }
        else if (p->access_pattern == PATTERN_RANDOM)
        {
            /* Random overwrite within the fixed file extent */
            offset = random_offset(file_size, p->op_size, seed);
        }
        else /* PATTERN_STRIDED */
        {
            /* Continuous stride index across phases.
             * Round down to op_size boundary so the offset is always
             * block-aligned even when stride_size is not a multiple of op_size. */
            long raw = (*stride_cursor * p->stride_size) %
                       (file_size > 0 ? file_size : p->stride_size);
            offset = (raw / p->op_size) * p->op_size;
            (*stride_cursor)++;
        }

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
 *   seed           — PRNG state (in/out)
 * Returns 0 on success, -1 if a fatal I/O error occurred.
 * ---------------------------------------------------------------------- */
static int do_read_phase(int fd, const Profile *p, long ops_in_phase,
                         long file_size, long *read_offset,
                         long *stride_cursor, long *global_op,
                         char *buf, unsigned int *seed)
{
    for (long i = 0; i < ops_in_phase; i++)
    {
        long offset;

        if (p->access_pattern == PATTERN_SEQUENTIAL)
        {
            offset = *read_offset;
            *read_offset += p->op_size;
            if (*read_offset + p->op_size > file_size)
                *read_offset = 0; /* wrap */
        }
        else if (p->access_pattern == PATTERN_RANDOM)
        {
            offset = random_offset(file_size, p->op_size, seed);
        }
        else /* PATTERN_STRIDED */
        {
            long raw = (*stride_cursor * p->stride_size) %
                       (file_size > 0 ? file_size : p->stride_size);
            offset = (raw / p->op_size) * p->op_size;
            (*stride_cursor)++;
        }

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

        int fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC, 0644);
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
        printf("[setup] %s written (%ld ops x %ld bytes)\n",
               filepath, w_ops, p->op_size);
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

    /* Open flags */
    int flags;
    if (p->mode == MODE_WORKLOAD && p->read_ratio >= 1.0)
        flags = O_RDONLY;
    else
        flags = O_RDWR | O_CREAT | O_TRUNC;

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
                               &write_stride_cursor, &global_op, buf, &seed) < 0)
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
                              &read_stride_cursor, &global_op, buf, &seed) < 0)
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
                    "Usage: %s <profile_name> <read_ratio> <access_pattern (0|1|2)>\n"
                    "          <stride_size> <op_size> <num_ops> <num_files>\n"
                    "          <num_phases> <fsync_interval> <work_dir> <mode (0|1)>\n"
                    "  mode 0 = setup (write files only, no Darshan — pure-read profiles only)\n"
                    "  mode 1 = workload (measured run, Darshan attached)\n",
                    argv[0]);
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

    /* Validate */
    if (p.op_size <= 0 || p.num_ops <= 0 || p.num_phases < 1)
    {
        if (rank == 0)
            fprintf(stderr, "Invalid parameters: op_size, num_ops must be >0; num_phases >= 1\n");
        MPI_Finalize();
        return 1;
    }
    if (p.access_pattern == PATTERN_STRIDED && p.stride_size <= 0)
    {
        if (rank == 0)
            fprintf(stderr, "stride_size must be >0 for strided access pattern\n");
        MPI_Finalize();
        return 1;
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
            if (rank == 0)
                printf("[setup] metadata_heavy has no setup phase — nothing to do.\n");
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

    printf("[workload] %s read_ops=%ld write_ops=%ld op_size=%ld num_files=%d "
           "num_phases=%d fsync_interval=%d access_pattern=%d\n",
           p.profile_name, total_read_ops, total_write_ops,
           p.op_size, p.num_files, p.num_phases,
           p.fsync_interval, p.access_pattern);

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
