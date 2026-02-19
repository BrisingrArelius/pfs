/*
 * synthetic_workload.c
 *
 * Simulates configurable I/O workloads for Darshan instrumentation.
 * All parameters are passed as CLI arguments by run_workloads.py.
 *
 * Usage:
 *   ./synthetic_workload <profile_name> <read_ratio> <access_pattern>
 *                        <stride_size> <op_size> <num_ops> <num_files>
 *                        <num_phases> <fsync_interval> <work_dir> <mode>
 *
 * access_pattern: 0 = sequential, 1 = random, 2 = strided
 *
 * mode:
 *   0 = SETUP   — write files to disk without Darshan attached.
 *                 Run this first for any profile that involves reads,
 *                 so the files exist before the measured run begins.
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
 * Utility: simple LCG random offset within [0, max_offset]
 * aligned to op_size boundaries
 * ---------------------------------------------------------------------- */
static long random_offset(long max_offset, long op_size, unsigned int *seed)
{
    long range = max_offset / op_size;
    if (range <= 0)
        return 0;
    long block = (long)(rand_r(seed) % range);
    return block * op_size;
}

/* -------------------------------------------------------------------------
 * Single-file workload: write phase
 * ---------------------------------------------------------------------- */
static void do_write_phase(int fd, const Profile *p, long ops_in_phase,
                           long *write_offset, unsigned int *seed)
{
    char *buf = make_buffer(p->op_size);
    long file_size = p->num_ops * p->op_size;

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
            /* random write: random chunk size append — seek to random position
             * within already-written extent, then write */
            long written_so_far = *write_offset;
            if (written_so_far == 0)
                offset = 0;
            else
                offset = random_offset(written_so_far, p->op_size, seed);
            *write_offset += p->op_size; /* track total bytes written */
        }
        else
        {
            /* strided */
            offset = (i * p->stride_size) % (file_size > 0 ? file_size : p->stride_size);
        }

        if (lseek(fd, offset, SEEK_SET) < 0)
        {
            perror("lseek (write)");
            free(buf);
            return;
        }
        if (write(fd, buf, p->op_size) < 0)
        {
            perror("write");
            free(buf);
            return;
        }

        if (p->fsync_interval > 0 && (i + 1) % p->fsync_interval == 0)
            fsync(fd);
    }

    free(buf);
}

/* -------------------------------------------------------------------------
 * Single-file workload: read phase
 * ---------------------------------------------------------------------- */
static void do_read_phase(int fd, const Profile *p, long ops_in_phase,
                          long *read_offset, long file_size, unsigned int *seed)
{
    char *buf = make_buffer(p->op_size);

    for (long i = 0; i < ops_in_phase; i++)
    {
        long offset;

        if (p->access_pattern == PATTERN_SEQUENTIAL)
        {
            offset = *read_offset;
            *read_offset += p->op_size;
            if (*read_offset + p->op_size > file_size)
                *read_offset = 0;
        }
        else if (p->access_pattern == PATTERN_RANDOM)
        {
            offset = random_offset(file_size, p->op_size, seed);
        }
        else
        {
            /* strided */
            offset = (i * p->stride_size) % (file_size > 0 ? file_size : p->stride_size);
        }

        if (lseek(fd, offset, SEEK_SET) < 0)
        {
            perror("lseek (read)");
            free(buf);
            return;
        }
        if (read(fd, buf, p->op_size) < 0)
        {
            perror("read");
            free(buf);
            return;
        }
    }

    free(buf);
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
            if (write(fd, buf, p->op_size) < 0)
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

    /* In workload mode for pure-read profiles, files already exist from setup.
     * Open without truncating. For write/mixed profiles, create fresh. */
    int flags;
    long file_size;

    if (p->mode == MODE_WORKLOAD && p->read_ratio >= 1.0)
    {
        /* File was pre-populated by setup — open read-only, don't touch it */
        flags = O_RDONLY;
        file_size = p->num_ops * p->op_size;
    }
    else
    {
        flags = O_RDWR | O_CREAT | O_TRUNC;
        file_size = (total_write_ops > 0 ? total_write_ops : total_read_ops) * p->op_size;
    }

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

    /* Divide ops evenly across phases, alternating write→read */
    long write_ops_per_write_phase = 0;
    long read_ops_per_read_phase = 0;
    int write_phases = 0;
    int read_phases = 0;

    /* Count how many of the num_phases are write vs read phases.
     * Phase 1 is always write. Phases alternate W, R, W, R, ...
     * Exception: read_ratio == 1.0 in WORKLOAD mode → no writes at all,
     *            file already exists from setup. All phases are reads. */
    if (p->mode == MODE_WORKLOAD && p->read_ratio >= 1.0)
    {
        /* Pure read workload mode: every phase is a read phase */
        read_phases = p->num_phases;
        write_phases = 0;
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

    if (write_phases > 0)
        write_ops_per_write_phase = total_write_ops / write_phases;
    if (read_phases > 0)
        read_ops_per_read_phase = total_read_ops / read_phases;

    long write_offset = 0;
    long read_offset = 0;
    long write_phase_count = 0;
    long read_phase_count = 0;

    for (int ph = 0; ph < p->num_phases; ph++)
    {
        /* In pure-read workload mode all phases are read phases */
        int is_write_phase;
        if (p->mode == MODE_WORKLOAD && p->read_ratio >= 1.0)
        {
            is_write_phase = 0;
        }
        else
        {
            is_write_phase = (ph % 2 == 0);
        }

        if (is_write_phase && write_phase_count < write_phases)
        {
            /* Last write phase gets any remainder ops */
            long ops = (write_phase_count == write_phases - 1)
                           ? (total_write_ops - write_ops_per_write_phase * (write_phases - 1))
                           : write_ops_per_write_phase;
            do_write_phase(fd, p, ops, &write_offset, &seed);
            write_phase_count++;
        }
        else if (!is_write_phase && read_phase_count < read_phases)
        {
            long ops = (read_phase_count == read_phases - 1)
                           ? (total_read_ops - read_ops_per_read_phase * (read_phases - 1))
                           : read_ops_per_read_phase;
            do_read_phase(fd, p, ops, &read_offset, file_size, &seed);
            read_phase_count++;
        }
    }

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
        if (write(fd, buf, p->op_size) < 0)
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
    if (argc < 12)
    {
        fprintf(stderr,
                "Usage: %s <profile_name> <read_ratio> <access_pattern (0|1|2)>\n"
                "          <stride_size> <op_size> <num_ops> <num_files>\n"
                "          <num_phases> <fsync_interval> <work_dir> <mode (0|1)>\n"
                "  mode 0 = setup (write files, no Darshan)\n"
                "  mode 1 = workload (measured run, Darshan attached)\n",
                argv[0]);
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
        fprintf(stderr, "Invalid parameters: op_size, num_ops must be >0; num_phases >= 1\n");
        return 1;
    }
    if (p.access_pattern == PATTERN_STRIDED && p.stride_size <= 0)
    {
        fprintf(stderr, "stride_size must be >0 for strided access pattern\n");
        return 1;
    }
    if (p.mode != MODE_SETUP && p.mode != MODE_WORKLOAD)
    {
        fprintf(stderr, "mode must be 0 (setup) or 1 (workload)\n");
        return 1;
    }

    /* Ensure work directory exists */
    mkdir(p.work_dir, 0755);

    /* Metadata workload: always runs as a single measured pass — no setup needed */
    if (strcmp(p.profile_name, "metadata_heavy") == 0)
    {
        if (p.mode == MODE_SETUP)
        {
            printf("[setup] metadata_heavy has no setup phase — nothing to do.\n");
            return 0;
        }
        run_metadata_workload(&p);
        return 0;
    }

    /* Setup mode: write files without Darshan, then exit */
    if (p.mode == MODE_SETUP)
    {
        run_setup(&p);
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

    return 0;
}
