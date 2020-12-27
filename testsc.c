/*
 * testsc: run PuTTY's crypto primitives under instrumentation that
 * checks for cache and timing side channels.
 *
 * The idea is: cryptographic code should avoid leaking secret data
 * through timing information, or through traces of its activity left
 * in the caches.
 *
 * (This property is sometimes called 'constant-time', although really
 * that's a misnomer. It would be impossible to avoid the execution
 * time varying for any number of reasons outside the code's control,
 * such as the prior contents of caches and branch predictors,
 * temperature-based CPU throttling, system load, etc. And in any case
 * you don't _need_ the execution time to be literally constant: you
 * just need it to be independent of your secrets. It can vary as much
 * as it likes based on anything else.)
 *
 * To avoid this, you need to ensure that various aspects of the
 * code's behaviour do not depend on the secret data. The control
 * flow, for a start - no conditional branches based on secrets - and
 * also the memory access pattern (no using secret data as an index
 * into a lookup table). A couple of other kinds of CPU instruction
 * also can't be trusted to run in constant time: we check for
 * register-controlled shifts and hardware divisions. (But, again,
 * it's perfectly fine to _use_ those instructions in the course of
 * crypto code. You just can't use a secret as any time-affecting
 * operand.)
 *
 * This test program works by running the same crypto primitive
 * multiple times, with different secret input data. The relevant
 * details of each run is logged to a file via the DynamoRIO-based
 * instrumentation system living in the subdirectory test/sclog. Then
 * we check over all the files and ensure they're identical.
 *
 * This program itself (testsc) is built by the ordinary PuTTY
 * makefiles. But run by itself, it will do nothing useful: it needs
 * to be run under DynamoRIO, with the sclog instrumentation library.
 *
 * Here's an example of how I built it:
 *
 * Download the DynamoRIO source. I did this by cloning
 * https://github.com/DynamoRIO/dynamorio.git, and at the time of
 * writing this, 259c182a75ce80112bcad329c97ada8d56ba854d was the head
 * commit.
 *
 * In the DynamoRIO checkout:
 *
 *   mkdir build
 *   cd build
 *   cmake -G Ninja ..
 *   ninja
 *
 * Now set the shell variable DRBUILD to be the location of the build
 * directory you did that in. (Or not, if you prefer, but the example
 * build commands below will assume that that's where the DynamoRIO
 * libraries, headers and runtime can be found.)
 *
 * Then, in test/sclog:
 *
 *   cmake -G Ninja -DCMAKE_PREFIX_PATH=$DRBUILD/cmake .
 *   ninja
 *
 * Finally, to run the actual test, set SCTMP to some temp directory
 * you don't mind filling with large temp files (several GB at a
 * time), and in the main PuTTY source directory (assuming that's
 * where testsc has been built):
 *
 *   $DRBUILD/bin64/drrun -c test/sclog/libsclog.so -- ./testsc -O $SCTMP
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "defs.h"
#include "putty.h"
#include "ssh.h"
#include "misc.h"
#include "mpint.h"
#include "ecc.h"

static NORETURN PRINTF_LIKE(1, 2) void fatal_error(const char *p, ...)
{
    va_list ap;
    fprintf(stderr, "testsc: ");
    va_start(ap, p);
    vfprintf(stderr, p, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

void out_of_memory(void) { fatal_error("out of memory"); }
FILE *f_open(const Filename *filename, char const *mode, bool is_private)
{ unreachable("this is a stub needed to link, and should never be called"); }
void old_keyfile_warning(void)
{ unreachable("this is a stub needed to link, and should never be called"); }

/*
 * A simple deterministic PRNG, without any of the Fortuna
 * complexities, for generating test inputs in a way that's repeatable
 * between runs of the program, even if only a subset of test cases is
 * run.
 */
static uint64_t random_counter = 0;
static const char *random_seedstr = NULL;
static uint8_t random_buf[MAX_HASH_LEN];
static size_t random_buf_limit = 0;
static ssh_hash *random_hash;

static void random_seed(const char *seedstr)
{
    random_seedstr = seedstr;
    random_counter = 0;
    random_buf_limit = 0;
}

void random_read(void *vbuf, size_t size)
{
    assert(random_seedstr);
    uint8_t *buf = (uint8_t *)vbuf;
    while (size-- > 0) {
        if (random_buf_limit == 0) {
            ssh_hash_reset(random_hash);
            put_asciz(random_hash, random_seedstr);
            put_uint64(random_hash, random_counter);
            random_counter++;
            random_buf_limit = ssh_hash_alg(random_hash)->hlen;
            ssh_hash_digest(random_hash, random_buf);
        }
        *buf++ = random_buf[random_buf_limit--];
    }
}

/*
 * Macro that defines a function, and also a volatile function pointer
 * pointing to it. Callers indirect through the function pointer
 * instead of directly calling the function, to ensure that the
 * compiler doesn't try to get clever by eliminating the call
 * completely, or inlining it.
 *
 * This is used to mark functions that DynamoRIO will look for to
 * intercept, and also to inhibit inlining and unrolling where they'd
 * cause a failure of experimental control in the main test.
 */
#define VOLATILE_WRAPPED_DEFN(qualifier, rettype, fn, params)   \
    qualifier rettype fn##_real params;                         \
    qualifier rettype (*volatile fn) params = fn##_real;        \
    qualifier rettype fn##_real params

VOLATILE_WRAPPED_DEFN(, void, log_to_file, (const char *filename))
{
    /*
     * This function is intercepted by the DynamoRIO side of the
     * mechanism. We use it to send instructions to the DR wrapper,
     * namely, 'please start logging to this file' or 'please stop
     * logging' (if filename == NULL). But we don't have to actually
     * do anything in _this_ program - all the functionality is in the
     * DR wrapper.
     */
}

static const char *outdir = NULL;
char *log_filename(const char *basename, size_t index)
{
    return dupprintf("%s/%s.%04"SIZEu, outdir, basename, index);
}

static char *last_filename;
static const char *test_basename;
static size_t test_index = 0;
void log_start(void)
{
    last_filename = log_filename(test_basename, test_index++);
    log_to_file(last_filename);
}
void log_end(void)
{
    log_to_file(NULL);
    sfree(last_filename);
}

VOLATILE_WRAPPED_DEFN(, intptr_t, dry_run, (void))
{
    /*
     * This is another function intercepted by DynamoRIO. In this
     * case, DR overrides this function to return 0 rather than 1, so
     * we can use it as a check for whether we're running under
     * instrumentation, or whether this is just a dry run which goes
     * through the motions but doesn't expect to find any log files
     * created.
     */
    return 1;
}

VOLATILE_WRAPPED_DEFN(static, size_t, looplimit, (size_t x))
{
    /*
     * looplimit() is the identity function on size_t, but the
     * compiler isn't allowed to rely on it being that. I use it to
     * make loops in the test functions look less attractive to
     * compilers' unrolling heuristics.
     */
    return x;
}

#define TESTLIST(X)                             \
    X(safe_mem_clear)                           

static void test_safe_mem_clear(void)
{
    char *dec = snewn(256, char);
    char *x = (char*)((((size_t)dec >> 6) + 1) << 6);
    smemclr(dec, 256);
    for (size_t i = 0; i < 64; i++) {
        log_start();
        smemclr(x + i, 128);
        log_end();
    }
}

struct test {
    const char *testname;
    void (*testfn)(void);
};

static const struct test tests[] = {
#define STRUCT_TEST(X) { #X, test_##X },
TESTLIST(STRUCT_TEST)
#undef STRUCT_TEST
};

void dputs(const char *buf)
{
    fputs(buf, stderr);
}

int main(int argc, char **argv)
{
    bool doing_opts = true;
    const char *pname = argv[0];
    uint8_t tests_to_run[lenof(tests)];
    bool keep_outfiles = false;
    bool test_names_given = false;

    memset(tests_to_run, 1, sizeof(tests_to_run));
    random_hash = ssh_hash_new(&ssh_sha256);

    while (--argc > 0) {
        char *p = *++argv;

        if (p[0] == '-' && doing_opts) {
            if (!strcmp(p, "-O")) {
                if (--argc <= 0) {
                    fprintf(stderr, "'-O' expects a directory name\n");
                    return 1;
                }
                outdir = *++argv;
            } else if (!strcmp(p, "-k") || !strcmp(p, "--keep")) {
                keep_outfiles = true;
            } else if (!strcmp(p, "--")) {
                doing_opts = false;
            } else if (!strcmp(p, "--help")) {
                printf("  usage: drrun -c test/sclog/libsclog.so -- "
                       "%s -O <outdir>\n", pname);
                printf("options: -O <outdir>           "
                       "put log files in the specified directory\n");
                printf("         -k, --keep            "
                       "do not delete log files for tests that passed\n");
                printf("   also: --help                "
                       "display this text\n");
                return 0;
            } else {
                fprintf(stderr, "unknown command line option '%s'\n", p);
                return 1;
            }
        } else {
            if (!test_names_given) {
                test_names_given = true;
                memset(tests_to_run, 0, sizeof(tests_to_run));
            }
            bool found_one = false;
            for (size_t i = 0; i < lenof(tests); i++) {
                if (wc_match(p, tests[i].testname)) {
                    tests_to_run[i] = 1;
                    found_one = true;
                }
            }
            if (!found_one) {
                fprintf(stderr, "no test name matched '%s'\n", p);
                return 1;
            }
        }
    }

    bool is_dry_run = dry_run();

    if (is_dry_run) {
        printf("Dry run (DynamoRIO instrumentation not detected)\n");
    } else {
        /* Print the address of main() in this run. The idea is that
         * if this image is compiled to be position-independent, then
         * PC values in the logs won't match the ones you get if you
         * disassemble the binary, so it'll be harder to match up the
         * log messages to the code. But if you know the address of a
         * fixed (and not inlined) function in both worlds, you can
         * find out the offset between them. */
        printf("Live run, main = %p\n", (void *)main);

        if (!outdir) {
            fprintf(stderr, "expected -O <outdir> option\n");
            return 1;
        }
        printf("Will write log files to %s\n", outdir);
    }

    for (size_t i = 0; i < lenof(tests); i++) {
        if (!tests_to_run[i])
            continue;
        const struct test *test = &tests[i];
        printf("Running test %s ... ", test->testname);
        fflush(stdout);

        random_seed(test->testname);
        test_basename = test->testname;
        test_index = 0;

        test->testfn();

        if (is_dry_run) {
            printf("dry run done\n");
            continue;                  /* test files won't exist anyway */
        }

        if (test_index < 2) {
            printf("FAIL: test did not generate multiple output files\n");
            goto test_done;
        }

        for (size_t i = 0; i < test_index; i++) {
            char *nextfile = log_filename(test_basename, i);
            FILE *nextfp = fopen(nextfile, "rb");
            if (!nextfp) {
                printf("ERR: %s: open: %s\n", nextfile, strerror(errno));
                goto test_done;
            }

            char bufn[4096];
            while (true) {
                size_t rn = fread(bufn, 1, sizeof(bufn), nextfp);
                printf("%s\n", bufn);
            }
            fclose(nextfp);
            sfree(nextfile);
        }
    }

    ssh_hash_free(random_hash);
    return 0;
}
