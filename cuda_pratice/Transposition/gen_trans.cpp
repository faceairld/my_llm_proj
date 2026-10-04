// Test-data generator for matrix transpose.
//   gen_trans <M> <N> <seed> <out.txt>
// Format:  line 1 = "M N",  then M lines of N values.
// Values are random integers in [-2048, 2048]. Every one is exact in fp16,
// so the reader's float->half conversion is lossless and the checker can
// compare exactly. With 4097 distinct values, an index mix-up in the kernel
// shows up as a mismatch almost everywhere instead of hiding behind repeats.
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: gen_trans <M> <N> <seed> <out.txt>\n");
        return 1;
    }
    long M = atol(argv[1]), N = atol(argv[2]);
    unsigned s = (unsigned)atol(argv[3]);
    const char* out = argv[4];
    if (M <= 0 || N <= 0) { fprintf(stderr, "M and N must be positive\n"); return 1; }
    if (s == 0) s = 1;

    FILE* f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", out); return 1; }
    static char buf[1 << 20];
    size_t p = 0;
    p += snprintf(buf + p, sizeof buf - p, "%ld %ld\n", M, N);

    for (long r = 0; r < M; ++r) {
        for (long c = 0; c < N; ++c) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;     // xorshift32
            int v = (int)(s % 4097u) - 2048;
            if (p + 16 > sizeof buf) { fwrite(buf, 1, p, f); p = 0; }
            p += snprintf(buf + p, sizeof buf - p, "%d", v);
            buf[p++] = (c + 1 == N) ? '\n' : ' ';
        }
    }
    fwrite(buf, 1, p, f);
    fclose(f);
    printf("wrote %s : M=%ld N=%ld (%ld values)\n", out, M, N, M * N);
    return 0;
}
