// Test-data generator for RMSNorm.
//   gen_rms <M> <N> <seed> <out.txt>
// Format:  line 1 = "M N",  then M lines of N values.
// Values sit on a 0.25 grid in [-4, 4] so every one is exactly representable
// in fp16 -- the reader's float->half conversion is then lossless and any
// mismatch the checker reports comes from the kernel, not from the input.
// N must be even (the kernel stores output as half2).
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: gen_rms <M> <N> <seed> <out.txt>\n");
        return 1;
    }
    long M = atol(argv[1]), N = atol(argv[2]);
    unsigned seed = (unsigned)atol(argv[3]);
    const char* out = argv[4];
    if (M <= 0 || N <= 0) { fprintf(stderr, "M and N must be positive\n"); return 1; }
    if (N % 2 != 0)       { fprintf(stderr, "N must be even\n");           return 1; }

    // 33 values: -4.00, -3.75, ... 3.75, 4.00
    char tbl[33][8]; int len[33];
    for (int i = 0; i < 33; ++i)
        len[i] = snprintf(tbl[i], sizeof tbl[i], "%g", (i - 16) * 0.25);

    FILE* f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", out); return 1; }
    static char buf[1 << 20];
    size_t p = 0;
    p += snprintf(buf + p, sizeof buf - p, "%ld %ld\n", M, N);

    unsigned s = seed ? seed : 1u;
    for (long r = 0; r < M; ++r) {
        for (long c = 0; c < N; ++c) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;     // xorshift32
            int k = (int)(s % 33u);
            if (p + 16 > sizeof buf) { fwrite(buf, 1, p, f); p = 0; }
            memcpy(buf + p, tbl[k], len[k]); p += len[k];
            buf[p++] = (c + 1 == N) ? '\n' : ' ';
        }
    }
    fwrite(buf, 1, p, f);
    fclose(f);
    printf("wrote %s : M=%ld N=%ld (%ld values)\n", out, M, N, M * N);
    return 0;
}
