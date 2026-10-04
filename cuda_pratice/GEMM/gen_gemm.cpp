// Test-input generator for gemm.cu
//
// File layout (whitespace-separated, read back with operator>>):
//     M N K alpha beta
//     A : M rows x K cols   (row-major, one row per line)
//     B : N rows x K cols   (row-major)  <-- this is B TRANSPOSED
//     C : M rows x N cols   (row-major)
//
// Note on B: the math is  D = alpha * A(MxK) * B(KxN) + beta * C(MxN),
// but read_file1 loads B as N*K and the kernel indexes it as
// input_b[n * K + k], i.e. the file stores B column-by-column = B^T.
//
// All entries are multiples of 0.25 in [-1, 1], so every value is exact in
// fp16, every product is a multiple of 0.0625, and the fp32 accumulation is
// exact as long as |result| < 128.  A double-precision CPU reference should
// therefore match the GPU output bit for bit.
//
// usage: gen_gemm <M> <N> <K> <alpha> <beta> <seed> <out.txt>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace std;

static const char* kVal[9] = {
    "-1 ", "-0.75 ", "-0.5 ", "-0.25 ", "0 ", "0.25 ", "0.5 ", "0.75 ", "1 "
};
static const int kLen[9] = { 3, 6, 5, 6, 2, 5, 4, 5, 2 };

static unsigned long long g_state;
static inline unsigned rnd()
{
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return (unsigned)(g_state >> 33);
}

static void emit(FILE* f, long long rows, long long cols, string& buf)
{
    for (long long r = 0; r < rows; ++r) {
        buf.clear();
        for (long long c = 0; c < cols; ++c) {
            int v = (int)(rnd() % 9u);
            buf.append(kVal[v], kLen[v]);
        }
        buf.push_back('\n');
        fwrite(buf.data(), 1, buf.size(), f);
    }
}

int main(int argc, char** argv)
{
    if (argc != 8) {
        fprintf(stderr, "usage: %s <M> <N> <K> <alpha> <beta> <seed> <out.txt>\n", argv[0]);
        return 1;
    }
    int   M     = atoi(argv[1]);
    int   N     = atoi(argv[2]);
    int   K     = atoi(argv[3]);
    float alpha = (float)atof(argv[4]);
    float beta  = (float)atof(argv[5]);
    g_state     = (unsigned long long)strtoull(argv[6], 0, 10) * 6364136223846793005ULL + 1442695040888963407ULL;
    if (g_state == 0) g_state = 88172645463325252ULL;

    if (M <= 0 || N <= 0 || K <= 0) { fprintf(stderr, "M, N, K must be positive\n"); return 1; }

    FILE* f = fopen(argv[7], "wb");          // "wb": no CRLF translation
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[7]); return 1; }

    fprintf(f, "%d %d %d %g %g\n", M, N, K, alpha, beta);

    string buf;
    buf.reserve(1 << 16);
    emit(f, M, K, buf);      // A   M x K
    emit(f, N, K, buf);      // B^T N x K
    emit(f, M, N, buf);      // C   M x N

    long long total = (long long)M * K + (long long)N * K + (long long)M * N;
    long long bytes = ftell(f);
    fclose(f);
    printf("%-18s  M=%-5d N=%-5d K=%-5d alpha=%g beta=%g   %lld values, %.2f MB\n",
           argv[7], M, N, K, alpha, beta, total, bytes / 1048576.0);
    return 0;
}
