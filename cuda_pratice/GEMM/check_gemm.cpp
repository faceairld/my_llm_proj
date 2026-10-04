// Verify gemm output against a double-precision CPU reference.
//
//   check_gemm <input.txt> <output.txt> [samples]
//
// Recomputes a random sample of output elements instead of the whole matrix,
// so it stays fast even at 4096.  Reference layout matches read_file1:
//   A  is M x K,  B is stored as N x K (B transposed),  C is M x N.
//
// An element is called EXACT when |ref - got| <= half an fp16 ulp at that
// magnitude, i.e. the GPU value is the correctly rounded fp16 of the true
// result.  With the 0.25-grid inputs from gen_gemm every element should be.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
using namespace std;

static char*  g_buf;
static char*  g_ptr;
static size_t g_len;

static bool slurp(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    fseek(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    fseek(f, 0, SEEK_SET);
    g_buf = (char*)malloc((size_t)n + 1);
    if (!g_buf) { fprintf(stderr, "out of memory for %s (%lld bytes)\n", path, n); fclose(f); return false; }
    g_len = fread(g_buf, 1, (size_t)n, f);
    g_buf[g_len] = 0;
    g_ptr = g_buf;
    fclose(f);
    return true;
}

// values are on a 0.0625 grid with at most 4 decimals; a plain parser is enough
static bool next_num(double& out)
{
    while (*g_ptr && (unsigned char)*g_ptr <= ' ') ++g_ptr;
    if (!*g_ptr) return false;
    char* end;
    out = strtod(g_ptr, &end);
    if (end == g_ptr) return false;
    g_ptr = end;
    return true;
}

static bool read_block(vector<float>& dst, size_t n, const char* name)
{
    dst.resize(n);
    double v;
    for (size_t i = 0; i < n; ++i) {
        if (!next_num(v)) { fprintf(stderr, "short read in %s at %zu / %zu\n", name, i, n); return false; }
        dst[i] = (float)v;
    }
    return true;
}

static double half_ulp(double v)
{
    double a = fabs(v);
    if (a < 6.103515625e-5) return 3.0e-8;              // subnormal fp16 region
    int e; frexp(a, &e);                                // a in [2^(e-1), 2^e)
    return ldexp(1.0, e - 1 - 11);                      // half of 2^(e-1-10)
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <input.txt> <output.txt> [samples]\n", argv[0]); return 1; }
    int samples = argc > 3 ? atoi(argv[3]) : 4000;

    if (!slurp(argv[1])) return 1;
    double dM, dN, dK, dAl, dBe;
    if (!(next_num(dM) && next_num(dN) && next_num(dK) && next_num(dAl) && next_num(dBe))) {
        fprintf(stderr, "bad header\n"); return 1;
    }
    int M = (int)dM, N = (int)dN, K = (int)dK;
    double alpha = dAl, beta = dBe;
    printf("M=%d N=%d K=%d alpha=%g beta=%g\n", M, N, K, alpha, beta);

    vector<float> A, Bt, C, O;
    if (!read_block(A,  (size_t)M * K, "A"))  return 1;
    if (!read_block(Bt, (size_t)N * K, "B"))  return 1;
    if (!read_block(C,  (size_t)M * N, "C"))  return 1;
    free(g_buf);

    if (!slurp(argv[2])) return 1;
    if (!read_block(O, (size_t)M * N, "output")) return 1;
    { double extra; if (next_num(extra)) printf("WARNING: output file has MORE than %lld values\n", (long long)M * N); }
    free(g_buf);

    srand(12345);
    double worst_abs = 0, worst_rel = 0;
    long long inexact = 0, wrong = 0;
    int wm = -1, wn = -1;
    for (int s = 0; s < samples; ++s) {
        int m = rand() % M, n = rand() % N;
        const float* a  = &A[(size_t)m * K];
        const float* bt = &Bt[(size_t)n * K];
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += (double)a[k] * (double)bt[k];
        double ref = alpha * acc + beta * (double)C[(size_t)m * N + n];
        double got = (double)O[(size_t)m * N + n];
        double e   = fabs(ref - got);
        if (e > half_ulp(ref) * 1.001) ++inexact;
        double rel = e / (fabs(ref) + 1e-9);
        if (rel > 1e-2) ++wrong;
        if (e > worst_abs) worst_abs = e;
        if (rel > worst_rel) { worst_rel = rel; wm = m; wn = n; }
    }
    printf("sampled %d elements\n", samples);
    printf("max abs err = %.6g\n", worst_abs);
    printf("max rel err = %.6g   (at m=%d n=%d)\n", worst_rel, wm, wn);
    printf("not correctly rounded : %lld / %d\n", inexact, samples);
    printf("clearly wrong (>1%%)   : %lld / %d\n", wrong, samples);
    printf("verdict: %s\n", (inexact == 0) ? "PASS (bit-exact)" : (wrong == 0 ? "PASS (within tolerance)" : "FAIL"));
    return 0;
}
