// Checker for RMSNorm output.
//   check_rms <input.txt> <output.txt> [rel_tol]
// Recomputes  y[r][c] = x[r][c] * rsqrt(mean(x[r]^2) + 1e-6)  in double and
// compares against the kernel's fp16 output.  Default tolerance 2e-3 is a few
// times the fp16 rounding step (2^-11 = 4.9e-4), so real bugs still show up.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace std;

static vector<double> slurp(const char* path, long* M, long* N, bool head)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    vector<char> raw(sz + 1);
    if (fread(raw.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(1); }
    raw[sz] = 0; fclose(f);
    char* p = raw.data();
    if (head) { *M = strtol(p, &p, 10); *N = strtol(p, &p, 10); }
    vector<double> v; v.reserve(head ? (size_t)(*M) * (*N) : 1024);
    for (;;) { char* q; double d = strtod(p, &q); if (q == p) break; v.push_back(d); p = q; }
    return v;
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: check_rms <input.txt> <output.txt> [rel_tol]\n"); return 1; }
    double tol = argc > 3 ? atof(argv[3]) : 2e-3;
    long M = 0, N = 0, d1 = 0, d2 = 0;
    vector<double> in  = slurp(argv[1], &M, &N, true);
    vector<double> got = slurp(argv[2], &d1, &d2, false);

    if ((long)in.size() != M * N) {
        printf("FAIL: input has %zu values, header says M*N = %ld\n", in.size(), M * N);
        return 1;
    }
    if ((long)got.size() != M * N) {
        printf("FAIL: output has %zu values, expected %ld\n", got.size(), M * N);
        return 1;
    }

    double worst = 0; long bad = 0; long wr = -1, wc = -1;
    for (long r = 0; r < M; ++r) {
        double s = 0;
        for (long c = 0; c < N; ++c) { double x = in[r * N + c]; s += x * x; }
        double scale = 1.0 / sqrt(s / (double)N + 1e-6);
        for (long c = 0; c < N; ++c) {
            double ref = in[r * N + c] * scale;
            double g   = got[r * N + c];
            double e   = fabs(g - ref) / (fabs(ref) + 1e-3);
            if (e > worst) { worst = e; wr = r; wc = c; }
            if (e > tol) ++bad;
        }
    }
    printf("M=%ld N=%ld  max rel err = %.4e  (row %ld col %ld)  bad = %ld / %ld  tol = %g\n",
           M, N, worst, wr, wc, bad, M * N, tol);
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
