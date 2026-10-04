// Checker for matrix transpose.
//   check_trans <input.txt> <output.txt>
// input : "M N", then M*N values (M rows x N cols, row-major)
// output: what trans.exe writes -- N lines of M values, no header
// Checks the shape line by line, then out[R][C] == in[C][R] exactly
// (gen_trans values are fp16-exact, so any difference is a kernel bug).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool slurp(const char* path, std::string& s)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    _fseeki64(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    s.resize((size_t)n);
    size_t got = n ? fread(&s[0], 1, (size_t)n, f) : 0;
    fclose(f);
    s.resize(got);
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: check_trans <input.txt> <output.txt>\n");
        return 1;
    }
    std::string si, so;
    if (!slurp(argv[1], si)) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    if (!slurp(argv[2], so)) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }

    // ---- input ----
    char* t = &si[0];
    char* e;
    long M = strtol(t, &e, 10); t = e;
    long N = strtol(t, &e, 10); t = e;
    if (M <= 0 || N <= 0) { fprintf(stderr, "bad header in %s\n", argv[1]); return 1; }
    std::vector<float> in;
    in.reserve((size_t)M * N);
    for (;;) {
        float v = strtof(t, &e);
        if (e == t) break;
        in.push_back(v);
        t = e;
    }
    if ((long long)in.size() != (long long)M * N) {
        fprintf(stderr, "input has %zu values, header says %ld x %ld\n", in.size(), M, N);
        return 1;
    }

    // ---- output, line by line ----
    std::vector<float> out;
    out.reserve((size_t)M * N);
    long lines = 0, bad_width = 0, first_bad_line = -1, first_bad_count = 0;
    char* q = &so[0];
    char* end = q + so.size();
    while (q < end) {
        char* nl = (char*)memchr(q, '\n', end - q);
        if (nl) *nl = '\0';                       // stop strtof at the line end
        long cnt = 0;
        char* u = q;
        for (;;) {
            float v = strtof(u, &e);
            if (e == u) break;
            out.push_back(v);
            ++cnt;
            u = e;
        }
        if (cnt > 0) {
            if (cnt != M && bad_width++ == 0) { first_bad_line = lines + 1; first_bad_count = cnt; }
            ++lines;
        }
        q = nl ? nl + 1 : end;
    }

    printf("  input %ld x %ld  ->  expect %ld x %ld  |  got %ld lines", M, N, N, M, lines);
    bool ok = true;
    if (lines != N || bad_width) {
        ok = false;
        printf(" (shape BAD");
        if (bad_width) printf(", %ld lines not %ld wide, first: line %ld has %ld", bad_width, M, first_bad_line, first_bad_count);
        printf(")");
    }
    if ((long long)out.size() != (long long)M * N) {
        printf("  |  %zu values, expected %lld  FAIL\n", out.size(), (long long)M * N);
        return 1;
    }

    long long bad = 0;
    long fr = -1, fc = -1;
    float fg = 0, fw = 0;
    for (long R = 0; R < N; ++R)
        for (long C = 0; C < M; ++C) {
            float g = out[(size_t)R * M + C], w = in[(size_t)C * N + R];
            if (g != w && bad++ == 0) { fr = R; fc = C; fg = g; fw = w; }
        }
    if (bad) ok = false;
    printf("  |  mismatches %lld", bad);
    if (bad) printf(" (first at R=%ld C=%ld: got %g, want %g)", fr, fc, fg, fw);
    printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
