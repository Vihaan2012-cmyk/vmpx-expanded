/* Seed: loop-carried FNV-1a hash. Exercises Virtualize (loops) and
 * EncodeArithmetic. Deterministic. Target: hash */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

uint64_t hash(const char *s) {
    uint64_t h = 1469598103934665603ull;
    while (*s) { h ^= (uint8_t)*s++; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char **argv) {
    const char *s = argc > 1 ? argv[1] : "the quick brown fox";
    uint64_t h = hash(s);
    for (int i = 0; i < 1000; i++) {
        char buf[32]; snprintf(buf, sizeof buf, "%llu", (unsigned long long)(h + i)); h ^= hash(buf);
    }
    printf("%016llx\n", (unsigned long long)h);
    return 0;
}
