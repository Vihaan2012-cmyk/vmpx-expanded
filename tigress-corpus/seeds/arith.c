/* Seed: integer arithmetic + bit mixing. Exercises Virtualize and
 * EncodeArithmetic (MBA). Deterministic: output depends only on argv. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t mix(uint64_t x, uint64_t y) {
    uint64_t a = (x ^ y) + 2 * (x & y);      /* == x + y, an MBA identity */
    uint64_t b = (x | y) - (x & y);          /* == x ^ y */
    uint64_t c = a * 0x9E3779B97F4A7C15ull;  /* a wrapping multiply */
    return (c ^ (c >> 29)) + b;
}

int main(int argc, char **argv) {
    uint64_t x = argc > 1 ? strtoull(argv[1], 0, 0) : 0x1234;
    uint64_t y = argc > 2 ? strtoull(argv[2], 0, 0) : 0x5678;
    uint64_t r = 0;
    for (int i = 0; i < 16; i++) r = mix(r + i, x) ^ mix(y, r);
    printf("%016llx\n", (unsigned long long)r);
    return 0;
}
