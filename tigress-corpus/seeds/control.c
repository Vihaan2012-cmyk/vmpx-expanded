/* Seed: nested control flow + small state machine. Exercises Flatten and
 * AddOpaque. Deterministic: output depends only on argv. Target: classify */
#include <stdio.h>
#include <stdlib.h>

int classify(int x) {
    int r = 0;
    for (int i = 0; i < x % 7 + 1; i++) {
        if ((x + i) % 3 == 0) r += 2;
        else if ((x ^ i) & 1) r -= 1;
        else switch (i & 3) { case 0: r += 5; break; case 1: r *= 2; break; default: r -= 3; }
    }
    return r;
}

int main(int argc, char **argv) {
    int x = argc > 1 ? atoi(argv[1]) : 42, acc = 0;
    for (int i = 0; i < 20; i++) acc += classify(x + i);
    printf("%d\n", acc);
    return 0;
}
