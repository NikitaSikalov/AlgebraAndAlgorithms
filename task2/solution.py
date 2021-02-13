#! /usr/bin/env python3

def main(n):
    gates = list()
    outputs = list(range(2 * n + 2))
    N = 3 * n
    for k in range(n):
        gates.append('GATE {} NOT {}'.format(N, k))
        gates.append('GATE {} AND {} {}'.format(N + 1, n + k, 2 * n + k))
        gates.append('GATE {} NOT {}'.format(N + 2, N + 1))
        gates.append('GATE {} OR {} {}'.format(N + 3, n + k, 2 * n + k))
        gates.append('GATE {} AND {} {}'.format(N + 4, N + 2, N + 3))
        gates.append('GATE {} AND {} {}'.format(N + 5, N, N + 4))
        gates.append('GATE {} NOT {}'.format(N + 6, N + 3))
        gates.append('GATE {} OR {} {}'.format(N + 7, N + 1, N + 6))
        gates.append('GATE {} AND {} {}'.format(N + 8, k, N + 7))
        gates.append('GATE {} OR {} {}'.format(N + 9, N + 5, N + 8))
        outputs[k] = N + 9
        gates.append('GATE {} AND {} {}'.format(N + 10, k, N + 4))
        gates.append('GATE {} OR {} {}'.format(N + 11, N + 1, N + 10))
        outputs[k + n + 2] = N + 11
        N += 12

    gates.append('GATE {} AND {} {}'.format(N, 0, 3 * n))
    outputs[n] = N
    outputs[n + 1] = N

    for gate in gates:
        print(gate)

    for i in range(len(outputs)):
        print('OUTPUT {} {}'.format(i, outputs[i]))


if __name__ == "__main__":
    n = int(input())
    main(n)
