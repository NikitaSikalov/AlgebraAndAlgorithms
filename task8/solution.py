#! /usr/bin/env python3

import numpy as np
import sys
from typing import List, Tuple

def cross_density(set1: List[int], set2: List[int], laplasian) -> Tuple[float, Tuple[int, int]]:
    assert(len(set1) <= len(set2))

    edges_number = 0
    for v1 in set1:
        for v2 in set2:
            edges_number += int(laplasian[v1, v2] == -1)
    density = edges_number / (len(set1) * len(set2))
    density_hash = (edges_number, len(set1) * len(set2))
    return density, density_hash


def main():
    _ = int(input())
    edges = set()
    index = 0
    vertecies_dict = dict()
    for line in sys.stdin:
        args = list(map(int, line.strip().split()))
        assert(len(args) == 2)
        v1, v2 = args[0], args[1]
        if v1 not in vertecies_dict:
            vertecies_dict[v1] = index
            index += 1
        if v2 not in vertecies_dict:
            vertecies_dict[v2] = index
            index += 1
        edges.add((v1, v2))

    dim = len(vertecies_dict)
    laplacian = np.zeros(shape=(dim, dim), dtype=int)

    vertecies = [0] * dim
    for key, value in vertecies_dict.items():
        vertecies[value] = key

    for v1, v2 in edges:
        v1_id = vertecies_dict[v1]
        v2_id = vertecies_dict[v2]
        if laplacian[v1_id, v2_id] == -1:
            continue
        laplacian[v1_id, v2_id] = -1
        laplacian[v2_id, v1_id] = -1
        laplacian[v1_id, v1_id] += 1
        laplacian[v2_id, v2_id] += 1

    assert(np.array_equal(np.zeros(dim), np.sum(laplacian, axis=0)))
    assert(np.array_equal(np.zeros(dim), np.sum(laplacian, axis=1)))

    _, eigenvectors = np.linalg.eigh(laplacian)
    v2 = eigenvectors[:, 1]
    v2 = [(i, value) for i, value in enumerate(v2)]
    v2.sort(key=lambda x: x[1], reverse=True)
    vertecies_order = [x[0] for x in v2]
    sets_vertecies = [(vertecies_order[:i], vertecies_order[i:]) for i in range(1, dim)]

    sets_dencities = []
    for vertecies_less_set, vertecies_bigger_set in sets_vertecies:
        if len(vertecies_bigger_set) < len(vertecies_less_set):
            vertecies_less_set, vertecies_bigger_set = vertecies_bigger_set, vertecies_less_set
        density, density_hash = cross_density(vertecies_less_set, vertecies_bigger_set, laplacian)
        sets_dencities.append((vertecies_less_set, density, density_hash))
        if len(vertecies_less_set) == len(vertecies_bigger_set):
            sets_dencities.append(((vertecies_bigger_set, density, density_hash)))

    _, _, density_hash = min(sets_dencities, key=lambda x: x[1])
    filtered_sets = filter(lambda x: x[2] == density_hash, sets_dencities)
    sets_with_min_slice = [sorted([vertecies[y] for y in x[0]]) for x in filtered_sets]

    sets_with_min_slice.sort()
    print(" ".join(map(str, sets_with_min_slice[0])))


if __name__ == "__main__":
    main()
