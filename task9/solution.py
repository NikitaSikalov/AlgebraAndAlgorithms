#! /usr/bin/env python3

if __name__ == '__main__':
    vertecies_number = int(input())
    weights = [0] * vertecies_number

    for i in range(vertecies_number):
        weights[i] = int(input())

    edges_number = int(input())
    edges = []
    for _ in range(edges_number):
        v, u = map(int, (input()).split(" "))
        edges.append((v, u))

    # algorithm vertex cover
    x = [0] * vertecies_number
    y = {edge:0 for edge in edges}

    for edge in edges:
        v, u = edge
        y[edge] = min(weights[v] - x[v], weights[u] - x[u])
        x[v] += y[edge]
        x[u] += y[edge]

    vertex_cover = []
    for v in range(vertecies_number):
        if weights[v] == x[v]:
            vertex_cover.append(v)

    print(' '.join(map(str, vertex_cover)))
