#! /usr/bin/env python3

from typing import List, Union
from random import randint
import sys

# Field base -> Z_p
P = 9973

# https://e-maxx.ru/algo/reverse_element
def reverse_numbers_in_field(p: int = P):
    # -1 -> means non-existing reverse element
    reverse_numbers = [-1] * p
    reverse_numbers[1] = 1
    for i in range(2, p):
        reverse_numbers[i] = p - ((p // i) * reverse_numbers[p % i]) % p
        assert((reverse_numbers[i] * i) % p == 1)
    return reverse_numbers

def add_edge(matrix: List[List[int]], v1: int, v2: int, p: int = P):
    x = randint(1, p - 1)
    matrix[v1][v2] = x

def find_row_number_with_non_zero_element(matrix: List[List[int]], start_row_index: int, column_number: int):
    for i in range(start_row_index, len(matrix)):
        if matrix[i][column_number] != 0:
            return i
    return None

def matrix_to_triangular_view(matrix: List[List[int]], reverse_elements: List[int], p: int = P):
    matrix_size = len(matrix)
    for i in range(1, matrix_size):
        # column that will be 0
        column_number = i - 1
        # find row with non-zero leading element
        row_number = find_row_number_with_non_zero_element(matrix, i - 1, column_number)
        if row_number is None:
            continue
        if row_number != i - 1:
            matrix[i - 1], matrix[row_number] = matrix[row_number], matrix[i - 1]

        row_number = i - 1
        a = matrix[row_number][column_number]
        assert(a > 0 and a < p)

        reverse_a = reverse_elements[a]
        assert(reverse_a > 0 and reverse_a < p)

        for j in range(row_number + 1, matrix_size):
            b = matrix[j][column_number]
            if b == 0:
                continue
            k = (b * reverse_a) % p
            assert((k * a) % p == b)
            new_row_j = []
            for x, y in zip(matrix[row_number], matrix[j]):
                n = (y - k * x) % p
                new_row_j.append(n if n >= 0 else p + n)
            matrix[j] = new_row_j
            assert(matrix[j][column_number] == 0)

    # check matrix with triangular view
    for row in range(matrix_size):
        for column in range(matrix_size):
            if row > column:
                assert(matrix[row][column] == 0)
            else:
                assert(matrix[row][column] >= 0)

def is_singular_matrix(matrix: List[List[int]]):
    for i in range(len(matrix)):
        if matrix[i][i] == 0:
            return True
    return False

def main():
    _ = int(input())
    edges = []
    max_vertex_number = -1
    for line in sys.stdin:
        if line.strip() == '':
            continue
        v1, v2 = [int(x) for x in line.split()]
        max_vertex_number = max(v1, v2, max_vertex_number)
        edges.append((v1, v2))

    number_of_vertexies = max_vertex_number + 1
    iterations_number = 20
    for _ in range(iterations_number):
        matrix = [[0] * number_of_vertexies for _ in range(number_of_vertexies)]
        for v1, v2 in edges:
            add_edge(matrix, v1 ,v2)

        reverse_elements = reverse_numbers_in_field()
        matrix_to_triangular_view(matrix, reverse_elements)
        if not is_singular_matrix(matrix):
            print('yes')
            return

    print('no')

if __name__ == "__main__":
    main()
