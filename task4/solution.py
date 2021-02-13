#! /usr/bin/env python3

from typing import List
import sys

BASE = 9

def get_upper_boundary_degree_of_two(n: int):
    res = 1
    tmp = n >> 1
    while tmp:
        tmp = tmp >> 1
        res *= 2
    return res * 2 if bool(res ^ n) else res

def log_n(n: int):
    res = 0
    tmp = n >> 1
    while tmp:
        tmp = tmp >> 1
        res += 1
    return res

def printMatrix(matrix):
    for row in matrix:
        print(' '.join([str(x) for x in row]))

# ---------------------------------------------------
def tests():
    # get_upper_boundary_degree_of_two
    assert(get_upper_boundary_degree_of_two(1) == 1)
    assert(get_upper_boundary_degree_of_two(2) == 2)
    assert(get_upper_boundary_degree_of_two(3) == 4)
    assert(get_upper_boundary_degree_of_two(63) == 64)

    assert(log_n(1) == 0)
    assert(log_n(2) == 1)
    assert(log_n(3) == 1)
    assert(log_n(4) == 2)
    assert(log_n(9) == 3)

# tests()
# ----------------------------------------------------

def extend_matrix_with_I(matrix: List[List[int]]):
    current_size = len(matrix)
    n = get_upper_boundary_degree_of_two(current_size)
    extended_matrix = []
    for i in range(n):
        arr = matrix[i].copy() + [0] * (n - current_size) if i < current_size else [ int(j == i) for j in range(n)]
        extended_matrix.append(arr)
    return extended_matrix

def split_to_blocks(matrix: List[List[int]]):
    A11 = []
    A12 = []
    A21 = []
    A22 = []
    n = len(matrix)
    for row_number in range(n):
        if row_number < n / 2:
            A11.append(matrix[row_number][:(n >> 1)])
            A12.append(matrix[row_number][(n >> 1):])
        else:
            A21.append(matrix[row_number][:(n >> 1)])
            A22.append(matrix[row_number][(n >> 1):])
    return A11, A12, A21, A22


def build_matrix_from_blocks(A11: List[List[int]], A12: List[List[int]], A21: List[List[int]], A22: List[List[int]]):
    matrix = []
    for row1, row2 in zip(A11, A12):
        matrix.append(row1 + row2)
    for row1, row2 in zip(A21, A22):
        matrix.append(row1 + row2)
    return matrix

def matrix_sum(*matrixes):
    res = []

    for rows in zip(*matrixes):
        res.append([sum(x) % BASE for x in zip(*rows)])
    return res

def matrix_diff(matrix1: List[List[int]], matrix2: List[List[int]]):
    assert(len(matrix1) == len(matrix2))
    res = []

    for row1, row2 in zip(matrix1, matrix2):
        res.append([(x - y) % BASE for x, y in zip(row1, row2)])
    return res

def matrix_product(matrix1: List[List[int]], matrix2: List[List[int]]):
    assert(len(matrix1) > 0)
    assert(len(matrix2) > 0)

    assert(len(matrix1) == len(matrix2[0]))
    assert(len(matrix1[0]) == len(matrix2))

    if (len(matrix1) == 1 and len(matrix2) == 1):
        n1 = matrix1[0][0]
        n2 = matrix2[0][0]
        return [[(n1 * n2) % BASE]]

    A11, A12, A21, A22 = split_to_blocks(matrix1)
    B11, B12, B21, B22 = split_to_blocks(matrix2)
    P1 = matrix_product(matrix_sum(A11, A22), matrix_sum(B11, B22))
    P2 = matrix_product(matrix_sum(A21, A22), B11)
    P3 = matrix_product(A11, matrix_diff(B12, B22))
    P4 = matrix_product(A22, matrix_diff(B21, B11))
    P5 = matrix_product(matrix_sum(A11, A12), B22)
    P6 = matrix_product(matrix_diff(A21, A11), matrix_sum(B11, B12))
    P7 = matrix_product(matrix_diff(A12, A22), matrix_sum(B21, B22))

    C11 = matrix_diff(matrix_sum(P1, P4, P7), P5)
    C12 = matrix_sum(P3, P5)
    C21 = matrix_sum(P2, P4)
    C22 = matrix_diff(matrix_sum(P1, P3, P6), P2)
    return build_matrix_from_blocks(C11, C12, C21, C22)

def formatted_matrix_pow(matrix: List[List[int]], degree: int):
    log_degree = log_n(degree)
    A = matrix
    for _ in range(log_degree):
        A = matrix_product(A, A)
    if (1 << log_degree) == degree:
        return A

    rest_degree = degree - (1 << log_degree)
    B = formatted_matrix_pow(matrix, rest_degree)
    return matrix_product(A, B)


def cut_matrix_with_size(matrix: List[List[int]], n: int):
    res = []
    for index, row in enumerate(matrix):
        if index == n:
            break
        res.append(row[:n])
    return res

def matrix_pow(matrix: List[List[int]]):
    initial_size = len(matrix)
    A = extend_matrix_with_I(matrix)
    A_n = formatted_matrix_pow(A, initial_size)
    return cut_matrix_with_size(A_n, initial_size)


if __name__ == "__main__":
    matrix = []
    for line in sys.stdin:
        matrix.append([int(x) for x in line.split()])

    A = matrix_pow(matrix)
    printMatrix(A)
