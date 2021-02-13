#! /usr/bin/env python3

from typing import List, Tuple, Union
import sys

# Считаем бинарные матрицы
BASE = 2

def _get_upper_boundary_degree_of_two(n: int):
    res = 1
    tmp = n >> 1
    while tmp:
        tmp = tmp >> 1
        res *= 2
    return res * 2 if bool(res ^ n) else res

def _extend_matrix_with_I(matrix: List[List[int]], size: int = None):
    rows_numbers = len(matrix)
    columns_number = len(matrix[0])
    max_dimension = max(rows_numbers, columns_number) if size is None else size
    n = _get_upper_boundary_degree_of_two(max_dimension)
    extended_matrix = []
    for i in range(n):
        arr = matrix[i] + [0] * (n - columns_number) if i < rows_numbers else [ int(j == i) for j in range(n)]
        extended_matrix.append(arr)
    return extended_matrix

def _split_to_blocks(matrix: List[List[int]], split_sizes: Tuple[int, int] = (-1, -1)):
    # split_sizes[0] = column_border, split_sizes[1] = row_border
    A, B = _split_matrix_horizontally(matrix, row_border=split_sizes[1])
    A11, A12 = _split_matrix_vertically(A, column_border=split_sizes[0])
    A21, A22 = _split_matrix_vertically(B, column_border=split_sizes[0])
    return A11, A12, A21, A22


def _build_matrix_from_blocks(A11: List[List[int]], A12: List[List[int]], A21: List[List[int]], A22: List[List[int]]):
    A = _build_matrix_from_vertical_blocks(A11, A12)
    B = _build_matrix_from_vertical_blocks(A21, A22)
    return _build_matrix_from_horizontal_blocks(A, B)

def _split_matrix_horizontally(matrix: List[List[int]], row_border: int = -1):
    A, B = [], []
    rows_number = len(matrix)
    if row_border == -1:
        row_border = rows_number // 2
    for i, row in enumerate(matrix):
        if i < row_border:
            A.append(row.copy())
        else:
            B.append(row.copy())
    return (A, B)

def _split_matrix_vertically(matrix: List[List[int]], column_border: int = -1):
    A, B = [], []
    columns_number = len(matrix[0])
    if column_border == -1:
        column_border = columns_number // 2
    transposed_matrix = transpose_matrix(matrix)
    A, B = _split_matrix_horizontally(transposed_matrix, column_border)
    A, B = transpose_matrix(A), transpose_matrix(B)
    return A, B

def _build_matrix_from_horizontal_blocks(A: List[List[int]], B: List[List[int]]):
    res = []
    for row in A:
        res.append(row.copy())
    for row in B:
        res.append(row.copy())
    return res

def _build_matrix_from_vertical_blocks(A: List[List[int]], B: List[List[int]]):

    return [row1 + row2 for row1, row2 in zip(A, B)]

def get_reverse_number(val: int):
    # обратный элемент в поле Z_BASE, где BASE - простое число
    # по МТФ
    return (val ** (BASE - 1)) % BASE

def matrix_sum(*matrixes):
    """Сумма матриц"""
    res = []
    for rows in zip(*matrixes):
        res.append([sum(x) % BASE for x in zip(*rows)])
    return res

def matrix_diff(matrix1: List[List[int]], matrix2: List[List[int]]):
    """Разность двух матриц"""
    res = []
    for row1, row2 in zip(matrix1, matrix2):
        res.append([(x - y) % BASE for x, y in zip(row1, row2)])
    return res

def _formatted_matrix_product(matrix1: List[List[int]], matrix2: List[List[int]]):
    if (len(matrix1) == 1 and len(matrix2) == 1):
        n1 = matrix1[0][0]
        n2 = matrix2[0][0]
        return [[(n1 * n2) % BASE]]

    A11, A12, A21, A22 = _split_to_blocks(matrix1)
    B11, B12, B21, B22 = _split_to_blocks(matrix2)
    P1 = _formatted_matrix_product(matrix_sum(A11, A22), matrix_sum(B11, B22))
    P2 = _formatted_matrix_product(matrix_sum(A21, A22), B11)
    P3 = _formatted_matrix_product(A11, matrix_diff(B12, B22))
    P4 = _formatted_matrix_product(A22, matrix_diff(B21, B11))
    P5 = _formatted_matrix_product(matrix_sum(A11, A12), B22)
    P6 = _formatted_matrix_product(matrix_diff(A21, A11), matrix_sum(B11, B12))
    P7 = _formatted_matrix_product(matrix_diff(A12, A22), matrix_sum(B21, B22))

    C11 = matrix_diff(matrix_sum(P1, P4, P7), P5)
    C12 = matrix_sum(P3, P5)
    C21 = matrix_sum(P2, P4)
    C22 = matrix_diff(matrix_sum(P1, P3, P6), P2)
    return _build_matrix_from_blocks(C11, C12, C21, C22)


def _cut_matrix_with_size(matrix: List[List[int]], rows_numbers: int, columns_number: int):
    res = []
    for index, row in enumerate(matrix):
        if index == rows_numbers:
            break
        res.append(row[:columns_number])
    return res

def _matrix_product_of_two(matrix1: List[List[int]], matrix2: List[List[int]]):
    rows_number1 = len(matrix1)
    columns_number1 = len(matrix1[0])
    rows_number2 = len(matrix2)
    columns_number2 = len(matrix2[0])

    extended_size_matrix = max(rows_number1, rows_number2, columns_number1, columns_number2)

    A = _extend_matrix_with_I(matrix1, extended_size_matrix)
    B = _extend_matrix_with_I(matrix2, extended_size_matrix)
    product_matrix = _formatted_matrix_product(A, B)
    return _cut_matrix_with_size(product_matrix, rows_number1, columns_number2)

def matrix_product(*matrixes) -> List[List[int]]:
    """"Произведение матриц"""
    res = None
    for matrix in matrixes:
        res = matrix if res is None else _matrix_product_of_two(res, matrix)
    return res

def matrix_dot_number(matrix: List[List[int]], val: int):
    res = []
    rows_number = len(matrix)
    for row in range(rows_number):
        res.append([x * val for x in matrix[row]])
    return res

def is_up_triangle_matrix(matrix: List[List[int]]):
    """Проверяет, что матрица верхнетреугольная"""
    rows_number = len(matrix)
    columns_number = len(matrix[0])
    for row in range(rows_number):
        for column in range(columns_number):
            if row > column and matrix[row][column] != 0:
                return False
    return True

def is_down_triangle_matrix(matrix: List[List[int]]):
    """Проверяет, что матрица нижнетреугольная"""
    transposed_matrix = transpose_matrix(matrix)
    return is_up_triangle_matrix(transposed_matrix)

def transpose_matrix(matrix: List[List[int]]):
    """"Транспонирование матрицы"""
    res = []
    rows_number = len(matrix)
    if rows_number == 0:
        return res
    columns_number = len(matrix[0])
    for column in range(columns_number):
        tmp = [matrix[i][column] for i in range(rows_number)]
        res.append(tmp)
    return res

def reverse_square_triangle_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """Обращение верхне/нижне треугольной матрицы"""
    if len(matrix) == 1:
        val = matrix[0][0]
        return [[get_reverse_number(val)]]

    is_down_triangled = is_down_triangle_matrix(matrix)
    matrix_to_revert = matrix if is_down_triangled else transpose_matrix(matrix)

    initial_rows_number = len(matrix_to_revert)
    initial_columns_number = len(matrix_to_revert[0])
    matrix_to_revert = _extend_matrix_with_I(matrix_to_revert)

    B, O, C, D = _split_to_blocks(matrix_to_revert)
    B_reversed = reverse_square_triangle_matrix(B)
    D_reversed = reverse_square_triangle_matrix(D)
    A11, A12, A21, A22 = B_reversed, O, matrix_product(matrix_dot_number(D_reversed, -1), C, B_reversed), D_reversed

    reversed_matrix = _cut_matrix_with_size(_build_matrix_from_blocks(A11, A12, A21, A22), initial_rows_number, initial_columns_number)
    if is_down_triangled:
        return reversed_matrix
    return transpose_matrix(reversed_matrix)

def eye_matrix(size: int) -> List[List[int]]:
    """Создает единичную матрицу"""
    matrix = []
    for i in range(size):
        matrix.append([int(j == i) for j in range(size)])
    return matrix

def zero_matrix(rows_number: int, columns_number: Union[int, None] = None) -> List[List[int]]:
    """Создает нулевую квадратную матрицу из нулей"""
    columns_number = columns_number if columns_number is not None else rows_number
    return [[0] * columns_number for _ in range(rows_number)]

def permutation_matrix(permutation: List[int]):
    """Создает матрицу перестановки"""
    size = len(permutation)
    matrix = [[0] * size for _ in range(size)]
    for i, val in enumerate(permutation):
        matrix[val][i] = 1
    return matrix

def inverse_permutation(permutation: List[int]):
    """Инвертирует матрицу перестановку"""
    n = len(permutation)
    inversed_permutation = [-1] * n
    for idx, val in enumerate(permutation):
        inversed_permutation[val] = idx

    return inversed_permutation

def matrix_product_with_permutation(matrix: List[List[int]], permutation: List[int]):
    """Умножает матрицу, на матрицу перестановки"""
    rows_number = len(matrix)
    columns_number = len(matrix[0])
    res = [[-1] * columns_number for _ in range(rows_number)]
    for row in range(rows_number):
        for column, val in enumerate(permutation):
            res[row][column] = matrix[row][val]

    return res

def permutations_product(permutation1: List[int], permutation2: List[int]):
    """Произведение матриц перестановки"""
    assert(len(permutation1) == len(permutation2))
    n = len(permutation1)
    res = [-1] * n
    for i in range(n):
        res[i] = permutation1[permutation2[i]]
    return res

def extend_permutation_matrix_with_I(permutation: List[int], extended_size: int):
    """Приписывает блок единичной матрицы в верхнем левом углу матрицы перестановки"""
    return list(range(extended_size)) + [val + extended_size for val in permutation]

def _formatted_lup_decomposion(matrix: List[List[int]]) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    rows_number = len(matrix)
    columns_number = len(matrix[0])

    if rows_number == 1:
        L = [[1]]
        not_zero_column = 0
        for i, val in enumerate(matrix[0]):
            if val != 0:
                not_zero_column = i
                break
        permutation = list(range(columns_number))
        permutation[0] = not_zero_column
        permutation[not_zero_column] = 0

        U = matrix_product_with_permutation(matrix, permutation)
        return (L, U, permutation)

    A = matrix
    m = len(A)

    B, C = _split_matrix_horizontally(A)

    L1, U1, P1 = _formatted_lup_decomposion(B)

    D = matrix_product_with_permutation(C, inverse_permutation(P1))
    E, _ = _split_matrix_vertically(U1, m // 2)
    F, _ = _split_matrix_vertically(D, m // 2)
    G = matrix_diff(D, matrix_product(F, reverse_square_triangle_matrix(E), U1))
    _, G_stroke = _split_matrix_vertically(G, m // 2)

    L2, U2, P2 = _formatted_lup_decomposion(G_stroke)

    P3 = extend_permutation_matrix_with_I(P2, m // 2)
    H = matrix_product_with_permutation(U1, inverse_permutation(P3))
    L = _build_matrix_from_blocks(L1, zero_matrix(m // 2), matrix_product(F, reverse_square_triangle_matrix(E)), L2)
    U = _build_matrix_from_horizontal_blocks(H, _build_matrix_from_vertical_blocks(zero_matrix(m // 2), U2))
    P = permutations_product(P3, P1)

    return (L, U, P)

def lup_decomposion(matrix: List[List[int]]):
    """НВП разложение произвольной квадратной матрицы"""
    rows_number = len(matrix)
    columns_number = len(matrix[0])
    A = _extend_matrix_with_I(matrix)
    L, U, permutation = _formatted_lup_decomposion(A)
    P = permutation_matrix(permutation)
    if len(A) != rows_number:
        L, _, _, _ = _split_to_blocks(L, split_sizes=(columns_number, rows_number))
        U, _, _, _ = _split_to_blocks(U, split_sizes=(columns_number, rows_number))
        P, _, _, _ = _split_to_blocks(P, split_sizes=(columns_number, rows_number))
    return L, U, P

def printMatrix(matrix):
    for row in matrix:
        print(' '.join([str(x) for x in row]))

def main():
    matrix = []
    for line in sys.stdin:
        if line.strip() == '':
            continue
        matrix.append([int(x) for x in line.split()])
    L, U, P = lup_decomposion(matrix)
    printMatrix(L)
    printMatrix(U)
    printMatrix(P)


if __name__ == "__main__":
    main()
