#! /usr/bin/env python3

def get_legendre_table(p: int):
    """Рассчет символов Лежандра в поле Z_p, p - простое"""
    res = [0]
    for n in range(1, p):
        sqrt = None
        for j in range(1, p):
            if (j * j) % p == n:
                # n - квадратичный вычет
                sqrt = j
        res.append(-1 if sqrt is None else 1)
    return res

def main():
    n = int(input())
    p = n - 1
    # предподсчет символов Лежандра
    legendre_table = get_legendre_table(p)

    # матрица Адамара с ведущим столбцов 1
    adamare_matrix_1 = [[1] * n]
    # матрица Адамара с ведущим столбцов 0
    # инвертированная к первой
    adamare_matrix_0 = [[0] * n]

    for row in range(p):
        adamare_matrix_1.append([1])
        adamare_matrix_0.append([0])
        for column in range(p):
            diff = row - column if row >= column else p + (row - column)
            item = -1 if row == column else legendre_table[diff]
            appended_item = item if item == 1 else 0
            adamare_matrix_1[row + 1].append(appended_item)
            adamare_matrix_0[row + 1].append(int(not appended_item))

    # конкатинируем обе матрицы
    res_matrix = adamare_matrix_0 + adamare_matrix_1

    # вывод результата
    res = []
    for row in res_matrix:
        res.append(''.join([str(x) for x in row]))
    res.sort()
    print('\n'.join(res))

if __name__ == "__main__":
    main()
