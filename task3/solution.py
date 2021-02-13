#!/usr/bin/env python3

from typing import Dict, List, Tuple, Union
from itertools import combinations
from enum import Enum
from math import factorial

# base functions ------------------------------------------------------->
def get_hash(x) -> str:
    """Хеширует значение x, где x - массив или список (hash не зависит от порядка элементов в списке)"""
    if isinstance(x, int) or isinstance(x, str):
        return str(x)
    return '|'.join([str(y) for y in sorted(x)])

def convert_value_to_bool(x: str) -> List[bool]:
    """Конвертирует строку из 0 и 1 в список из True и False"""
    assert(isinstance(x, str))

    return [bool(int(y)) for y in x]

def convert_value_to_str(x: Union[List[bool], Tuple[bool]]) -> str:
    """Конвертирует список из True и False в строку из 0 и 1"""
    assert(len(x) > 0)
    assert(isinstance(x[0], bool))

    return ''.join([str(int(y)) for y in x])

def values_and(x1: str, x2: str) -> str:
    """Побитовое <И> 2-ух строк"""
    assert(len(x1) == len(x2))

    values1 = convert_value_to_bool(x1)
    values2 = convert_value_to_bool(x2)
    return convert_value_to_str([x and y for x, y in zip(values1, values2)])

def values_or(str_values: Union[List[str], Tuple[str]]) -> str:
    """Побитовое <ИЛИ> 2-ух строк"""
    assert(len(str_values) > 0)

    value = convert_value_to_bool(str_values[0])
    base_len = len(value)
    for str_value in str_values[1:]:
        assert(len(str_value) == base_len)

        bool_value = convert_value_to_bool(str_value)
        next_value = [x or y for x, y in zip(value, bool_value)]
        value = next_value
    return convert_value_to_str(value)

def get_variables_sets(n: int) -> List[List[bool]]:
    """Все возможные значения из 0 и 1 для n переменных"""
    variables_combintaions = []
    indexies = range(n)
    for true_number in range(n + 1):
        for true_positions in combinations(indexies, true_number):
            values = [False] * n
            for i in true_positions:
                values[i] = True
            variables_combintaions.append(values)
    return variables_combintaions

def get_number_sets(n: int, variables_count: int):
    """Генерирует все возможные наборы чисел от 0 до (n - 1) длины variables_count"""
    return list(combinations(range(n), variables_count))


def test_base_functions():
    # get_hash
    assert(get_hash(1) == '1')
    assert(get_hash('1') == '1')
    assert(get_hash([20, 5, 10, 1]) == '1|5|10|20')

    # convert_value_to_bool
    assert(convert_value_to_bool('1101') == [True, True, False, True])
    assert(convert_value_to_bool('0') == [False])
    assert(convert_value_to_bool('1') == [True])

    # convert_value_to_str
    assert(convert_value_to_str([True, True, False, True]) == '1101')
    assert(convert_value_to_str([True]) == '1')
    assert(convert_value_to_str([False]) == '0')

    # values_and
    assert(values_and('11001', '01100') == '01000')
    assert(values_and('1', '0') == '0')
    assert(values_and('1', '1') == '1')

    # values_or
    assert(values_or(['1001', '0101']) == '1101')
    assert(values_or(['0000', '0101', '1111']) == '1111')
    assert(values_or(['1', '0']) == '1')
    assert(values_or(['0', '0', '0']) == '0')
    assert(values_or(['0000']) == '0000')

    # get_variables_sets
    variable_sets = get_variables_sets(4)
    assert(len(variable_sets) == 2 ** 4)
    assert(len(set([convert_value_to_str(x) for x in variable_sets])) == 2 ** 4)

    # get_number_sets
    number_sets = get_number_sets(8, 3)
    assert(len(number_sets) == factorial(8) / (factorial(3) * factorial(5)))

    print('Tests passed!')


# test_base_functions()

# ----------------------------------------------------------------------------->

class GateType(Enum):
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'
    FIXTURE = 'FIXTURE'

class Gate:
    def __init__(self, gate_type: GateType, out: int, in1: int, in2: Union[int, None] = None) -> None:
        self.type = gate_type
        self.out = out
        self.in1 = in1
        self.in2 = in2

    def print_gate(self):
        if self.type == GateType.NOT or self.in2 is None:
            return 'GATE {out} {type} {in1}'.format(out=self.out, type=self.type.value, in1=self.in1)
        return 'GATE {out} {type} {in1} {in2}'.format(
            out=self.out,
            type=self.type.value,
            in1=min(self.in1, self.in2),
            in2=max(self.in1, self.in2))

class GateFactory:
    def __init__(self, N: int) -> None:
        self.gates: List[Gate] = []
        self._start: int = N

    def create_new_gate(self, gate_type: GateType, in1: int, in2: Union[int, None] = None) -> Gate:
        out = self._start + len(self.gates)
        gate = Gate(gate_type=gate_type, out=out, in1=in1, in2=in2)
        self.gates.append(gate)
        return gate

    def print_gates(self):
        return '\n'.join([gate.print_gate() for gate in self.gates])


class UniversalMultipole:
    """Класс реализации универсального многополюсника n-го порядка"""
    def __init__(self, n: int) -> None:
        # количество входных переменных
        self.number_of_variables: int = n
        # набор из 0 и 1 в таблице истинности
        self.variables_table: List[List[bool]] = get_variables_sets(n)
        # фабрика функциональных элементов
        self.gate_factory = GateFactory(n)
        # hash таблица конъюнкий, значения которой это (gate, value)
        # одиночные переменные также считаем конъюнкциями для упрощения работы алгоритма
        self.conjuctions_table: Dict[str, Tuple[Gate, str]] = dict()
        # значения на таблице истинности
        self.values_table: Dict[str, Gate] = dict()
        # конкатинации длины n, из которых мы будем составлять итоговую ДНФ
        self.final_conjuctions: List[str] = []

    def create_schema(self):
        self._init_conjuctions_table()
        self._init_negate_gates()
        self._init_conjuctions()
        self._init_disjuctions()
        self._add_zero_gate()

        assert(len(self.values_table) == 2 ** (2 ** self.number_of_variables))
        assert(len(self.gate_factory.gates) == 2 ** (2 ** self.number_of_variables) - self.number_of_variables)

    def print_schema(self):
        """Выводит функциональную схему"""
        n = self.number_of_variables
        gates = self.gate_factory.print_gates()
        outputs = '\n'.join(['OUTPUT {i} {i}'.format(i=i) for i in range(2 ** (2 ** n))])
        print('\n'.join([gates, outputs]))

    def _init_conjuctions_table(self):
        """Инициализация таблицы конъюнкций с учетом входных значений"""
        for i in range(self.number_of_variables):
            fixture_gate = Gate(gate_type=GateType.FIXTURE, out=i, in1=-1)
            value = self._get_conjuction_value(i)
            conjuction_hash = get_hash(i)
            self.conjuctions_table[conjuction_hash] = (fixture_gate, value)
            self.values_table[value] = fixture_gate

        assert(len(self.conjuctions_table) == self.number_of_variables)
        assert(len(set([val for _, val in self.conjuctions_table.values()])) == self.number_of_variables)
        assert(len(self.values_table) == self.number_of_variables)

    def _init_negate_gates(self):
        """Формирует выходы с отрицаниями входных переменных"""
        for i in range(self.number_of_variables):
            gate = self.gate_factory.create_new_gate(GateType.NOT, i)
            k = i + self.number_of_variables
            value = self._get_conjuction_value(k)
            conjuction_hash = get_hash(k)
            self.conjuctions_table[conjuction_hash] = gate, value
            self.values_table[value] = gate

        assert(len(self.conjuctions_table) == 2 * self.number_of_variables)
        assert(len(set([val for _, val in self.conjuctions_table.values()])) == 2 * self.number_of_variables)
        assert(len(self.values_table) == 2 * self.number_of_variables)

    def _init_conjuctions(self):
        """Генерирует все возможные конъюнкции длины от 2 до n"""
        for variables_count in range(2, self.number_of_variables + 1):
            for conjuction in self._get_conjuctions(variables_count):
                gate = self._add_gate_for_conjuction(conjuction)
                value = self._get_conjuction_value(conjuction)
                conjuction_hash = get_hash(conjuction)
                self.conjuctions_table[conjuction_hash] = (gate, value)
                self.values_table[value] = gate
                if variables_count == self.number_of_variables:
                    self.final_conjuctions.append(conjuction_hash)
        if self.number_of_variables == 1:
            self.final_conjuctions = [get_hash(0), get_hash(1)]

        assert(len(self.final_conjuctions) == 2 ** self.number_of_variables)
        assert(len(self.conjuctions_table) == len(self.values_table))

    def _init_disjuctions(self):
        """Генерирует функциональные элементы дизъюнкций"""
        disjuctions_count = len(self.final_conjuctions)
        for terms_number in range(2, len(self.final_conjuctions) + 1):
            disjuctions = list(combinations(self.final_conjuctions, terms_number))
            for disjuction in disjuctions:
                disjuctions_count += 1
                value = self._get_disjunction_value(disjuction)
                if value not in self.values_table:
                    gate = self._add_gate_for_disjuction(disjuction)
                    self.values_table[value] = gate

        # количество обработанных дизъюнкций == количеству функций - 1 (за исключеним 0)
        assert(disjuctions_count == 2 ** (2 ** self.number_of_variables) - 1)

    def _get_conjuctions(self, variables_count: int):
        """Получение всевозможных ненулевых конъюнкций длины n"""
        return self._filter_conjuctions(get_number_sets(2 * self.number_of_variables, variables_count))

    def _filter_conjuctions(self, conjuctions):
        """Фильтрует конъюнкции от тожденственных 0"""
        def filter_predicate(conjuction):
            value = self._get_conjuction_value(conjuction)
            return '1' in value
        return filter(filter_predicate, conjuctions)

    def _get_conjuction_value(self, conjuction: Union[int, List[int], Tuple]) -> str:
        """Получает значения конъюнкции на таблице значений"""
        if isinstance(conjuction, int):
            x = conjuction
            n = self.number_of_variables
            if conjuction < n:
                return convert_value_to_str([arr[x] for arr in self.variables_table])
            else:
                return convert_value_to_str([not arr[x - n] for arr in self.variables_table])

        assert(len(conjuction) >= 2)

        _, value1 = self.conjuctions_table[get_hash(conjuction[0])]
        if get_hash(conjuction[1:]) not in self.conjuctions_table:
            return '0' * (2 ** self.number_of_variables)
        _, value2 = self.conjuctions_table[get_hash(conjuction[1:])]
        return values_and(value1, value2)

    def _get_disjunction_value(self, disjuction: Union[List[str], Tuple]) -> str:
        """Расчитывает столбец значений для дизъюнкции"""
        conjuctions_values = [self.conjuctions_table[conjuction_hash][1] for conjuction_hash in disjuction]
        return values_or(conjuctions_values)

    def _add_gate_for_conjuction(self, conjuction: Union[List[int], Tuple]) -> Gate:
        """Добавляет функциональный элемент схемы для конъюнкции"""
        hash1 = get_hash(conjuction[0])
        hash2 = get_hash(conjuction[1:])
        gate1, _ = self.conjuctions_table[hash1]
        gate2, _ = self.conjuctions_table[hash2]
        return self.gate_factory.create_new_gate(GateType.AND, gate1.out, gate2.out)

    def _add_gate_for_disjuction(self, disjuction: Union[List[str], Tuple]) -> Gate:
        """Добавляет функциональный элемент схемы для дизъюнкции"""
        conjuction_hash = disjuction[0]
        rest_disjuction_value = self._get_disjunction_value(disjuction[1:])
        gate1, _ = self.conjuctions_table[conjuction_hash]
        gate2 = self.values_table[rest_disjuction_value]
        return self.gate_factory.create_new_gate(GateType.OR, gate1.out, gate2.out)

    def _add_zero_gate(self):
        """Просто добавляет к функциональной схеме 0 элемент"""
        gate = self.gate_factory.create_new_gate(GateType.AND, 0, self.number_of_variables)
        zero_value = '0' * (2 ** self.number_of_variables)
        self.values_table[zero_value] = gate



def main():
    n = int(input())
    universal_multipole = UniversalMultipole(n)
    universal_multipole.create_schema()
    universal_multipole.print_schema()

if __name__ == "__main__":
    main()
