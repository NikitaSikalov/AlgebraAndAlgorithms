#! /usr/bin/env python3

def compute_schema(number_of_inputs: int):
    step = 1
    numbers = list(range(number_of_inputs))
    ans = list()
    least_unused_number = number_of_inputs
    while step < number_of_inputs:
        pointer = 0
        next_numbers = numbers.copy()
        while pointer + step < number_of_inputs:
            ans.append('GATE {} OR {} {}'.format(least_unused_number, numbers[pointer], numbers[pointer + step]))
            next_numbers[pointer + step] = least_unused_number
            least_unused_number += 1
            pointer += 1
        step *= 2
        numbers = next_numbers

    for i in range(number_of_inputs):
        ans.append('OUTPUT {} {}'.format(i, numbers[i]))
    print('\n'.join(ans))


if __name__ == "__main__":
    number_of_inputs = int(input())
    compute_schema(number_of_inputs)
