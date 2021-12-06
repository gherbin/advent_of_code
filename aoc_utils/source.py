import numpy as np
import copy

import os
import sys
sys.setrecursionlimit(10000)

def sliding_sum(numbers, width=3):
    """
    :param width: width of the sliding window
    :param numbers: list of numbers
    :return: a sliding sum of numbers
    """
    return sliding_sum_rec(numbers, width, [])

def sliding_sum_rec(numbers, width, current):
    if len(numbers) == 0:
        return current
    else:
        current.append(sum_n_first(numbers,width))
        return sliding_sum_rec(numbers[1:], width, current)


def sum_n_first(numbers, n):
    sum = 0
    if n > len(numbers):
        return sum
    for i in range(n):
        sum += numbers[i]
    return sum


def bin2int(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))


def get_boards_from_input(input_str):

    # 2021-day4
    boards = []

    for line in input_str:
        if line[0] == '\n':
            boards += [ [ ] ]
        else:
            line = line.strip('\n').split()
            line = line
            line = [int(i) for i in line]
            boards[-1] += [line]
    boards = np.array(boards)
    return boards

def get_lines_from_input(input_str):
    #2021-day5
    """
    :param input_str:
    :return: sources= list of (x0,y0) of each line; targets = list of (x_end, y_end) of each line
    """
    sources = []
    targets = []
    for line in input_str:
        cs=line.strip().split("->")
        source = cs[0].split(",")
        target = cs[1].split(",")
        sources += [[int(c) for c in source]]
        targets += [[int(c) for c in target]]
    return sources, targets