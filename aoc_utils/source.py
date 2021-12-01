import numpy as np
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