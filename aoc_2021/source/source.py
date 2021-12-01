from aoc_utils.source import *


def day1():
    print("2021 - Day 1")

    d0=float("inf")
    count_increase = 0
    with open(r"aoc_2021/inputs/day1.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [float(d) for d in lines]
    # part 2: modify lines so that it matches the sum of a sliding window of 3
    lines = sliding_sum(lines)

    for d in lines:
        if d > d0:
            count_increase += 1
        d0 = d
    return count_increase
