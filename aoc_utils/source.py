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


def is_low_point(array, ind_x, ind_y):
    """
    :param array:
    :param ind_x:
    :param ind_y:
    :return: True if array[ind_x, ind_y] is a low point
    """
    val = array[ind_x, ind_y]

    if (ind_x > 0) and (array[ind_x - 1, ind_y] <= val):
        return False
    if (ind_x < array.shape[0]-1) and (array[ind_x + 1, ind_y] <= val):
        return False
    if (ind_y > 0) and (array[ind_x, ind_y - 1] <= val):
        return False
    if (ind_y < array.shape[1]-1) and (array[ind_x , ind_y + 1] <= val):
        return False

    return True


def count_around(coords, areas):
    """
    :param coords: center of a basin (=low point)
    :param areas:
    :return: the size of the basin centered in coords in areas. It's implemented using a recursive function
    """
    return count_around_rec([coords], areas, 0, set([]))


def count_around_rec(list_of_coords, areas, cur_counter, visited_coords):
    """
    :param list_of_coords: points still to visit
    :param areas: map to visit
    :param cur_counter: current accumulator of basin size
    :param visited_coords: already visited node, to ensure not to count twice
    :return: the size of a basin
    """
    max_x = areas.shape[0]-1
    max_y = areas.shape[1]-1
    min_x = 0
    min_y = 0

    if len(list_of_coords) == 0:
        return cur_counter
    else:
        coords = list_of_coords[0]
        new_visited_coords = copy.deepcopy(visited_coords)
        new_visited_coords.add(coords)
        x, y = coords
        if areas[x, y] == 9:
            return count_around_rec(list_of_coords[1:], areas, cur_counter, new_visited_coords)
        else:
            cur_counter += 1
            to_add_coords = list({(min(x + 1, max_x), y),
                             (max(x - 1, min_x), y),
                             (x, min(y + 1, max_y)),
                             (x, max(y - 1, min_y))}.difference(new_visited_coords))
            new_coords = list(set(list_of_coords[1:] + to_add_coords))

            return count_around_rec(new_coords, areas, cur_counter, new_visited_coords)
