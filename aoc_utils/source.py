import numpy as np
import copy

import os
import sys
from collections import deque, Counter
import cProfile
import pstats

profile = cProfile.Profile()

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
        current.append(sum_n_first(numbers, width))
        return sliding_sum_rec(numbers[1:], width, current)


def sum_n_first(numbers, n):
    sum = 0
    if n > len(numbers):
        return sum
    for i in range(n):
        sum += numbers[i]
    return sum


def bin2int(binary):
    if binary.__class__.__name__ not in ["list", "ndarray"]:
        binary = [int(i) for i in binary]

    return sum(val * (2 ** idx) for idx, val in enumerate(reversed(binary)))


def get_boards_from_input(input_str):
    # 2021-day4
    boards = []

    for line in input_str:
        if line[0] == '\n':
            boards += [[]]
        else:
            line = line.strip('\n').split()
            line = line
            line = [int(i) for i in line]
            boards[-1] += [line]
    boards = np.array(boards)
    return boards


def get_lines_from_input(input_str):
    # 2021-day5
    """
    :param input_str:
    :return: sources= list of (x0,y0) of each line; targets = list of (x_end, y_end) of each line
    """
    sources = []
    targets = []
    for line in input_str:
        cs = line.strip().split("->")
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
    if (ind_x < array.shape[0] - 1) and (array[ind_x + 1, ind_y] <= val):
        return False
    if (ind_y > 0) and (array[ind_x, ind_y - 1] <= val):
        return False
    if (ind_y < array.shape[1] - 1) and (array[ind_x, ind_y + 1] <= val):
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
    max_x = areas.shape[0] - 1
    max_y = areas.shape[1] - 1
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


def find_first_illegal_char(chars):
    open_chars = ("<", "{", "(", "[")
    closing_chars = (">", "}", ")", "]")
    all_ok_chars = open_chars + closing_chars
    q = deque()
    for c in chars:
        if c not in all_ok_chars:
            return c
        elif c in open_chars:
            q.append(c)
        elif c in closing_chars:
            el = q.pop()
            index = open_chars.index(el)
            if c == closing_chars[index]:
                continue
            else:
                return c


def compute_score_illegal_chars(lines):
    closing_chars = (">", "}", ")", "]")
    penalty = (25137, 1197, 3, 57)

    errors = [find_first_illegal_char(line) for line in lines]
    occurrences = Counter(errors)
    scores = 0
    for cc in closing_chars:
        scores += occurrences[cc] * penalty[closing_chars.index(cc)]

    lines_cleaned = [line for line, error in zip(lines, errors) if error is None]
    return scores, lines_cleaned


def autocomplete(chars):
    # todo : pity, lots of stuff common with find_first_illegal_char. To be changed (but 2 stars already ;-) )
    open_chars = ("<", "{", "(", "[")
    closing_chars = (">", "}", ")", "]")
    q = deque()
    for c in chars:
        if c in open_chars:
            q.append(c)
        elif c in closing_chars:
            q.pop()
            # index_ = open_chars.index(el)
    missing_chunck = [closing_chars[open_chars.index(e)] for e in q]
    missing_chunck.reverse()

    penalty = (4, 3, 1, 2)
    score = 0
    for char in missing_chunck:
        score = 5 * score + penalty[closing_chars.index(char)]
    return score


def broadcast_indices(x, y, max_x, max_y, min_x=0, min_y=0):
    indices_around = np.array([[1, 0], [1, 1], [0, 1], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]).T
    xs = indices_around[0] + x
    ys = indices_around[1] + y
    xs = np.clip(xs, min_x, max_x)
    ys = np.clip(ys, min_y, max_y)

    indices = np.array([xs, ys])
    indices = np.unique(indices, axis=1)
    return indices


def octopus_flashes(energies):
    max_x = energies.shape[0] - 1
    max_y = energies.shape[1] - 1
    count_flash = 0

    has_flashed = np.zeros_like(energies)
    indices_flash = np.array([np.array([], dtype=np.int), np.array([], dtype=np.int)])

    energies = energies + 1
    # energies = np.clip(energies, 0, 9)

    cond_to_iterate = len(np.where((energies > 9) * (has_flashed == 0))[0]) > 0
    while cond_to_iterate:
        # get indices that flashes (greater than 9, not yet flashed)
        indices_flash = np.where((energies > 9) * (has_flashed == 0))
        x_flash, y_flash = indices_flash
        has_flashed[indices_flash[0], indices_flash[1]] = 1

        # init to incremet structure
        ind_incr = np.array([np.array([], dtype=np.int), np.array([], dtype=np.int)])

        # get indices to increment
        for x, y in zip(x_flash, y_flash):
            ind_ = broadcast_indices(x, y, max_x, max_y)
            ind_incr = np.hstack([ind_incr, ind_])
        # increment
        for x, y in zip(ind_incr[0], ind_incr[1]):
            energies[x][y] += 1

        cond_to_iterate = (len(np.where((energies > 9) * (has_flashed == 0))[0])) > 0

    has_flashed_ind = np.where(has_flashed == 1)
    for x, y in zip(has_flashed_ind[0], has_flashed_ind[1]):
        energies[x][y] = 0

    count_flash += np.count_nonzero(has_flashed)

    return energies, count_flash
    # for x in range(max_x):
    #     for y in range(max_y):


def is_cap_letter(letter):
    return letter in {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z'}


def is_big_cave(cave):
    return is_cap_letter(cave[0])


def is_small_cave(cave):
    return not (is_big_cave(cave))


def is_complete_path(path):
    return path[-1] == 'end'


def contains_twice_small_cave(path):
    small_path = [c for c in path if is_small_cave(c)]
    return len(small_path) == len(set(small_path))


def get_all_possible_paths(junctions):
    path_hashes = set({})  # list of the hash of found paths

    partial_paths = [['start', ], ]  # list of partial paths (a partial path is a list of caves;)
    partial_double_small_cave = [False, ]
    partial_path_hashes = [hash(tuple(['start', ]))]

    filename = "complete_paths.txt"
    with open(filename, 'w+'):
        pass

    while (len(partial_paths)) > 0:
        cur_path = partial_paths[0]
        partial_paths = partial_paths[1:]
        partial_path_hashes = partial_path_hashes[1:]

        cur_double_small_cave = partial_double_small_cave[0]
        partial_double_small_cave = partial_double_small_cave[1:]

        # double_small_cave = contains_twice_small_cave(cur_path)

        next_possible_caves = [cave for cave in junctions[cur_path[-1]]
                               if (is_big_cave(cave) or
                                   (is_small_cave(cave) and not (cave in cur_path)) or
                                   (is_small_cave(cave) and cur_double_small_cave))]
        # print("Next possible caves from {} --> {}".format(cur_path[-1], next_possible_caves))
        new_possible_paths = [cur_path + [new_cave] for new_cave in next_possible_caves]
        # print("New possible paths = {}".format(new_possible_paths))
        for p in new_possible_paths:
            h = hash(tuple(p))
            if is_complete_path(p):
                if not (h in path_hashes):
                    path_hashes.add(h)
                    # with open(filename, 'a+') as f:
                    #     f.write(str(p))
                    #     f.write("\n")
            else:
                if not (h in partial_path_hashes):
                    partial_paths += [p]
                    partial_path_hashes += [h]
                    partial_double_small_cave += [contains_twice_small_cave(p)]

    return len(path_hashes)


def fold(map, orientation, k):
    """
    :param orientation: {x, y}
    :param k: line around which to fold
    :return:
    """
    xmax, ymax = map.shape

    if orientation == 'x':
        map_base = map[:, 0:k]
        map_to_be_folded = map[:, k + 1:]
        map_to_be_folded = np.fliplr(map_to_be_folded)
    elif orientation == 'y':
        map_base = map[0:k, :]
        map_to_be_folded = map[k + 1:, :]
        map_to_be_folded = np.flipud(map_to_be_folded)

    # print("Map Base =\n{}".format(map_base.shape))
    # print("Map to be Folded =\n{}".format(map_to_be_folded.shape))

    map_combined = combine(map_base, map_to_be_folded)

    return map_combined


def combine(map_base, map_folded):
    map_combined = map_base
    for x in range(map_base.shape[1]):
        for y in range(map_base.shape[0]):
            if (map_base[y, x] == '#' or map_folded[y, x] == '#'):
                map_combined[y, x] = '#'

    return map_combined


def update_pair_counts(pair_count, database):
    # pairs at the beginning of the step
    pairs = [p for p in pair_count.keys() if pair_count[p] > 0]
    pair_count_bck = copy.deepcopy(pair_count)
    for p in pairs:
        k = pair_count_bck[p]
        pair_count[p] -= k
        first_new_pair = p[0] + database[p]
        second_new_pair = database[p] + p[1]
        pair_count[first_new_pair] += k
        pair_count[second_new_pair] += k
    return pair_count


def insert_pairs(start_code, counts, database):
    return insert_pairs_rec(start_code, database, counts, "")


def insert_pairs_rec(start_code, database, counts, acc):
    if len(start_code) == 1:
        if acc == "":
            counts[start_code[0]] += 1
        return acc + start_code[0], counts
    else:
        current_pair = start_code[0:2]
        print("current_pair = {}".format(current_pair))

        new_start_code = start_code[1:]
        acc += (current_pair[0] + database[current_pair])
        counts[current_pair[0]] += 1
        counts[database[current_pair]] += 1
        # print("current_pair: {}".format(current_pair))
        return insert_pairs_rec(new_start_code, database, counts, acc)


def remove_loops(partial_path, new_candidates):
    pp_tmp = (partial_path[0], partial_path[1])
    cands = [cand for cand in new_candidates if (cand != pp_tmp)]
    return cands


def is_goal(pp, risk_levels):
    return (pp[0] == (risk_levels.shape[0] - 1)) and (pp[1] == (risk_levels.shape[1] - 1))


def hamming(x0, y0, risk_levels):
    xt = risk_levels.shape[0] - 1
    yt = risk_levels.shape[1] - 1
    d = xt - x0 + yt - y0
    return d


def _build_hamming_matrix(risk_levels):
    hammings = np.zeros_like(risk_levels)
    for i in range(risk_levels.shape[0]):
        for j in range(risk_levels.shape[1]):
            hammings[i][j] = hamming(i, j, risk_levels)
    return hammings


def new_way(risk_levels, orientation="up_left"):
    # hammings = _build_hamming_matrix(risk_levels)
    costs = np.ones_like(risk_levels) * 9999999
    costs[0][0] = 0
    max_i = risk_levels.shape[0] - 1
    max_j = risk_levels.shape[1] - 1
    # if orientation == "up_left":
    #     for i in range(1, max_i + 1):
    #         neighbors = [costs[i - 1][0]]  # costs[min(i+1, max_i)][0], costs[i][1]]
    #         costs[i][0] = risk_levels[i, 0] + min(neighbors)
    #     for j in range(1, max_j + 1):
    #         neighbors = [costs[0][j - 1]]  # , costs[0][min(j+1, max_j)], costs[1][j]]
    #         costs[0][j] = risk_levels[0, j] + min(neighbors)
    #     for i in range(1, max_i + 1):
    #         for j in range(1, max_j + 1):
    #             neighbors = [costs[i - 1][j], costs[i][j - 1]]
    #             costs[i][j] = risk_levels[i, j] + min(neighbors)
    back = np.zeros_like(costs)
    while not (np.all(back == costs)):
        back = copy.deepcopy(costs)
        for i in range(1, max_i):
            neighbors = [costs[i - 1][0], costs[min(i + 1, max_i)][0], costs[i][1]]
            tmp = risk_levels[i, 0] + min(neighbors)
            costs[i][0] = min([costs[i][0], tmp])
        for j in range(1, max_j):
            neighbors = [costs[0][j - 1], costs[0][min(j + 1, max_j)], costs[1][j]]
            tmp = risk_levels[0, j] + min(neighbors)
            costs[0][j] = min([tmp, costs[0][j]])
        for i in range(1, max_i + 1):
            for j in range(1, max_j + 1):
                neighbors = [costs[i - 1][j], costs[i][j - 1], costs[min(i + 1, max_i)][j], costs[i][min(j + 1, max_j)]]
                tmp = risk_levels[i, j] + min(neighbors)
                costs[i][j] = min([tmp, costs[i][j]])
    print(costs)

    return costs


def get_lowest_risk_path(risk_levels):
    queue = [[0, 0, hamming(0, 0, risk_levels), 0]]
    while (len(queue) > 0) and not is_goal(queue[0], risk_levels):
        pp = queue[0]
        queue = queue[1:]
        # print("Partial path =\n{}".format(pp))

        new_candidates = list(get_next_possible_state(pp[0], pp[1], risk_levels))
        new_candidates = remove_loops(pp, new_candidates)
        print("New candidates =\n{}".format(new_candidates))
        new_path_costs = [pp[3] + risk_levels[c[0], c[1]] for c in new_candidates]
        new_path_heur = [hamming(c[0], c[1], risk_levels) for c in new_candidates]

        # print("new_path_costs =\n{}".format(new_path_costs))

        new_paths = [(new_candidates[i][0],
                      new_candidates[i][1],
                      new_path_heur[i],
                      new_path_costs[i]) for i in range(len(new_path_costs))]

        # queue += [pp + [np] for np in new_paths]
        queue += new_paths
        queue.sort(key=lambda x: x[2] + x[3])
        print("New Queue sorted =\n{}".format(queue))
        queue = remove_redundant(queue)
        print(queue)

        # print(len(queue))

    return queue[0]


def remove_redundant(queue):
    for qq in queue:
        for pp in queue:
            # print("pp = {}".format(pp))
            # print("qq = {}".format(qq))
            if pp[0] == qq[0] and pp[1] == qq[1] and pp[2] >= qq[2]:
                # print("pp to remove : {}".format(pp))
                queue.remove(pp)

    return queue


def get_next_possible_state(cur_x, cur_y, risk_levels):
    max_x = risk_levels.shape[0] - 1
    max_y = risk_levels.shape[1] - 1
    min_x = 0
    min_y = 0
    return {(min(cur_x + 1, max_x), cur_y), (max(min_x, cur_x - 1), cur_y), (cur_x, min(cur_y + 1, max_y)),
            (cur_x, max(min_y, cur_y - 1))}


#
# def convert_bits_message(hex_message):
#     packet_version = hex_message[0:3]
#     packet_id = hex_message[3:6]
#
#     if packet_version == 4:
#

def zero_only(chars):
    return np.all(["0" == c for c in chars])


def simplify_expressions(operations, acc, counter_b1, counter_b2, last):
    # print("last = {}".format(len(last)))
    if len(operations) == 0:
        return acc
    expr = operations[0]
    operations = operations[1:]
    if expr == "b1":
        to_be_added = get_ref_elem_to_be_added(acc, last)
        to_be_added += [[]]
        last += ["b1"]
        return simplify_expressions(operations, acc, counter_b1, counter_b2, last)
    elif expr == "b2":
        to_be_added = get_ref_elem_to_be_added(acc, last)
        to_be_added += [[]]
        last += ["b2"]
        return simplify_expressions(operations, acc, counter_b1, counter_b2, last)
    elif expr == "e1":
        last = last[:-1]
        # acc += [[]]
        return simplify_expressions(operations, acc, counter_b1, counter_b2, last)
    elif expr == "e2":
        # last = last[:-1]
        return simplify_expressions(operations, acc, counter_b1, counter_b2, last)
    elif expr.__class__.__name__ == 'tuple':
        if len(last) == 0:
            to_be_added = get_ref_elem_to_be_added(acc, last)
        elif last[-1] == "b2" and counter_b2[-1] > 0:
            to_be_added = get_ref_elem_to_be_added(acc, last)
            counter_b2[-1] -= 1
        elif last[-1] == "b2" and counter_b2[-1] == 0:
            counter_b2 = counter_b2[:-1]
            last = last[:-1]
            to_be_added = get_ref_elem_to_be_added(acc, last)
        elif last[-1] == "b1":
            # acc[-1].append(expr)
            to_be_added = get_ref_elem_to_be_added(acc, last)

        to_be_added += [(expr[0],)]

        if expr[1] is not None:
            counter_b2 += [expr[1]]
        elif expr[2] is not None:
            counter_b1 += [expr[2]]
        return simplify_expressions(operations, acc, counter_b1, counter_b2, last)
    else:
        if last[-1] == "b2" and counter_b2[-1] > 0:
            to_be_added = get_ref_elem_to_be_added(acc, last)
            to_be_added += [expr]
            counter_b2[-1] -= 1

        elif last[-1] == "b2" and counter_b2[-1] == 0:
            counter_b2 = counter_b2[:-1]
            last = last[:-1]
            to_be_added = get_ref_elem_to_be_added(acc, last)
            to_be_added += [expr]
        elif last[-1] == "b1":
            # acc[-1].append(expr)
            to_be_added = get_ref_elem_to_be_added(acc, last)
            to_be_added += [expr]
        return simplify_expressions(operations, acc, counter_b1, counter_b2, last)


def get_ref_elem_to_be_added(acc, last):
    if len(last) == 0:
        return acc
    if len(last) == 1:
        return acc[-1]
    else:
        tmp_acc = acc
        for i in range(len(last)):
            tmp_acc = tmp_acc[-1]
        return tmp_acc


def eval_tree(tree, cur_res, waiting_args):
    if len(tree) == 0:
        return cur_res
    else:
        op = tree[0]
        args = tree[1]
        cond = all([isinstance(item, int) for item in args])
        print("condition list of ints: {}".format(cond))
        if cond:
            cur_res += [evaluate_operator(op, args)]
            return eval_tree(tree[2:], cur_res, waiting_args)
        else:
            print("now what: {}{}{}".format(op, args, tree[2:]))
            # if isinstance(args[0], int):
            #     waiting_args += [args[0]]


def decode_packet(packet, acc, modes, counter_lens, counter_packets, level, sum_versions):
    print(
        "counter_lens; counter_packets;level  = {},{},{}\n\tAcc = {}".format(counter_lens, counter_packets, level, acc))
    if len(modes):
        if len(counter_lens) > 0 and counter_lens[-1] == 0 and modes[-1] == "lens":
            acc += ["e1"]
            counter_lens = counter_lens[:-1]
            modes = modes[:-1]
            level -= 1
            return decode_packet(packet,
                                 acc,
                                 modes,
                                 counter_lens,
                                 counter_packets,
                                 level,
                                 sum_versions)
        if len(counter_packets) > 0 and counter_packets[-1] == 0 and modes[-1] == "packs":
            acc += ["e2"]
            counter_packets = counter_packets[:-1]
            modes = modes[:-1]
            level -= 1
            return decode_packet(packet,
                                 acc,
                                 modes,
                                 counter_lens,
                                 counter_packets,
                                 level,
                                 sum_versions)
    if len(packet) == 0:
        return acc, sum_versions
    if zero_only(packet):
        return acc, sum_versions

    version = bin2int(packet[0:3])
    sum_versions += version

    id = bin2int(packet[3:6])
    header_size = 6
    if id == 4:
        value_type = "literal"
    else:
        value_type = "operator"

    if value_type == "literal":
        # todo: padding with 0 to match modulo 4 ??
        remaining = packet[header_size:]
        rem_splits = [remaining[i:i + 5] for i in range(0, len(remaining), 5)]
        index_last = 0
        for sp in rem_splits:
            if sp[0] == '1':
                index_last += 1
            if sp[0] == '0':
                index_last += 1
                break
        rem_splits = rem_splits[0:index_last]
        rem_splits_bin = [spl[1:] for spl in rem_splits]
        dec = bin2int("".join(rem_splits_bin))
        acc.append(dec)
        if modes[-1] == "lens":
            counter_lens[-1] -= (header_size + 5 * len(rem_splits))
        elif modes[-1] == "packs":
            counter_packets[-1] -= 1
        return decode_packet(packet[6 + 5 * index_last:],
                             acc,
                             modes,
                             counter_lens,
                             counter_packets,
                             level,
                             sum_versions)

    elif value_type == "operator":
        level += 1

        length_type_id_index = 6
        length_type_id = packet[length_type_id_index]
        if length_type_id == '0':
            modes += ["lens"]
            total_length = packet[length_type_id_index + 1:
                                  length_type_id_index + 1 + 15]
            total_length_dec = bin2int(total_length)
            # subpackets = packet[length_type_id_index + 1 + 15:
            #                     length_type_id_index + 1 + 15 + total_length_dec]
            #
            # remainder = packet[length_type_id_index + 1 + 15 + total_length_dec:]

            acc += [(id, None, total_length_dec), "b1"]
            subpackets = packet[length_type_id_index + 1 + 15:]
            counter_lens += [total_length_dec]
            decoded = decode_packet(subpackets,
                                    acc,
                                    modes,
                                    counter_lens,
                                    counter_packets,
                                    level,
                                    sum_versions)
            tmp = decoded[0]  # + ["e1"]
            sum_versions = decoded[1]
            acc = tmp
            return decoded[0], decoded[1]
            # return decode_packet(remainder,
            #                      acc,
            #                      modes,
            #                      counter_lens,
            #                      counter_packets,
            #                      sum_versions)
        if length_type_id == '1':
            modes += ["packs"]
            nb_of_subpackets = packet[length_type_id_index + 1:
                                      length_type_id_index + 1 + 11]
            nb_of_subpackets = bin2int(nb_of_subpackets)
            # print("nb of subpackets = {}".format(nb_of_subpackets))
            subpackets = packet[length_type_id_index + 1 + 11:]
            # acc += ['[']
            acc += [(id, nb_of_subpackets, None), 'b2']
            counter_packets += [nb_of_subpackets]
            decoded = decode_packet(subpackets,
                                    acc,
                                    modes,
                                    counter_lens,
                                    counter_packets,
                                    level,
                                    sum_versions)
            tmp = decoded[0]  # + ["e2"]
            return tmp, decoded[1]


def evaluate_operator(type_id, expression):
    print("type id as it comes:{}".format(type_id))
    print("expression as it comes:{}".format(expression))
    # expression = expression[0]
    if type_id == 0:
        return np.sum(expression)
    if type_id == 1:
        return np.prod(expression)
    if type_id == 2:
        return np.min(expression)
    if type_id == 3:
        return np.max(expression)
    if type_id == 5:
        if expression[0] > expression[1]:
            return 1
        else:
            return 0
    if type_id == 6:
        if expression[0] < expression[1]:
            return 1
        else:
            return 0
    if type_id == 7:
        if expression[0] == expression[1]:
            return 1
        else:
            return 0
    else:
        raise ValueError(type_id)


#
# if len(rem_splits[-1]) != 5:
#     rem_splits=rem_splits[0:-1]
# print("Rem Splits = {}".format(rem_splits))

# while np.mod(len(remaining),4) != 0:
#     remaining = "0"+remaining


HB = {
    "0": "0000",
    "1": "0001",
    "2": "0010",
    "3": "0011",
    "4": "0100",
    "5": "0101",
    "6": "0110",
    "7": "0111",
    "8": "1000",
    "9": "1001",
    "A": "1010",
    "B": "1011",
    "C": "1100",
    "D": "1101",
    "E": "1110",
    "F": "1111"
}


def hex2bin(str_chain):
    res = []
    str_chain.rstrip("0")
    for c in str_chain:
        res += [HB[c]]
    res = "".join(res)
    return res


def do_step_backward(x, y, vx, vy, vx_rem):
    vx_rem = vx_rem + 1
    prev_vx = np.max([vx_rem, 0])
    prev_vy = vy + 1
    prev_x = x - vx
    prev_y = y - vy
    return prev_x, prev_y, prev_vx, prev_vy, vx_rem


def get_max_height(target_xs, target_ys):
    data = {}
    cur_best = 0
    for i in np.arange(target_xs[0], target_xs[1]):
        for j in np.arange(target_ys[0], target_ys[1]):
            for vi_rem in np.arange(-2 * len(np.arange(target_xs[0], target_xs[1])),
                                    0):  # 2 * len(np.arange(target_xs[0], target_xs[1]))):
                for vj in np.arange(-2 * len(np.arange(target_ys[0], target_ys[1])), 0):
                    x = i
                    y = j
                    vx_rem = vi_rem
                    vx = np.max([vx_rem, 0])
                    vy = vj
                    cond = need_to_stop(x, y, vx, vy, vx_rem, target_xs, target_ys, cur_best)
                    h0 = 0

                    # print("({},{}) with speed {} [{}]; {} ".format(x, y, vx, vx_rem, vy))
                    while not cond:
                        h0 = np.max([h0, y])

                        x, y, vx, vy, vx_rem = do_step_backward(x, y, vx, vy, vx_rem)

                        cond = need_to_stop(x, y, vx, vy, vx_rem, target_xs, target_ys, cur_best)
                        # if x == 0 and y == 0:
                        if y == 0:
                            if (vx, vy) in data.keys():
                                data[(vx, vy)] = np.max([h0, data[(vx, vy)]])
                            else:
                                data[(vx, vy)] = h0
                                if h0 > cur_best:
                                    cur_best = h0
                                    print("current best = {}".format(cur_best))

    print(cur_best)
    return max(data.values())


def do_step_forward(x, y, vx, vy):
    vx = max(vx - 1, 0)
    vy = vy - 1
    x = x + vx
    y = y + vy
    return x, y, vx, vy


def need_to_stop(x, y, vx, vy, vx_rem, target_x, target_y, cur_best):
    if (x > target_x[1]) and (vx >= 0):
        print("cond d'arret en X")
        return True
    if (y < target_y[0]) and (vy <= 0):
        print("cond d'arret en Y")
        return True
    if x < 0:
        return True
    if vx < 0:
        return True
    if vy == 0 and y < cur_best:
        return True
    if vx_rem > 0 and y < cur_best:
        return True

    return False


def build_linked_list(frm, to, base_dict, acc):
    if frm == '0':
        return acc
    ks = list(base_dict.keys())
    if (frm, to) in ks:
        acc += [(frm, to)]
        return acc
    else:
        for it in ks:
            if it[0] == frm:
                acc += [(frm, it[1])]
                return build_linked_list(it[1], to, base_dict, acc)
            # else:
            #     raise ValueError("Impossible to find an entry for frm == >{}<".format(it[0]))


def build_path_to_ref(frm, to, base_dict):
    goal_is_reached = False
    queue = [[(frm, k)] for k in base_dict[frm]]
    # print("starting queue = {}".format(queue))
    goal_is_reached = queue[0][-1][1] == to
    # print("end of first path = {}".format(queue[0][-1][1] ))
    # print(f"len(queue) : {len(queue)}")
    while not (len(queue) == 0 or goal_is_reached):
        partial_path = queue[0]
        queue = queue[1:]
        new_paths = [partial_path + [(partial_path[-1][1], k)] for k in base_dict[partial_path[-1][1]]
                     if (partial_path[-1][1], k) not in partial_path]
        # print(new_paths)
        queue = queue + new_paths
        # print(queue[0])
        if queue[0][-1][1] == to:
            goal_is_reached = True

    # print("end of first path = {}".format(queue[0][-1][1] ))
    # print(f"len(queue) : {len(queue)}")
    if goal_is_reached:
        return queue[0]
    else:
        raise ValueError("not found goal")


def convert_base(inp, rotAB, signAB, tAB):
    """

    :param inp: vector input as given in input data
    :param rotAB: rotation matrix from base A to base B
    :param signAB: sign vector from based A to base B
    :param tAB: origin
    :return: input in base B
    """

    return np.matmul(inp, rotAB) * signAB - tAB


def convert_to_ref(inps, path_to_ref, rotations, signs, ts):
    res = inps
    for p in path_to_ref:
        res = convert_base(res, rotations[p], signs[p], ts[p])
    return res


def manhattan(x, y):
    return np.sum(np.abs(x - y))


def manhattan_2d(x1, x2):
    return np.sum(np.abs(x1[0] - x2[0]) + np.abs(x1[1] - x2[1]))

    # #############
    # #...........#
    # ###B#C#A#D###
    #    B#D#C#A
    #    #######


def distance_pod(p1, p2):
    dist_x = abs(p1[1] - p2[1])
    dist_y = 0
    if (p1[0] == 0) and (p2[0] == 0):
        dist_y = 0
    elif p1[0] == p2[0] and p1[1] == p2[1]:
        dist_y = 0
    elif p1[1] == p2[1]:
        dist_y = abs(p2[0] - p1[0])
    else:
        # in (x1, {1,2}), (x2, {1,2}) with p1[1] != p2[1]
        dist_y = p1[0] + p2[0]
    assert dist_y >= 0, "Distance y > 0"

    return dist_x + dist_y
