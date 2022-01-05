import itertools
from itertools import chain

from aoc_2021.source import launcher
from aoc_2021.source.Bingo import Bingo
from aoc_2021.source.DeterministicDice import *
from aoc_2021.source.Display import Display
from aoc_2021.source.LanternFish import FastFish
from aoc_2021.source.Scanner import Scanner
from aoc_2021.source.SnailFishNumber import constructor, test_find_regular, test_explode, debug_regular, test_split, \
    test_addition, test_magnitude
from aoc_2021.source.Submarine import Submarine, SubmarineWrapper
from aoc_2021.source.VentLines import build_vents_map
from aoc_2021.source.bits_parser import bin_parser
from aoc_2021.source.cuboids import run
from aoc_2021.source.cucumber import Cucumber
from aoc_2021.source.image_enhancement import get_roi, c_2int, expand_image, iae, pretty_print, crop_image
from aoc_2021.source.monad import ALU
from aoc_utils.source import *
from aoc_2021.source.amphipods import game
from matplotlib import pyplot as plt


def day25():
    with open(r"aoc_2021/inputs/day25.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [[c for c in line.strip()] for line in lines]

    cub = Cucumber(lines)
    cub_emp_prev = {}
    counter = 0
    while cub.emp != cub_emp_prev:
        cub_emp_prev = cub.emp.copy()
        cub.do_step()
        counter += 1
    print("Nb of steps = {}".format(counter))

def day24():
    with open(r"aoc_2021/inputs/day24.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [line.strip() for line in lines]
    alu = ALU()

    # solved using MiniZinc ;-)


def day23():
    with open(r"aoc_2021/inputs/day23_dummy.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [line.strip() for line in lines]
    print(lines)
    game()
    return


def day22():
    print("2021 - Day 22")
    return run()


def day21():
    print("2021 - Day 21")
    starts = [4, 8]
    dice = np.arange(1, 101)
    game = DiceGame(starts)
    cond = False
    while not cond:
        cond = game.do_turn()
        print(game)
        print(game.player1)
        print(game.player2)
        print("------------")
    print(game.ddice)
    return count_wins(7, 0, 6, 0)


def day20():
    print("2021 - Day 20")
    with open(r"aoc_2021/inputs/day20.txt", 'r') as file_input:
        lines = file_input.readlines()
    encoded_chars = lines[0]
    inp_map = [l.strip() for l in lines[2:]]
    input_as_list = []
    for ind in range(len(inp_map)):
        input_as_list += [[c for c in inp_map[ind]]]

    pretty_print(input_as_list)

    test_im = expand_image(input_as_list, expansion=100)

    for _ in range(50):
        print(f"Doing : {_}")
        test_im = iae(test_im, encoded_chars, expansion=0)
        test_im = crop_image(test_im, 1)
    pretty_print(test_im)

    flattened = [c for sublist in test_im for c in sublist]
    bin = [1 if c == "#" else 0 for c in flattened]
    count = np.count_nonzero(bin)
    return count


def day19():
    print("2021 - Day 19")
    with open(r"aoc_2021/inputs/day19.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [line.strip() for line in lines]
    scanners = []
    for line in lines:
        if "scanner" in line:
            id = line.split("scanner ")[1].strip(" ---")
            # print("id found = {}".format(id))
            scanners.append(Scanner(id))
        elif line == "":
            continue
        else:
            scanners[-1].coordinates.append([int(c) for c in line.split(",")])

    for sc in scanners:
        sc.init()
    print(scanners[0].coordinates)

    cnt = 0
    done = {}
    for pair in itertools.permutations(range(len(scanners)), 2):
        done[pair] = False

    ps = {}
    for sc in scanners:
        ps[sc.id] = set()

    rotations = {}
    ts = {}
    signs = {}

    for pair in itertools.permutations(range(len(scanners)), 2):
        if not done[pair]:
            done[(pair[0], pair[1])] = scanners[pair[0]].matches(scanners[pair[1]])
            if done[pair]:
                rotations[(str(pair[0]), str(pair[1]))] = scanners[pair[0]].rotation
                ts[(str(pair[0]), str(pair[1]))] = scanners[pair[0]].origin
                signs[(str(pair[0]), str(pair[1]))] = scanners[pair[0]].signs

                ps[str(pair[0])].add(str(pair[1]))
                ps[str(pair[1])].add(str(pair[0]))

    for k, v in ts.items():
        print("{}:{}".format(k, v))

    path_to_refs = {}
    ref = '0'
    for sc in scanners:
        if sc.id != '0':
            path_to_refs[sc.id] = build_path_to_ref(sc.id, ref, ps)

    beacons = scanners[0].coordinates

    origins = {}
    origins['0'] = np.array([0, 0, 0])  # scanners[0].origin
    for sc in scanners[1:]:
        inps = sc.coordinates
        points = convert_to_ref(inps, path_to_refs[sc.id], rotations, signs, ts)
        beacons = np.concatenate([beacons, points], axis=0)
        origins[sc.id] = convert_to_ref(np.array([[0, 0, 0]]), path_to_refs[sc.id], rotations, signs, ts)

    print(beacons.shape)
    beacons = np.unique(beacons, axis=0)
    print(beacons.shape)

    distances = {}
    print(origins)
    for pair in itertools.combinations(range(len(scanners)), 2):
        ind0 = str(pair[0])
        ind1 = str(pair[1])
        distances[(ind0, ind1)] = manhattan(origins[ind0], origins[ind1])

    print(distances)
    print("Max distances = {}".format(np.max([d for d in distances.values()])))
    return


def day18():
    print("2021 - Day 18")
    with open(r"aoc_2021/inputs/day18.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [line.strip() for line in lines]
    lines = [eval(line) for line in lines]
    print("Nb of lines = {}".format(len(lines)))
    # part 1
    # sn = constructor(lines[0])
    # for L in lines[1:]:
    #     sn = sn.add(constructor(L))
    # print(sn.get_magnitude())
    # part 2
    res = []
    for l1, l2 in itertools.combinations(lines, 2):
        sn1 = constructor(l1)
        sn2 = constructor(l2)

        sn12 = sn1.add(constructor(l2))
        sn21 = sn2.add(constructor(l1))
        res1 = sn12.get_magnitude()
        res2 = sn21.get_magnitude()
        print(f"{res1},{res2}")
        res += [res1, res2]

    # test_magnitude()
    # test_find_regular()
    # test_explode()
    # debug_regular()
    # test_split()
    # test_addition()
    return np.max(res)


def day17():
    print("2021 - Day 17")
    with open(r"aoc_2021/inputs/day17.txt", 'r') as file_input:
        lines = file_input.readlines()
    line = lines[0]
    target_info = line.split("=")
    target_xs = np.array([int(x) for x in target_info[1].split(",")[0].split("..")])
    target_ys = np.array([int(y) for y in target_info[2].split("..")])
    print("Target info = {}{}".format(target_xs, target_ys))
    launcher.part1()
    launcher.part2()
    return


def day16():
    print("2021 - Day 16")
    with open(r"aoc_2021/inputs/day16.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [line.strip() for line in lines]
    for message in lines:
        # message = lines[0]
        print(message)
        bin_message = hex2bin(message)
        print(len(message))
        print(len(bin_message))
        while len(bin_message) < 4 * len(message):
            bin_message = '0' + bin_message
        score = bin_parser(0, 0, bin_message)
    return score


def day15():
    print("2021 - Day 15")
    with open(r"aoc_2021/inputs/day15.txt", 'r') as file_input:
        lines = file_input.readlines()
    risk_levels = np.array([np.array([int(c) for c in line.strip()]) for line in lines])
    print(risk_levels)
    shape_init = risk_levels.shape
    risk_levels = np.tile(risk_levels, (5, 5))
    t = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])
    t = np.repeat(np.repeat(t, shape_init[1], axis=1), shape_init[0], axis=0)
    print(risk_levels.shape)
    print(t.shape)


    rt = np.mod(t + risk_levels, 9)
    ind0 = np.where(rt == 0)
    print(len(ind0[0]))
    rt[rt == 0] = 9
    ind0 = np.where(rt == 0)
    print(len(ind0[0]))

    costs = new_way(rt)
    score = costs[-1][-1]
    return score


def day14():
    print("2021 - Day 14")
    with open(r"aoc_2021/inputs/day14.txt", 'r') as file_input:
        lines = file_input.readlines()

    start_code = lines[0].strip()
    pairs = lines[2:]
    all_elements = set([pair.split("->")[1].strip() for pair in pairs])
    elements_counts = {elem: 0 for elem in all_elements}

    db = {}  # more efficient if O(1) ?
    for pair in pairs:
        splits = pair.split(" -> ")
        db[splits[0].strip()] = splits[1].strip()
    #
    print("Start code:\n{}".format(start_code))
    print("database  :\n{}".format(db))

    pairs_chunck = [start_code[i:i + 2] for i in range(0, len(start_code), 1) if len(start_code[i:i + 2]) == 2]
    pair_counts = {p: 0 for p in db.keys()}
    for pair in pairs_chunck:
        pair_counts[pair] += 1

    # pair_counts_tmp = {p:0 for p in db.keys()}
    max_steps = 40
    for step in range(1, max_steps + 1):
        pair_counts = update_pair_counts(pair_counts, db)

    for pair in pair_counts.keys():
        elements_counts[pair[0]] += pair_counts[pair]
        elements_counts[pair[1]] += pair_counts[pair]
    # compensate for
    elements_counts[start_code[0]] += 1
    elements_counts[start_code[-1]] += 1

    numbers = np.array([el for el in elements_counts.values()]) // 2
    numbers.sort()
    res = numbers[-1] - numbers[0]
    print(np.sum(numbers))
    return res


def day13():
    print("2021 - Day 13")
    with open(r"aoc_2021/inputs/day13.txt", 'r') as file_input:
        lines = file_input.readlines()

    fold_along = []
    coords = set({})
    for line in lines:
        if "," in line:
            line = line.split(",")
            x, y = int(line[0]), int(line[1])
            coords.add((x, y))
        elif line.startswith("fold along"):
            line_split = line.split("=")
            fold_along += [(line_split[0][-1], int(line_split[1]))]

    map_coords = [['.'] * (1 + max([c[0] for c in coords]))] * (1 + max([c[1] for c in coords]))
    map_coords = np.array(map_coords, dtype=str)

    map_coords[[c[1] for c in coords], [c[0] for c in coords]] = '#'
    new_map_coords = map_coords
    for orient, k in fold_along:
        new_map_coords = fold(new_map_coords, orient, k)

    to_show_map = np.zeros_like(new_map_coords, dtype=int)
    to_show_map[new_map_coords == '#'] = 1
    plt.imshow(to_show_map)
    plt.show()


def day12():
    print("2021 - Day 12")
    with open(r"aoc_2021/inputs/day12_dummy.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split('-') for line in lines]
    all_caves = [cave for line in lines for cave in line]
    junctions = {cave: [] for cave in all_caves}
    for junct in lines:
        junctions[junct[0]] += [junct[1]]
        junctions[junct[1]] += [junct[0]]

    for k in junctions.keys():
        junctions[k] = [cave for cave in junctions[k] if cave != 'start']

    print(junctions)
    # junctions = all possibles junctions from cave to cave
    counter = get_all_possible_paths(junctions)
    # print(lines)
    # print(junctions)
    return counter


def day11():
    print("2021 - Day 11")
    with open(r"aoc_2021/inputs/day11.txt", 'r') as file_input:
        lines = file_input.readlines()

    energies = np.array([np.array([int(c) for c in line.strip()]) for line in lines])
    counter = 0
    for step in range(1, 101):
        energies, counter_tmp = octopus_flashes(energies)
        counter += counter_tmp
        # print("Step {}: {}\n{}".format(step, counter, energies))

    # part 2
    step_full = 0
    for step in range(1, 1000):
        energies, counter_tmp = octopus_flashes(energies)
        # print("Step {}: {}\n{}".format(step, counter_tmp, energies))

        if counter_tmp == 100:
            step_full = step + 100  # because of part 1
            print(step_full)
            break

    return counter, step_full


def day10():
    print("2021 - Day 10")
    with open(r"aoc_2021/inputs/day10.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [[d for d in line.strip()] for line in lines]
    res, lines_cleaned = compute_score_illegal_chars(lines)

    # part 2
    autocomplete_chunks = [autocomplete(line) for line in lines_cleaned]
    autocomplete_chunks.sort()
    ind = len(autocomplete_chunks) // 2
    res2 = autocomplete_chunks[ind]

    return res, res2


def day9():
    print("2021 - Day 9")
    with open(r"aoc_2021/inputs/day9.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = np.array([np.array([int(d) for d in line.strip()]) for line in lines])

    # part 1
    is_low = np.zeros_like(lines)

    for x_ in np.arange(0, lines.shape[0]):
        for y_ in np.arange(0, lines.shape[1]):
            is_low[x_, y_] = is_low_point(lines, x_, y_)
    is_low = np.array(is_low, dtype=bool)
    X_masked = np.ma.masked_array(lines, ~is_low)

    print("risk level = ", np.sum(X_masked) + X_masked.count())

    # part 2
    # from all the low points, count the number of elements in all directions until a "9" is reached
    lows = np.where(is_low)  # lows is a tuple of ([xs], [ys]) of the low points
    cs = []
    for x_, y_ in zip(lows[0], lows[1]):
        print(x_, y_)
        c = count_around((x_, y_), lines)
        cs += [c]

    cs = list(np.sort(cs))
    cs.reverse()
    score = cs[0] * cs[1] * cs[2]
    return score


def day8():
    print("2021 - Day 8")
    with open(r"aoc_2021/inputs/day8.txt", 'r') as file_input:
        lines = file_input.readlines()
    lines = [line.strip().split("|") for line in lines]
    d = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # digit 1 --> 2 segments
    # digit 4 --> 4 segments
    # digit 7 --> 3 segments
    # digit 8 --> 7 segments

    # part 1
    samples = [l[1].split(" ") for l in lines]
    for digits in samples:
        for digit in digits:
            d[len(digit)] += 1

    print(d[2] + d[4] + d[3] + d[7])

    # part 2
    #  --x0--
    # x3    x1
    #  --x2--
    # x6    x4
    #  --x5--

    # print("line = {}".format(line))
    #
    decoded_digits = []
    for line in lines:
        display = Display()

        decoded_digit = []
        source = line[0].strip().split(" ")
        for code in source:
            display.process_code(code)
        for code in line[1].strip().split(" "):
            dig = display.decode(code)
            decoded_digit += [dig]
        decoded_digits += [1000 * decoded_digit[0] + 100 * decoded_digit[1] + 10 * decoded_digit[2] + decoded_digit[3]]

    return np.sum(np.array(decoded_digits))


def day7():
    print("2021 - Day 7")
    with open(r"aoc_2021/inputs/day7.txt", 'r') as file_input:
        lines = file_input.readlines()
    line = np.array([int(d) for d in lines[0].split(",")])

    horiz_min = np.min(line)
    horiz_max = np.max(line)
    possibilities = np.arange(horiz_min, horiz_max)
    sum_signs = 9999999
    best_candidate = horiz_min
    for p in possibilities:
        tmp = np.sum(np.sign(line - p))
        if sum_signs > np.abs(tmp):
            sum_signs = np.abs(tmp)
            best_candidate = p

    def compute_fuel(inputs, target):
        return np.sum(np.abs(inputs - target))

    fuel = compute_fuel(line, best_candidate)

    # part 2
    def compute_one_fuel_part2(input, target):
        return np.max(np.cumsum(np.arange(np.abs(input - target) + 1)))

    def compute_fuel_part2(inputs, target):
        return np.sum([compute_one_fuel_part2(inp, target) for inp in inputs])

    best_candidate = int(1 / len(line) * np.sum(line))
    fuel = compute_fuel_part2(line, best_candidate)

    return fuel


def day1():
    print("2021 - Day 1")
    d0 = float("inf")
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


def day2():
    print("2021 - Day 2")
    sub = Submarine()
    subwrap = SubmarineWrapper(sub)
    with open(r"aoc_2021/inputs/day2.txt", 'r') as file_input:
        lines = file_input.readlines()

    for l in lines:
        subwrap.move(l)
    return subwrap.submarine


def day3():
    print("2021 - Day 3")
    sub = Submarine()
    with open(r"aoc_2021/inputs/day3.txt", 'r') as file_input:
        lines = file_input.readlines()
    sub.compute_rates(lines)
    print(sub)


def day4():
    print("2021 - Day 4")
    score = None
    with open(r"aoc_2021/inputs/day4.txt", 'r') as file_input:
        lines = file_input.readlines()

    drawn_numbers = [int(n) for n in lines[0].split(",")]

    # boards = [n.strip('\n') for n in lines[1:]]
    # boards = [int(n) if n not in ['', ', '] for n in boards]
    boards = get_boards_from_input(lines[1:])

    def _part1():
        bingo = Bingo(boards)
        for d in drawn_numbers:
            bingo.mark_cell(d)
            res = bingo.is_bingo()
            if res:
                break
        score = bingo.compute_score()
        return score

    def _part2():
        bingo = Bingo(boards)
        for d in drawn_numbers:
            bingo.mark_cell(d)
            res = bingo.is_bingo()
            print(bingo.runs)
            if np.all(bingo.runs):
                break
        print(np.argmax(bingo.runs))

        score = bingo.compute_score(np.argmax(bingo.runs))
        print(bingo.boards[np.argmax(bingo.runs), :, :])
        print(bingo.boards_binary[np.argmax(bingo.runs), :, :])
        return score

    # score = _part1()
    score = _part2()
    # print(bingo.bingo)
    # print(np.sum((1-bingo.boards_binary[bingo.bingo_board,:,:])*bingo.boards[bingo.bingo_board,:,:]))
    return score


def day5():
    overlapping_points = 0
    print("2021 - Day 5")
    with open(r"aoc_2021/inputs/day5.txt", 'r') as file_input:
        lines = file_input.readlines()
    sources, targets = get_lines_from_input(lines)
    map = build_vents_map(sources, targets)
    overlapping_points = np.where(map > 1)

    return len(overlapping_points[0])


def day6():
    print("2021 - Day 6")
    with open(r"aoc_2021/inputs/day6.txt", 'r') as file_input:
        lines = file_input.readlines()
    line = [int(c[0]) for c in lines[0].split(",")]
    # print(line)

    # part 1: not optimized at all
    # fs = Fishes(line)
    # print(fs.get_countdowns())
    # for _ in range(80):
    #     fs.model_day()
    # return len(fs.get_countdowns())

    # part 2: fast fish -> keep only counts
    fs = FastFish(line)
    for _ in range(256):
        fs.model_day()
    return np.sum(fs.fishes)
