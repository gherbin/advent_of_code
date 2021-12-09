from aoc_2021.source.Bingo import Bingo
from aoc_2021.source.Display import Display
from aoc_2021.source.LanternFish import FastFish
from aoc_2021.source.Submarine import Submarine, SubmarineWrapper
from aoc_2021.source.VentLines import build_vents_map
from aoc_utils.source import *


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
