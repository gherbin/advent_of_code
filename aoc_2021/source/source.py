from aoc_2021.source.Bingo import Bingo
from aoc_2021.source.LanternFish import Fishes, get_fishes, FastFish
from aoc_2021.source.Submarine import Submarine, SubmarineWrapper
from aoc_2021.source.VentLines import build_vents_map
from aoc_utils.source import *


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
        print(bingo.boards[np.argmax(bingo.runs),:,:])
        print(bingo.boards_binary[np.argmax(bingo.runs),:,:])
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
    overlapping_points = np.where(map>1)

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
