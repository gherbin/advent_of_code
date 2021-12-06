from aoc_utils.source import *

def get_fishes(line):
    line = np.array(line)
    d = [len(np.where(line == a)[0]) for a in range(9)]
    return d


class FastFish:
    def __init__(self, line):
        self.fishes = get_fishes(line)

    def model_day(self):
        to_add = self.fishes[0]
        self.fishes = self.fishes[1:] + [to_add]
        self.fishes[6] += to_add


class Lanternfish:
    def __init__(self, countdown=8):
        self.countdown = countdown

    def model_day(self):
        self.countdown -= 1
        if self.countdown == -1:
            self.countdown = 6


class Fishes:
    def __init__(self, initial_status):
        self.fishes = [Lanternfish(f) for f in initial_status]
        self.fishes_to_add = []

    def model_day(self):
        for fish in self.fishes:
            fish.model_day()
        self.fishes += copy.deepcopy(self.fishes_to_add)
        self.fishes_to_add = []
        for _ in range(len([f for f in self.fishes if f.countdown == 0])):
            self.fishes_to_add.append(Lanternfish())
        # print([fta.countdown for fta in self.fishes_to_add])
        # print(self.get_countdowns())
        # print("-----------------------------------------------------------------")

    def get_countdowns(self):
        return [f.countdown for f in self.fishes]

    def get_nb_of_fishes(self):
        return len(self.fishes)


