from aoc_utils.source import *


class DeterministicDice:
    def __init__(self):
        self.faces = range(1, 101)
        self.i = -1
        self.counter = 0

    def __next__(self):
        self.i = np.mod(self.i + 1, 100)
        res = self.faces[self.i]
        self.counter += 1
        return res

    def __call__(self):
        return self.__next__()

    def __repr__(self):
        return "DeterministicDice. Current Face = {}, counter = {}".format(self.faces[self.i], self.counter)

class Player:
    def __init__(self, id, space):
        self.id = id
        self.score = 0
        self.space = space - 1
        self.plateau = [n for n in range(1, 11)]

    def move(self, sp):
        self.space = np.mod(self.space + sp, 10)
        self.score += self.plateau[self.space]
        return self.plateau[self.space]

    def __repr__(self):
        return f"< Player {self.id}, score = {self.score}, space = {self.plateau[self.space]}>"


class DiceGame:
    def __init__(self, starting):
        self.player1 = Player(1, starting[0])
        self.player2 = Player(2, starting[1])
        self.ddice = DeterministicDice()
        self.current_player = self.player2

    def do_turn(self):
        if self.current_player == self.player1:
            self.current_player = self.player2
        elif self.current_player == self.player2:
            self.current_player = self.player1

        tmp = next(self.ddice)
        tmp += next(self.ddice)
        tmp += next(self.ddice)
        print("tmp = {}".format(tmp))
        self.current_player.move(tmp)
        return self.is_a_winner()

    def is_a_winner(self):
        if self.current_player.score >= 1000:
            return True
        return False

    def __repr__(self):
        return "< Game. Cur Player = {} >".format(self.current_player.id)


def to_index(i):
    return i - 1


# for i in range(len(locations_1)):
#     if locations_1[i] > 0:
#         print("init: position = {}, score = {}")
#         current_spot = i
#         for dice_res in dice_results:
#             new_spot = to_index(np.mod(current_spot + dice_res, 10))
#             new_score = score + plateau[new_spot]
#
#             print("Previous score = {};\n New score = {} for dice = {}".format(score, new_score, dice_res))


plateau = [_ for _ in range(1, 11)]
# score_occurences_1 = [0] * 22
# score_occurences_2 = [0] * 22
# locations_1 = [0] * 10
# locations_2 = [0] * 10
# locations_1[to_index(4)] = 1
# locations_2[to_index(8)] = 1
# dice_results = [3, 4, 5, 6, 7, 8, 9]
# dice_counts = [1, 3, 6, 7, 6, 3, 1]
# game = {}
# game1 = {}
# game2 = {}
# for pair in itertools.product(plateau, [_ for _ in range(0, 22)]):
#     game1[pair] = 0
#     game2[pair] = 0
# game1[(4, 0)] = 1
# game2[(8, 0)] = 1
#
# game1["win"] = 0
# game2["win"] = 0

# def do_step(active_player_dict, passive_player_dict):
#     game_dict_freeze = copy.deepcopy(active_player_dict)
#     keys_treated = []
#     for k in game_dict_freeze.keys():
#         if k == "win":
#             continue
#         cur_pos = k[0]
#         cur_score = k[1]
#         nb = game_dict_freeze[k]
#         # print("cur_pos: {}, cur_score: {}, nb : {}".format(cur_pos, cur_score, nb))
#         keys_treated.append(k)
#         if nb > 0:
#             active_player_dict[k] = 0
#             for id_dice in range(len(dice_results)):
#                 dice = dice_results[id_dice]  # result of the dice
#                 count = dice_counts[id_dice]  # number of time the result appears (quantum effect)
#                 new_pos = np.mod(to_index(cur_pos + dice), 10)
#                 new_score = cur_score + plateau[new_pos]
#
#                 if new_score >= 21:
#                     active_player_dict["win"] += count * nb
#                 else:
#                     # print("prev_pos {}, prev_score {}, counts {}, dice {} ==> new_pos+1 {}, new_score {}, new_counts {}" \
#                     #       .format(cur_pos, cur_score, count, dice, new_pos+1, new_score, count*nb))
#                     # print("game_dict before : {}".format( game_dict[(new_pos+1, new_score)]))
#                     active_player_dict[(new_pos + 1, new_score)] += count * nb
#                     for k in passive_player_dict.keys():
#                         if k == "win":
#                             continue
#                         passive_player_dict[k] += count*nb
#
#                     # print("game_dict [{}] after: {}".format((new_pos+1, new_score),
#                     #                                         game_dict[(new_pos+1, new_score)]))
#     # passive_player_dict = {k: 27 * v for k, v in passive_player_dict.items()}
#
#     # assert np.sum([game_dict[k] for k in game_dict.keys()]) == 27 * np.sum([game_dict_freeze[k] for k in game_dict_freeze.keys()])
#     return active_player_dict, passive_player_dict
#
#
# # print("win1 = {}".format(win1))
# # print(game1)
# # print(game1_stop)
# # print(game1 == game1_stop)
# def pretty_print_game(game_dict):
#     for item in game_dict.items():
#         # print(item[1])
#         if item[1] > 0:
#             print(item)
#     print("----------------")
# #
#
# def stopping_conditions(player_dict1, player_dict2):
#     if player_dict1["win"] > 444356092776315:
#         return True
#     if player_dict1["win"] == 0 or player_dict2["win"] == 0:
#         return False
#     ongoing_1 = np.sum([player_dict1[k] for k in player_dict1.keys() if k != "win"])
#     ongoing_2 = np.sum([player_dict1[k] for k in player_dict1.keys() if k != "win"])
#     if ongoing_1 == 0:
#         print("ongoing_1 = 0")
#         return True
#     if ongoing_2 == 0:
#         print("ongoing_2 = 0")
#         return True
#     return ongoing_2 + player_dict1["win"] == ongoing_1 + player_dict2["win"]
#
#
# passive = game1
# active = game2
# iter = 0
# while not stopping_conditions(game1, game2):
#     print("iter = {}".format(iter))
#     if active == game1:
#         active = game2
#         passive = game1
#     elif active == game2:
#         active = game1
#         passive = game2
#     active, passive = do_step(active_player_dict=active, passive_player_dict=passive)
#     # pretty_print_game(game1)
#     # pretty_print_game(game2)
#     iter += 1

#
# print("\tNb of wins game1 :{}".format(game1["win"]))
# print("\tNb of wins game2 :{}".format(game2["win"]))
#
#

# cheat page => https://www.youtube.com/watch?v=a6ZdJEntKkk&ab_channel=JonathanPaulson

result_db = {}  # data base of results

def count_wins(pos1, score1, pos2, score2):
    if (pos1, score1, pos2, score2) in result_db:
        return result_db[(pos1, score1, pos2, score2)]
    if score1 >= 21:
        return (1, 0)
    if score2 >= 21:
        return (0, 1)
    ans = (0, 0)
    for d1 in [1, 2, 3]:
        for d2 in [1, 2, 3]:
            for d3 in [1, 2, 3]:
                new_pos1 = np.mod((pos1 + d1 + d2 + d3), 10)
                new_score1 = score1 + plateau[new_pos1-1]
                w1, w2 = count_wins(pos2, score2, new_pos1, new_score1)  # swap p1 and p2 for turns
                ans = (ans[0] + w2, ans[1] + w1)
    result_db[(pos1, score1, pos2, score2)] = ans
    return ans
