from aoc_utils.source import *


class Bingo:

    def __init__(self, boards):
        self.boards = boards  # np array structure containing all the boards as [N, c, c]
        self.boards_binary = np.zeros_like(self.boards)
        self.bingo = None
        self.bingo_board = None
        self.last_called = None
        self.runs = [0]*self.boards.shape[0]
        self.cur_run = 0

    def is_bingo(self):
        """
        returns the line or column where there is a bingo (all items marked), or False
        :param boards_binary:
        :return:
        """
        is_bingo = False
        for b in np.arange(0, self.boards.shape[0]):
            if self.runs[b] == 0:
                for ax in [0, 1]:  # vert and horiz
                    a = np.all(self.boards_binary[b, :, :], axis=ax)
                    if np.any(a):
                        is_bingo = True
                        self.bingo_board = b
                        self.runs[self.bingo_board] = self.cur_run
                        if ax == 0:
                            self.bingo = np.squeeze(self.boards[b, :, np.where(a)])
                        elif ax == 1:
                            self.bingo = np.squeeze(self.boards[b, np.where(a), :])
        return is_bingo

    def mark_cell(self, number):
        self.cur_run += 1
        self.last_called = number
        indices = np.where(self.boards == number)
        self.boards_binary[indices] = 1

    def compute_score(self, board=None):
        if board is None:
            board = self.bingo_board
        sum_unmarked = np.sum((1-self.boards_binary[board,:,:])*self.boards[board,:,:])
        score = sum_unmarked*self.last_called
        return score

