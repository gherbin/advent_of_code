import copy


class Cucumber:
    def __init__(self, lines):
        self.x_max = len(lines)
        self.y_max = len(lines[0])
        self.dce = {}
        self.dcs = {}
        self.emp = {}
        for i in range(len(lines)):
            for j in range(len(lines[0])):
                if lines[i][j] == ">":
                    self.dce[(i, j)] = True
                    self.dcs[(i, j)] = False
                    self.emp[(i, j)] = False
                elif lines[i][j] == "v":
                    self.dce[(i, j)] = False
                    self.dcs[(i, j)] = True
                    self.emp[(i, j)] = False
                elif lines[i][j] == ".":
                    self.dce[(i, j)] = False
                    self.dcs[(i, j)] = False
                    self.emp[(i, j)] = True
                else:
                    raise ValueError(f"Unexpected input found:{lines[i][j]}")

    def do_step(self):
        self._attempt_move("east")
        self._attempt_move("south")
        print(self)

    def _attempt_move(self, direction):
        if direction == "east":
            local_emp = copy.deepcopy(self.emp)
            local_dce = copy.deepcopy(self.dce)

            for i, j in self.dce.keys():
                if local_emp[(i, (j + 1) % self.y_max)] and local_dce[(i,j)]:
                    self.dce[(i, j)] = False
                    self.dce[(i, (j + 1) % self.y_max)] = True
                    self.emp[(i, j)] = True
                    self.emp[(i, (j + 1) % self.y_max)] = False

        elif direction == "south":
            local_emp = copy.deepcopy(self.emp)
            local_dcs = copy.deepcopy(self.dcs)
            for i, j in self.dcs.keys():
                if local_emp[((i + 1) % self.x_max, j)] and local_dcs[(i,j)]:
                    self.dcs[(i, j)] = False
                    self.dcs[((i + 1) % self.x_max, j)] = True
                    self.emp[(i, j)] = True
                    self.emp[((i + 1) % self.x_max, j)] = False

        else:
            raise ValueError(f"Weird direction ! {direction}")


    def __repr__(self):
        L = []
        for i in range(self.x_max):
            L.append([])
            for j in range(self.y_max):
                if self.dce[(i, j)]:
                    L[-1].append('>')
                elif self.dcs[(i, j)]:
                    L[-1].append('v')
                elif self.emp[(i, j)]:
                    L[-1].append('.')

            L[-1].append("\n")
        return "".join([sublist for llist in L for sublist in llist])
