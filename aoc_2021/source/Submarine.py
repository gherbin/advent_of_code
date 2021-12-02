class Submarine:
    def __init__(self):
        self.horiz_pos = 0
        self.depth_pos = 0
        self.aim = 0

    def forward(self, dist):
        self.horiz_pos += dist
        self.depth_pos += self.aim*dist

    def up(self, dist):
        # self.depth_pos -= dist
        self.aim -= dist

    def down(self, dist):
        # self.depth_pos += dist
        self.aim += dist

    def __repr__(self):
        return "Submarine (H, D) = ({},{}); product = {}".format(self.horiz_pos,
                                                                 self.depth_pos,
                                                                 self.horiz_pos*self.depth_pos)


class SubmarineWrapper:
    def __init__(self, submarine):
        self.submarine = submarine

    def move(self, command_string):
        """
        wraps the details for the submarine moves.
        :param command_string: typ. forward INT, as a line of the puzzle input
        :return:
        """
        infos = command_string.split(" ")
        command = infos[0]
        dist = int(infos[1])
        if command == "forward":
            self.submarine.forward(dist)
        elif command == "up":
            self.submarine.up(dist)
        elif command == "down":
            self.submarine.down(dist)
        else:
            raise ValueError("command not understood: {}".format(command))