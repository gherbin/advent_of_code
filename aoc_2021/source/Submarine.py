from aoc_utils.source import *


class Submarine:
    def __init__(self):
        self.horiz_pos = 0
        self.depth_pos = 0
        self.aim = 0
        self.gamma_rate = None
        self.eps_rate = None
        self.power_consumption = None

        self.ox_gen = None
        self.co2_scrub = None

    def forward(self, dist):
        self.horiz_pos += dist
        self.depth_pos += self.aim * dist

    def up(self, dist):
        # self.depth_pos -= dist
        self.aim -= dist

    def down(self, dist):
        # self.depth_pos += dist
        self.aim += dist

    def compute_rates(self, lines):
        diagnostic_report = DiagnosticReport(lines)

        self.gamma_rate = diagnostic_report.compute_ratings("gamma")
        self.eps_rate = diagnostic_report.compute_ratings("eps")
        self.power_consumption = self.gamma_rate * self.eps_rate

        self.ox_gen = diagnostic_report.compute_ratings("ox_gen")
        self.co2_scrub = diagnostic_report.compute_ratings("co2_scrub")

    def __repr__(self):
        return "Submarine (H, D) = ({},{}); product = {};" \
               " gamma = {}, eps = {}; power_cons = {};\n" \
               "oxygen generator ={} ; co2 scrub = {}; product = {}".format(self.horiz_pos,
                                                                            self.depth_pos,
                                                                            self.horiz_pos * self.depth_pos,
                                                                            self.gamma_rate,
                                                                            self.eps_rate,
                                                                            self.power_consumption,
                                                                            self.ox_gen,
                                                                            self.co2_scrub,
                                                                            self.ox_gen * self.co2_scrub)


class DiagnosticReport:
    def __init__(self, lines):
        self.dr = np.array([[int(c) for c in l.rstrip("\n")] for l in lines])

    def compute_ratings(self, rating):
        """

        :param rating:
        :return: the computed rating
        todo: redundancy between gamma and eps computations could be removed by computing both simultaneously
        """
        dr = copy.deepcopy(self.dr)
        if rating == "ox_gen":
            return self.__apply_bit_criteria(dr, 0, "most")
        elif rating == "co2_scrub":
            return self.__apply_bit_criteria(dr, 0, "least")
        elif rating == "gamma":
            max_counts = dr.shape[0]
            sums = np.sum(dr, axis=0)
            most_common_bits = [1 if c > max_counts // 2 else 0 for c in sums]
            return bin2int(most_common_bits)
        elif rating == "eps":
            max_counts = dr.shape[0]
            sums = np.sum(dr, axis=0)
            most_common_bits = [1 if c > max_counts // 2 else 0 for c in sums]
            least_common_bits = [0 if c == 1 else 1 for c in most_common_bits]
            return bin2int(least_common_bits)

    def __apply_bit_criteria(self, dr, bit_considered, criteria):
        """

        :param dr: current diagnostic report
        :param bit_considered: index of the bit to consider in current step
        :param criteria: "most" or "least"
        :return:
        """
        if len(dr) == 1:
            return bin2int(np.array(dr)[0])
        else:
            max_counts = dr.shape[0]
            cb = 0
            # sums = np.sum(dr, axis=0)
            if criteria == "most":
                if np.sum(dr[:, bit_considered]) >= max_counts / 2:
                    cb = 1
            elif criteria == "least":
                if np.sum(dr[:, bit_considered]) < max_counts / 2:
                    cb = 1
            dr_new = dr[np.where(dr[:, bit_considered] == cb)]
            return self.__apply_bit_criteria(dr_new, bit_considered + 1, criteria)


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
