from aoc_utils.source import *

R0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R90 = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
R180 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R270 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])


def rx(ang):
    ang = np.radians(ang)
    return np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])


def ry(ang):
    ang = np.radians(ang)
    return np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])


def rz(ang):
    ang = np.radians(ang)
    return np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])


Rxy90 = np.matmul(rx(90), ry(90)).astype(int)
Rxy180 = np.matmul(rx(180), ry(180)).astype(int)
Rxy270 = np.matmul(rx(270), ry(270)).astype(int)
Rxz90 = np.matmul(rx(90), rz(90)).astype(int)
Rxz180 = np.matmul(rx(180), rz(180)).astype(int)
Rxz270 = np.matmul(rx(270), rz(270)).astype(int)
Ryz90 = np.matmul(ry(90), rz(90)).astype(int)
Ryz180 = np.matmul(ry(180), rz(180)).astype(int)
Ryz270 = np.matmul(ry(270), rz(270)).astype(int)

Rx90 = rx(90).astype(int)
Ry90 = ry(90).astype(int)
Rz90 = rz(90).astype(int)

Rx180 = rx(180).astype(int)
Ry180 = ry(180).astype(int)
Rz180 = rz(180).astype(int)

Rx270 = rx(270).astype(int)
Ry270 = ry(270).astype(int)
Rz270 = rz(270).astype(int)

S0 = [1, 1, 1]
S1 = [-1, 1, 1]
S2 = [1, -1, 1]
S3 = [-1, -1, 1]
S4 = [1, 1, -1]
S5 = [-1, 1, -1]
S6 = [1, -1, -1]
S7 = [-1, -1, -1]


class Scanner:

    def __init__(self, id):
        self.id = id
        self.origin = None
        self.index_rotation = None
        self.rotation = None
        self.signs = None
        self.index_signs = None
        self.reference_scanner = None
        if self.id == '0':
            self.origin = np.array([0, 0, 0])
            self.index_rotation = 0
            self.index_signs = 0
            self.reference_scanner = '0'
            self.rotation = np.eye(3)
            self.signs = np.ones((3,))
        self.coordinates = []
        self.coordinates_adjusted = []
        self.variant = []

    def init(self):
        self.coordinates = np.array(self.coordinates)

    def __repr__(self):
        return f"< scanner {self.id} :\n" \
               f"\treference    :{self.reference_scanner}\n" \
               f"\torigin       :{self.origin}\n" \
               f"\tidx orient.  :{self.index_rotation}\n" \
               f"\tidx signs    :{self.index_signs}\n" \
               f"\tidx orient.  :\n{self.rotation}\n" \
               f"\tidx signs    :{self.signs} >"

    def generate_variants(self):
        if not isinstance(self.coordinates, np.ndarray):
            self.init()
        # 24 possible variant. the first one is the one given -> generate the 23 others.
        # print(self.coordinates.shape)
        for r in [R0, R90, R180, R270]:
            for s in [S0, S1, S2, S3, S4, S5, S6, S7]:
                var = copy.deepcopy(self.coordinates)
                var = np.matmul(var, r)
                var = var * s
                self.variant.append(var)
                with open("var.log", 'a+') as f:
                    f.write("r; s = {} ; {}\n".format(r, s))
                    np.savetxt(f, var, fmt='%.0f')
        self.variant = np.array(self.variant)
        # print(len(self.variant))

    # def At_least_twelve_distances(self):

    def matches(self, ref_scanner):
        if not isinstance(self.coordinates, np.ndarray):
            self.init()
        with open("test.log", 'w+') as f:
            f.write("\n")
        rotations = [R0, R90, R270,
                     Rx90, Ry90, Rz90,
                     Rx180, Ry180, Rz180,
                     Rx270, Ry270, Rz270,
                     Rxy90, Rxy180, Rxy270,
                     Rxz90, Rxz180, Rxz270,
                     Ryz90, Ryz180, Ryz270]
        signs = [S0, S1, S2, S3, S4, S5, S6, S7]
        for index_r in range(len(rotations)):
            r = rotations[index_r]
            for index_s in range(len(signs)):
                s = signs[index_s]

                # " modify coordinates to find matching "
                var = copy.deepcopy(self.coordinates)
                # print("var\n={}".format(var))

                var = np.matmul(var, r)
                var = var * s
                ref = copy.deepcopy(ref_scanner.coordinates)

                refr = np.reshape(ref, newshape=(ref.shape[0], 1, ref.shape[1]))
                refr = np.repeat(refr, var.shape[0], axis=1)
                res = var - refr
                resr = np.reshape(res, newshape=(res.shape[0] * res.shape[1], res.shape[2]))
                val, indices, counts = np.unique(resr, axis=0, return_counts=True, return_index=True)
                if np.max(counts) >= 12:
                    print("counts max = {}".format(np.max(counts)))
                    print("vector max = {}".format(val[np.argmax(counts)]))
                    print("orientation = {}, sign = {}".format(index_r, index_s))
                    # print("new coords of {} with respect to {}:\n{}".format(self.id,
                    #                                                         ref_scanner.id,
                    #                                                         var - val[np.argmax(counts)]))
                    # self.coordinates_adjusted = var - val[np.argmax(counts)]
                    self.origin = val[np.argmax(counts)]
                    self.index_rotation = index_r
                    self.rotation = r
                    self.index_signs = index_s
                    self.signs = s
                    self.reference_scanner = ref_scanner.id
                    print("---------------------")
                    return np.max(counts)
        return False




