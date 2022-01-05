from aoc_utils import *


class ALU:
    def __init__(self):
        self.dims = {'x': 0, 'y': 0, 'z': 0, 'w': 0}
        self.inp_val = 0

    def __repr__(self):
        return f"w: {self.dims['w']}, x: {self.dims['x']}, " \
               f"y: {self.dims['y']}, z: {self.dims['z']}"

    def execute_program(self, instructions):
        for instruction in instructions:
            self.execute_one(instruction)
        print(self.dims)

    def execute_one(self, instruction):
        instruction = instruction.strip().split()
        cmd = instruction[0]
        a = instruction[1]
        if len(instruction) > 2:
            b = instruction[2]
            if b in {'x', 'y', 'z', 'w'}:
                b = self.dims[b]
            else:
                b = int(b)
        if cmd == "inp":
            assert a in {'x', 'y', 'z', 'w'}
            self.dims[a] = self.inp_val
            # self.inp_val = a
        elif cmd == "add":
            self.dims[a] = self.dims[a] + b
        elif cmd == "mul":
            self.dims[a] = self.dims[a] * b
        elif cmd == "div":
            assert b != 0, "div: b != 0"
            self.dims[a] = self.dims[a] // b
        elif cmd == "mod":
            assert self.dims[a] >= 0, "mod: a >= 0"
            assert b > 0, "mod: b > 0"
            self.dims[a] = self.dims[a] % b
        elif cmd == "eql":
            self.dims[a] = int(self.dims[a] == b)
        else:
            raise ValueError(f"instruction not understood : {instruction}")


def do():
    "13579246899999"
    digits = [1, 3, 5, 7, 9, 2, 4, 6, 8, 9, 9, 9, 9, 9]
    digits = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    digits = [9, 9, 6, 9, 1, 8, 9, 1, 9, 7, 9, 9, 3, 8]
    xadd = [13, 11, 14, -5, 14, 10, 12, -14, -8, 13, 0, -5, -9, -1]
    zdiv = [1, 1, 1, 26, 1, 1, 1, 26, 26, 1, 26, 26, 26, 26]
    yadd = [0, 3, 8, 5, 13, 9, 6, 1, 1, 2, 7, 5, 8, 15]
    assert len(digits) == 14
    assert len(xadd) == 14
    assert len(zdiv) == 14
    assert len(yadd) == 14

    zprev = 0
    z = [0]*15
    # z[0] = 0
    print(f"z = {z}")
    trick_z = 1
    stack_ = 0
    for i in range(len(digits)):
        print(f"----------{i}---------")
        zprev = z[i - 1 + trick_z]
        print("zprev = {}".format(zprev))

        if zdiv[i] == 26:
            print("zprev_before (26) = {}".format(zprev))
            zprev = z[i - 1 + trick_z] // 26
            print("zprev_after  (26) = {}".format(zprev))

        if (((z[i-1 + trick_z] % 26) + xadd[i]) == digits[i]) == 0:
            zprev = 26 * zprev + digits[i] + yadd[i]

        z[i+trick_z] = zprev
    print(z)

if __name__ == '__main__':
    do()
