from aoc_utils.source import *
from enum import Enum

VERSION_SIZE = 3
TYPE_ID_SIZE = 3
cum_ver = 0


def bin_parser(i, level, packet):
    print("Debug: [{}] ({}), level {}".format(len(packet), i, level))

    ver = bin2int(packet[i:i + VERSION_SIZE])
    global cum_ver
    cum_ver += ver
    print("Cum Ver = {}".format(cum_ver))

    i = i + VERSION_SIZE
    type_id = bin2int(packet[i:i + TYPE_ID_SIZE])
    i = i + TYPE_ID_SIZE

    if type_id == 4:

        print("literal")
        bin_number = ""
        while True:
            next_number = packet[i:i + 5]
            i += 5
            if next_number[0] == "0":
                bin_number += next_number[1:]
                print("bin number = {}".format(bin_number))
                v = bin2int(bin_number)
                print("v = {}".format(bin_number))
                return v, i
            else:
                bin_number += next_number[1:]
    else:
        ltid = packet[i:i + 1]
        i += 1
        level += 1
        values = []
        if ltid == "0":
            tlb = bin2int(packet[i:i + 15])
            i += 15
            read_init_i = i
            while (i - read_init_i) != tlb:
                v, next_i = bin_parser(i, level, packet)
                values += [v]
                i = next_i
        elif ltid == "1":
            nop = bin2int(packet[i:i + 11])
            i += 11
            for n in range(nop):
                v, next_i = bin_parser(i, level, packet)
                values += [v]
                i = next_i

        if type_id == 0:
            return sum(values), i
        elif type_id == 1:
            ans = 1
            for v in values:
                ans *= v
            return ans, i
        elif type_id == 2:
            return min(values), i
        elif type_id == 3:
            return max(values), i
        elif type_id == 5:
            return (1 if values[0] > values[1] else 0), i
        elif type_id == 6:
            return (1 if values[0] < values[1] else 0), i
        elif type_id == 7:
            return (1 if values[0] == values[1] else 0), i
        else:
            assert False, type_id
        # return evaluate_operator(type_id, values), i

        # return 0, i+1
