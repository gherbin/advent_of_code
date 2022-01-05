from aoc_utils.source import *

def get_roi(inp, px, py):
    # expansion = expanding pixels around original image
    roi = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if ((px+i) < 0) or ((py+j) < 0):
                c_ = 's'
            elif ((px+i) >= len(inp)) or ((py+j) >= len(inp[0])):
                c_ = 'L'
            else:
                c_ = inp[px+i][py+j]
            roi += [c_]
    return roi


def expand_image(input_image, expansion=1):
    if expansion < 0:
        raise ValueError("Expansion shall be larger than 0, is {}".format(expansion))
    out = []
    xs = len(input_image[0])
    # ys = len(input_image)
    nxs = 2*expansion + xs
    # nys = 4 + ys
    for _ in range(expansion):
        out += ['.'*nxs]
    for j in range(len(input_image)):
        l = ['.']*expansion + input_image[j] + ['.']*expansion
        out += [l]
    for _ in range(expansion):
        out += ['.'*nxs]
    return out

def crop_image(input_image, recession = 1):
    out = []
    for j in range(recession, len(input_image)-recession):
        l = input_image[j][recession: len(input_image[0])-recession]
        out += [l]
    return out


def c_2int(l):
    b = [1 if c == "#" else 0 for c in l]
    return bin2int(b)


def iae(inp, ref_chars, expansion = 0):
    assert expansion == 0, "Test: expansion should be 0"
    #scroll an exmp
    output_image = []
    for i in range(len(inp)):#+2*expansion):
        output_image+= [[]]
        for j in range(len(inp[0])):#+2*expansion):
            output_image[-1] += '-' #expanded_image[j][i]
    # through all lists in list
    for j in range(0, len(output_image)):
        # through chars in string
        for i in range(0, len(output_image[0])):
            char_ = get_roi(inp, j-expansion, i-expansion, None)
            dec_ = c_2int(char_)
            new_char_ = ref_chars[dec_]
            output_image[j][i] = new_char_#
            #

    return output_image


def pretty_print(list_of_lists):
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    for line in list_of_lists:
        print("".join(line))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
