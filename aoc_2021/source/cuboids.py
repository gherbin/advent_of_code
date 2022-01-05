import itertools

from aoc_utils.source import *


def convert_to_coords(line):
    list_of_strs = line.strip().split("=")
    status = "on" in list_of_strs[0]
    xs = tuple([int(a) for a in list_of_strs[1].split(",")[0].split("..")])
    ys = tuple([int(a) for a in list_of_strs[2].split(",")[0].split("..")])
    zs = tuple([int(a) for a in list_of_strs[3].split("..")])
    return status, xs, ys, zs


def organize_coords(c):
    return c[1][0], c[1][1], c[2][0], c[2][1], c[3][0], c[3][1]


def not_intersect(c0, c1):
    x0i, x0f, y0i, y0f, z0i, z0f = c0[1][0], c0[1][1], c0[2][0], c0[2][1], c0[3][0], c0[3][1]
    x1i, x1f, y1i, y1f, z1i, z1f =  c1[1][0], c1[1][1], c1[2][0], c1[2][1], c1[3][0], c1[3][1]
    return (x1i > x0f) or (x1f < x0i) or \
           (y1i > y0f) or (y1f < y0i) or \
           (z1i > z0f) or (z1f < z0i)


def is_included(c0, c1):
    x0i, x0f, y0i, y0f, z0i, z0f = c0[1][0], c0[1][1], c0[2][0], c0[2][1], c0[3][0], c0[3][1]
    x1i, x1f, y1i, y1f, z1i, z1f =  c1[1][0], c1[1][1], c1[2][0], c1[2][1], c1[3][0], c1[3][1]
    return (x0i >= x1i) and (x0f <= x1f) and \
           (y0i >= y1i) and (y0f <= y1f) and \
           (z0i >= z1i) and (z0f <= z1f)


def c0_eq_c1(c0, c1, axis):
    #  ---c0---
    #  ---c1---
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(c0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(c1)
    if axis == "x":
        return (x0f == x1f) and (x0i == x1i)
    elif axis == "y":
        return (y0f == y1f) and (y0i == y1i)
    elif axis == "z":
        return (z0f == z1f) and (z0i == z1i)


def c0_in_c1(c0, c1, axis):
    #      ---c0---
    #  -------c1-------
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(c0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(c1)
    if axis == "x":
        return (x0f < x1f) and (x0i > x1i)
    elif axis == "y":
        return (y0f < y1f) and (y0i > y1i)
    elif axis == "z":
        return (z0f < z1f) and (z0i > z1i)


def c0_out_c1(c0, c1, axis):
    # --------c0---------
    #      ---c1---
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(c0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(c1)
    if axis == "x":
        return (x1f < x0f) and (x1i > x0i)
    elif axis == "y":
        return (y1f < y0f) and (y1i > y0i)
    elif axis == "z":
        return (z1f < z0f) and (z1i > z0i)


def c0_greater_c1(c0, c1, axis):
    """
    #      -----c0------
    #   -------c1-----
    """
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(c0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(c1)
    if axis == "x":
        return (x1i < x0i) and (x1f < x0f)
    elif axis == "y":
        return (y1i < y0i) and (y1f < y0f)
    elif axis == "z":
        return (z1i < z0i) and (z1f < z0f)


def c0_smaller_c1(c0, c1, axis):
    """
    # -----c0------
    #   -------c1-----
    """
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(c0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(c1)
    if axis == "x":
        return (x0i < x1i) and (x0f < x1f)
    elif axis == "y":
        return (y0i < y1i) and (y0f < y1f)
    elif axis == "z":
        return (z0i < z1i) and (z0f < z1f)


def c0_equal_greater_c1(c0, c1, axis):
    """
    # -------c0---------
    # -------c1-----
    """
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(c0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(c1)
    if axis == "x":
        return (x0i == x1i) and (x0f > x1f)
    elif axis == "y":
        return (y0i == y1i) and (y0f > y1f)
    elif axis == "z":
        return (z0i == z1i) and (z0f > z1f)


def c0_equal_smaller_c1(c0, c1, axis):
    """
    # -------c0--
    # -------c1-----
    """
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(c0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(c1)
    if axis == "x":
        return (x0i == x1i) and (x0f < x1f)
    elif axis == "y":
        return (y0i == y1i) and (y0f < y1f)
    elif axis == "z":
        return (z0i == z1i) and (z0f < z1f)


def c0_greater_equal_c1(c0, c1, axis):
    """
    #     -----c0-----
    # ---------c1-----
    """
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(c0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(c1)
    if axis == "x":
        return (x0i > x1i) and (x0f == x1f)
    elif axis == "y":
        return (y0i > y1i) and (y0f == y1f)
    elif axis == "z":
        return (z0i > z1i) and (z0f == z1f)


def c0_smaller_equal_c1(c0, c1, axis):
    """
    # -------c0-----
    #      --c1-----
    """
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(c0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(c1)
    if axis == "x":
        return (x0i < x1i) and (x0f == x1f)
    elif axis == "y":
        return (y0i < y1i) and (y0f == y1f)
    elif axis == "z":
        return (z0i < z1i) and (z0f == z1f)


def merge(v0, v1, axis):
    """

    :param v0: a volume [True, [xi,xf],[yi,yf], [zi,zf]]
    :param v1: a volume [True, [xi,xf],[yi,yf], [zi,zf]]
    :param axis: {x, y, z, all} => axis to be merged
    :return: the merged volume
    """
    if axis == "all":
        return v0

    if axis == 'x':
        index_ = 1
    elif axis == 'y':
        index_ = 2
    elif axis == 'z':
        index_ = 3
    else:
        raise ValueError("Axis is not known{}".format(axis))
    to_merge = [v0[index_][0], v0[index_][1], v1[index_][0], v1[index_][1]]
    # print('to_merge = {}'.format(to_merge))
    to_merge.sort()
    new_coords = (to_merge[0], to_merge[-1])
    # print(f'new_coords = {new_coords}')
    if axis == 'x':
        return v0[0], new_coords, v0[2], v0[3]
    if axis == 'y':
        return v0[0], v0[1], new_coords, v0[3]
    if axis == 'z':
        return v0[0], v0[1], v0[2], new_coords


def can_be_merged(v0, v1):
    if v0[0] != v1[0]:
        return False
    x0i, x0f, y0i, y0f, z0i, z0f = organize_coords(v0)
    x1i, x1f, y1i, y1f, z1i, z1f = organize_coords(v1)
    if x0i == x1i and x0f == x1f:
        if y0i == y1i and y0f == y1f:
            if z0i == z0f and z1i == z1f:
                return "all"
            else:
                # CHECK that coordinates are contiguous !
                if (z0f == (z1i - 1)) or (z1f == (z0i - 1)) or (z1f == z0i) or (z0f == z1i):
                    return 'z'
        elif z0i == z0f and z1i == z1f:
            # CHECK that coordinates are contiguous !
            if (y0f == (y1i - 1)) or (y1f == (y0i - 1)) or (y0f == y1i) or (y1f == y0i):
                return 'y'
    elif y0i == y1i and y0f == y1f and z0i == z0f and z1i == z1f:
        # CHECK that coordinates are contiguous !
        if (x0f == (x1i - 1)) or (x1f == (x0i-1)) or (x0f == x1i) or (x1f == x0i):
            return "x"
    return False


def can_be_merged_list(vs):
    # _vs = list(vs)
    # ls = []
    # for i in range(len(_vs)):
    #     for j in range(i + 1, len(_vs)):
    #         if can_be_merged(_vs[i], _vs[j]):
    #             ls.append((_vs[i], _vs[j]))
    res = {pair for pair in itertools.combinations(vs, 2) if can_be_merged(*pair)}
    # print(res)
    return res


def merge_volumes_rec(volumes):
    # print("[MERGE] volumes start: {}".format(volumes))
    all_to_be_merged = can_be_merged_list(volumes)

    if len(all_to_be_merged) == 0:
        if can_be_merged_list(volumes):
            raise RuntimeError("That should not be !")
        return volumes
    cleaner = copy.deepcopy(volumes)
    assert cleaner == volumes
    for pair in all_to_be_merged:
        # print("to be merged")
        # print(pair)
        # print(volumes)
        if (pair[0] == pair[1] and cleaner.count(pair[0]) >= 2) or \
                (pair[0] != pair[1] and (pair[0] in cleaner) and (pair[1] in cleaner)):
            t_ = len(cleaner)
            cleaner.remove(pair[0])
            assert len(cleaner) == (t_ - 1)
            cleaner.remove(pair[1])
            assert len(cleaner) == (t_ - 2)

            ax = can_be_merged(pair[0], pair[1])
            merged_v = merge(pair[0], pair[1], ax)
            cleaner.add(merged_v)
            # assert len(cleaner) == (t_ - 1), f"new length = {len(cleaner)}, was {t_}"
            # print("[MERGE] volumes end if: {}".format(cleaner))

    # assert cleaner != volumes, "Merge hasn't done anything !!"
    # print("[MERGE] volumes end: {}".format(cleaner))
    return merge_volumes_rec(cleaner)


def get_bounds(c0, c1):
    assert not (not_intersect(c0, c1)), "c0 and c1 are supposed to intersect !"
    bounds = {}

    axes = ['x', 'y', 'z']
    for ax in range(len(axes)):
        if c0_eq_c1(c0, c1, axes[ax]):
            bounds[axes[ax]] = [c0[ax + 1][0], c0[ax + 1][1]]
        elif c0_out_c1(c0, c1, axes[ax]):
            bounds[axes[ax]] = [c0[ax + 1][0], c1[ax + 1][0] - 1,
                                c1[ax + 1][0], c1[ax + 1][1],
                                c1[ax + 1][1] + 1, c0[ax + 1][1]]
        elif c0_in_c1(c0, c1, axes[ax]):
            bounds[axes[ax]] = [c1[ax + 1][0], c0[ax + 1][0] - 1,
                                c0[ax + 1][0], c0[ax + 1][1],
                                c0[ax + 1][1] + 1, c1[ax + 1][1]]
        elif c0_greater_c1(c0, c1, axes[ax]):
            bounds[axes[ax]] = [c1[ax + 1][0], c0[ax + 1][0] - 1,
                                c0[ax + 1][0], c1[ax + 1][1],
                                c1[ax + 1][1] + 1, c0[ax + 1][1]]
        elif c0_smaller_c1(c0, c1, axes[ax]):
            bounds[axes[ax]] = [c0[ax + 1][0], c1[ax + 1][0] - 1,
                                c1[ax + 1][0], c0[ax + 1][1],
                                c0[ax + 1][1] + 1, c1[ax + 1][1]]
        elif c0_smaller_equal_c1(c0, c1, axes[ax]):
            bounds[axes[ax]] = [c0[ax + 1][0], c1[ax + 1][0] - 1,
                                c1[ax + 1][0], c1[ax + 1][1]]
        elif c0_greater_equal_c1(c0, c1, axes[ax]):
            bounds[axes[ax]] = [c1[ax + 1][0], c0[ax + 1][0] - 1,
                                c0[ax + 1][0], c1[ax + 1][1]]
        elif c0_equal_smaller_c1(c0, c1, axes[ax]):
            bounds[axes[ax]] = [c0[ax + 1][0], c0[ax + 1][1],
                                c0[ax + 1][1] + 1, c1[ax + 1][1]]
        elif c0_equal_greater_c1(c0, c1, axes[ax]):
            bounds[axes[ax]] = [c1[ax + 1][0], c1[ax + 1][1],
                                c1[ax + 1][1] + 1, c0[ax + 1][1]]
        else:
            raise RuntimeError(f"configuration c0 c1 unknown: {c0} vs {c1}")

    return bounds['x'], bounds['y'], bounds['z']


def split(c_tosplit, c_ref):
    """
    :param c_tosplit:
    :param c_ref:
    :return: a list of cuboids that don't intersect, covering the whole volume
    """

    if not_intersect(c_tosplit, c_ref):
        return set([c_ref])

    elif is_included(c_ref, c_tosplit):
        return set([])

    else:
        # assert(not(not_intersect(c_tosplit, c_ref)))
        bounds_x, bounds_y, bounds_z = get_bounds(c_tosplit, c_ref)
        # build volume lists
        volumes = set()  # could be changed by set
        for ix in range(0, len(bounds_x), 2):
            for iy in range(0, len(bounds_y), 2):
                for iz in range(0, len(bounds_z), 2):
                    v = ["tbd",
                         (bounds_x[ix], bounds_x[ix + 1]),
                         (bounds_y[iy], bounds_y[iy + 1]),
                         (bounds_z[iz], bounds_z[iz + 1])]
                    assert bounds_x[ix] <= bounds_x[ix + 1], "X not increasing"
                    assert bounds_y[iy] <= bounds_y[iy + 1], "Y not increasing"
                    assert bounds_z[iz] <= bounds_z[iz + 1], "Z not increasing"
                    assert c_ref[0], "Should be True bordel!"

                    # print("[split debug candidate]0 v = {}".format(v))
                    if (is_included(v, c_tosplit)) or (is_included(c_tosplit, v)):
                        v[0] = False # added later... c_tosplit[0]
                    elif (is_included(v, c_ref)) or (is_included(c_ref, v)):
                        v[0] = c_ref[0]
                    else:
                        v[0] = False

                    if v[0] and is_included(v, c_ref):
                        volumes.add(tuple(v))

        return volumes


# def remove_included(volumes: set):
#     for c1, c2 in itertools.combinations(volumes, 2):
#         if is_included(c1, c2):
#             volumes.remove(c1)
#         elif is_included(c2, c1):
#             volumes.remove(c2)
#     return volumes


# def is_ok(volumes):
#     res = True
#     if len(volumes) < 2:
#         res = True
#     else:
#         for c1, c2 in itertools.combinations(volumes, 2):
#             # if not (not_intersect(c1, c2)):
#             #     print("[is ok] Interesecting: {} && {}\t\t Are included ? {}".format(c1, c2,
#             #                                                                          is_included(c1, c2) + is_included(
#             #                                                                              c2, c1)))
#
#             res = (res and not_intersect(c1, c2))
#     return res


def compute_volume(volumes):
    vol = 0
    for v in volumes:
        tmp = (abs(v[1][1] - v[1][0]) + 1) * (abs(v[2][1] - v[2][0]) + 1) * (abs(v[3][1] - v[3][0]) + 1)
        vol += tmp
    return vol


def test_merge():
    with open("day22_dummy.txt", 'r') as f:
        lines = f.readlines()
    coords = [convert_to_coords(line) for line in lines]
    vs = split(coords[1], coords[0])
    print(f"Volume to me merged =\n{vs}")
    print(f"vs {len(vs)}:\n{vs}")

    cvs = merge_volumes_rec(vs)
    print(f"cvs {len(cvs)}:\n{cvs}")


def test_c0_c1():
    with open("day22_dummy.txt", 'r') as f:
        lines = f.readlines()
    coords = [convert_to_coords(line) for line in lines]
    counts = np.zeros(shape=(len(list(itertools.combinations(coords, 2))), 3))
    axes = ["x", "y", "z"]
    i = 0
    for a, b in itertools.combinations(coords, 2):
        print(a, b)
        for ax in range(len(axes)):
            print("{},{},{},{},{},{},{},{},{}".format(c0_eq_c1(a, b, axes[ax]),
                                                      c0_out_c1(a, b, axes[ax]),
                                                      c0_in_c1(a, b, axes[ax]),
                                                      c0_greater_c1(a, b, axes[ax]),
                                                      c0_smaller_c1(a, b, axes[ax]),
                                                      c0_smaller_equal_c1(a, b, axes[ax]),
                                                      c0_greater_equal_c1(a, b, axes[ax]),
                                                      c0_equal_smaller_c1(a, b, axes[ax]),
                                                      c0_equal_greater_c1(a, b, axes[ax])))
            res = c0_eq_c1(a, b, axes[ax]) + \
                  c0_out_c1(a, b, axes[ax]) + \
                  c0_in_c1(a, b, axes[ax]) + \
                  c0_greater_c1(a, b, axes[ax]) + \
                  c0_smaller_c1(a, b, axes[ax]) + \
                  c0_smaller_equal_c1(a, b, axes[ax]) + \
                  c0_greater_equal_c1(a, b, axes[ax]) + \
                  c0_equal_smaller_c1(a, b, axes[ax]) + \
                  c0_equal_greater_c1(a, b, axes[ax])
            counts[i, ax] = res
        i += 1
    np.savetxt("test.out", counts, fmt="%i")


def test_bounds():
    with open("day22.txt", 'r') as f:
        lines = f.readlines()
    coords = [convert_to_coords(line) for line in lines]

    bounds_x, bounds_y, bounds_z = get_bounds(coords[2], coords[1])
    print(bounds_x)
    print(bounds_y)
    print(bounds_z)
#
# def make_ok(volumes):
#     # counter_ = 0
#     while not is_ok(volumes):
#         # print("counter = {}".format(counter_))
#         # counter_ += 1
#         for c0, c1 in itertools.combinations(volumes.copy(), 2):
#             if not (not_intersect(c0, c1)) and (c0 in volumes) and (c1 in volumes):
#                 tmp_ = split(c0, c1)
#                 volumes.remove(c0)
#                 volumes.remove(c1)
#                 volumes = volumes.union(tmp_)
#     return volumes

def run():
    with open("aoc_2021/inputs/day22.txt", 'r') as f:
        lines = f.readlines()
    coords = [convert_to_coords(line) for line in lines]
    # print("[test run] coords:{}".format(coords))
    # print("[test run] coords[0]:\n{}".format(coords[0]))
    pixels_on = set()
    pixels_on.add(coords[0])
    # print("[test run] pixels on:\n{}".format(pixels_on))
    # with open("run.out", 'w+') as f:
    #     f.write(str(pixels_on))
    #     f.write("\n")

    for i in range(1, len(coords)):
        print(len(pixels_on))
        # with open("run.out", 'a+') as f:
        #     f.write(str(pixels_on))
        #     f.write("\n")
        new_volume = coords[i]
        new_px_on = set()
        for v in pixels_on:
            # pixels_on.remove(v)
            # print("[debug] bounds = {}".format(get_bounds(new_volume, v)))
            # print("[test_run]: new_volume {}".format(new_volume))
            # print("[test_run]: v {}".format(v))
            tmp_ = split(new_volume, v)
            # tmp_ = merge_volumes_rec(tmp_)
            # print("[test_run]: tmp_ class type = {}".format(tmp_.__class__.__name__))
            # print("[test_run]: tmp_ set : {}".format(tmp_))
            # tmp_ = make_ok(tmp_)

            # assert is_ok(tmp_)
            new_px_on = new_px_on.union(tmp_)
            # new_px_on = remove_included(new_px_on)
            # new_px_on = merge_volumes_rec(new_px_on)
            # print("[test_run]: new_px_on set : {}".format(new_px_on))
            # new_px_on = make_ok(new_px_on)
            # assert is_ok(new_px_on)
        if new_volume[0]:
            new_px_on.add(tuple(new_volume))

        # pixels_on = pixels_on.union(new_px_on)
        # pixels_on = remove_included(pixels_on)
        pixels_on = new_px_on.copy()
        # pixels_on = merge_volumes_rec(pixels_on)
        # pixels_on = make_ok(pixels_on)
        # print("length pixels on = {}".format(len(pixels_on)))
        # assert is_ok(pixels_on), f"\npixels_on:{pixels_on}"
    # print(f"\npixels_on:{pixels_on}")
    vol = compute_volume(pixels_on)
    print(f"vol = {vol}")
    return vol

if __name__ == '__main__':
    # test_merge()
    # test_c0_c1()
    run()
