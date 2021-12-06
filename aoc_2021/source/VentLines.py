from aoc_utils.source import *


def build_vents_map(sources, targets):
    sources = np.array(sources)
    targets = np.array(targets)
    max_coord = np.max(np.array([np.max(sources), np.max(targets)]))+1
    maps = np.zeros(shape=(max_coord, max_coord))

    for i in np.arange(0, len(sources)):
        if vert_or_horiz(sources[i], targets[i]):
            # part 1
            coords = get_coords_impacted(sources[i], targets[i])
        else:
            # part 2 - diagonal lines
            coords = get_coords_impacted(sources[i], targets[i])
        maps[coords[:,1], coords[:,0]] += 1
    return maps


def vert_or_horiz(source, target):
    if (source[0] == target[0]) or (source[1] == target[1]):
        return True
    return False


def get_coords_impacted(source, target):
    # return a list of coordinates of the lines
    coords = None
    if source[0] == target[0]:
        #if source[1] < target[1]:
        tmp = np.linspace(source[1], target[1], np.abs(target[1]-source[1])+1,endpoint=True, dtype=np.int)
        coords = np.array(list(zip( [source[0]]*len(tmp), tmp )))
    elif source[1] == target[1]:
        tmp = np.linspace(source[0], target[0], np.abs(target[0]-source[0])+1,endpoint=True, dtype=np.int)
        coords = np.array(list(zip( tmp, [source[1]]*len(tmp))))
    else:
        tmp_x = np.linspace(source[0], target[0], np.abs(target[0]-source[0])+1,endpoint=True, dtype=np.int)
        tmp_y = np.linspace(source[1], target[1], np.abs(target[1]-source[1])+1,endpoint=True, dtype=np.int)
        coords = np.array(list(zip( tmp_x, tmp_y )))

    return coords