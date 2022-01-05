from aoc_utils.source import *
import itertools
import time

spots = {(0, i) for i in range(11)} \
    .union({(1, 2), (1, 4), (1, 6), (1, 8),
            (2, 2), (2, 4), (2, 6), (2, 8),
            (3, 2), (3, 4), (3, 6), (3, 8),
            (4, 2), (4, 4), (4, 6), (4, 8)})

init_available_spots = {k: spots.copy().difference({k}).difference({(0, 2), (0, 4), (0, 6), (0, 8)}) for k in spots}
with open('debug.out', 'w+') as f:
    f.write("start\n")


class Amphipod:
    x = None

    def __init__(self, pos):
        self.asp = copy.deepcopy(init_available_spots)
        self.next_asp = copy.deepcopy(init_available_spots)
        self.curpos = pos
        self.hallway = False

    def update_asp(self):
        if self.curpos[0] == 0:
            # in the hallway -> cannot stop in the hallway anymore
            self.hallway = True
            for k in self.asp.keys():
                self.asp[k] = self.asp[k].difference({(0, i) for i in range(11)})
        elif (self.curpos[0] in [1, 2, 3, 4]) and self.hallway:
            # in a room and already stopped in the hallway -> not in another room nor in the hallway
            for k in self.asp.keys():
                self.asp[k] = self.asp[k].difference(
                    {(0, i) for i in range(11)}.
                        union({(1, j) for j in {2, 4, 6, 8} if j != self.curpos[1]}).
                        union({(2, j) for j in {2, 4, 6, 8} if j != self.curpos[1]}).
                        union({(3, j) for j in {2, 4, 6, 8} if j != self.curpos[1]}).
                        union({(4, j) for j in {2, 4, 6, 8} if j != self.curpos[1]}))
        elif (self.curpos[0] in [1, 2, 3, 4]) and not (self.hallway):
            # in a room, and not passed by the hallway yet -> cannot stop in front of a room
            for k in self.asp.keys():
                self.asp[k] = self.asp[k].difference(
                    {(0, self.curpos[1])}
                )
        else:
            raise RuntimeError(f"Where am I ? curpos: {self.curpos}, hallway: {self.hallway},\n{self}")

    def _can_move_to(self, spot):
        # print(f"self = {self}{self.curpos} " + f"self class x = {self.__class__.x}")
        # print(f"spot = {spot}")

        if self.curpos[0] == 0:
            # si on est en (0, X)
           if (spot[0] > 0) and (spot[1] == self.__class__.x):
                # sinon, si on veut aller en (Y, Z) où Z ne correspond pas à la classe => False !
                return True
        elif self.curpos[0] > 0:
            # si on est en (X, ?) ou X > 0, alors...
            if spot[0] == 0:
                # on peut aller en (0, Z)
                return True
            elif (spot[0] > 0) and (spot[1] != self.curpos[1]) and (spot[1] == self.__class__.x):
                # on peut aller en (?, Z) si Z est différent de la position courante
                return True
        return False

    def update_next_asp(self, locations):
        if self.is_fixed(locations):
            for k in self.asp.keys():
                self.next_asp[k] = set({})
        else:
            for k in self.asp.keys():
                self.next_asp[k] = {sp for sp in self.asp[k]
                                    if ((locations.locations[sp] is None) and
                                        (locations.exist_path(k, sp)) and
                                        (self._can_move_to(sp))
                                        )}
                for y_ in [2, 4, 6, 8]:  # go full down if possible (dont stop in (1, 4) if (4, 4) is accessible)
                    if (4, y_) in self.next_asp[k]:
                        self.next_asp[k] = self.next_asp[k].difference({(1, y_), (2, y_), (3, y_)})
                    elif (3, y_) in self.next_asp[k]:
                        self.next_asp[k] = self.next_asp[k].difference({(1, y_), (2, y_)})
                    elif (2, y_) in self.next_asp[k]:
                        self.next_asp[k] = self.next_asp[k].difference({(1, y_)})
                    elif ((self.curpos[0] == 4)):
                        assert not (self.is_fixed(locations)), "???"
                        # si on est en (2, ?) et on doit monter
                        self.next_asp[k] = self.next_asp[k].difference({(1, y_), (2, y_), (3, y_)})
            for k in self.asp.keys():
                for y_ in [2, 4, 6, 8]:
                    if ((3, y_) in self.next_asp[k]) and not (isinstance(locations.locations[(4, y_)], type(self))):
                        self.next_asp[k] = self.next_asp[k].difference({(3, y_)})
                    if ((2, y_) in self.next_asp[k]) and \
                            not (isinstance(locations.locations[(4, y_)], type(self)) and
                                 isinstance(locations.locations[(3, y_)], type(self))):
                        self.next_asp[k] = self.next_asp[k].difference({(2, y_)})
                    if ((1, y_) in self.next_asp[k]) and \
                            not (all([isinstance(locations.locations[(x_, y_)], type(self)) for x_ in [2, 3, 4]])):
                        self.next_asp[k] = self.next_asp[k].difference({(1, y_)})
            for k in {(i, j) for i in range(1, 5) for j in range(2, 9, 2)}:
                # go full up if possible, if required

                # [ (1, 2), (1, 4), (1, 6), (1, 8), (2, 2), (2, 4), (2, 6), (2, 8),
                #   (3, 2), (3, 4), (3, 6), (3, 8), (4, 2), (4, 4), (4, 6), (4, 8)]
                # if any([(0, x_) in self.next_asp[k] for x_ in range(11)]):
                if self.curpos[1] != self.__class__.x:
                    self.next_asp[k] = self.next_asp[k]. \
                        difference({(l_, y_) for l_ in [1, 2, 3] for y_ in [2, 4, 6, 8]})

    def is_blocked(self):
        print("is_blocked_pod")

    def is_fixed(self, locations):
        print("is_fixed")


class A(Amphipod):
    energy = 1
    x = 2

    def __init__(self, pos):
        super().__init__(pos)
        self.energy = 1

    def __repr__(self):
        return 'A'

    def is_blocked(self):
        if self.hallway and (self.curpos[0] > 0) and (self.curpos[1] != 2):
            return True
        return False

    def is_fixed(self, locations):
        """

        :param locations:
        :return: True if at the bottom, or not at the bottom but all belows are of the correct type
        """
        # ...
        # #.#
        # #.#
        # #A#
        # #A#
        if self.curpos[1] == A.x:
            if all([isinstance(locations.locations[(pos_, 2)], A) for pos_ in range(self.curpos[0] + 1, 4 + 1)]):
                return True
            else:
                # print(f"{self} in {self.curpos} -> "
                #       f"is_fixed is False [{(4,A.x)} is : {locations.locations[(4, A.x)]}]")
                return False
        else:
            return False


class B(Amphipod):
    energy = 10
    x = 4

    def __init__(self, pos):
        super().__init__(pos)
        self.energy = 10

    def __repr__(self):
        return 'B'

    def is_blocked(self):
        if self.hallway and (self.curpos[0] > 0) and (self.curpos[1] != 4):
            return True
        return False

    def is_fixed(self, locations):
        if self.curpos[1] == B.x:
            if all([isinstance(locations.locations[(pos_, B.x)], B) for pos_ in range(self.curpos[0] + 1, 4 + 1)]):
                return True
            else:

                return False
        else:
            return False


class C(Amphipod):
    energy = 100
    x = 6

    def __init__(self, pos):
        super().__init__(pos)
        self.energy = 100

    def __repr__(self):
        return 'C'

    def is_blocked(self):
        if self.hallway and (self.curpos[0] > 0) and (self.curpos[1] != 6):
            return True
        return False

    def is_fixed(self, locations):
        if self.curpos[1] == C.x:
            if all([isinstance(locations.locations[(pos_, C.x)], C) for pos_ in range(self.curpos[0] + 1, 4 + 1)]):
                return True
            else:
                # print(f"{self} in {self.curpos} -> "
                #       f"is_fixed is False [(4,{C.x}) is : {locations.locations[(4, C.x)]}]")
                return False
        else:
            return False


class D(Amphipod):
    energy = 1000
    x = 8

    def __init__(self, pos):
        super().__init__(pos)
        self.energy = 1000

    def __repr__(self):
        return 'D'

    def is_blocked(self):
        if self.hallway and (self.curpos[0] > 0) and (self.curpos[1] != 8):
            return True
        return False

    def is_fixed(self, locations):
        if self.curpos[1] == D.x:
            if all([isinstance(locations.locations[(pos_, D.x)], D) for pos_ in range(self.curpos[0] + 1, 4 + 1)]):
                return True
            else:
                # print(f"{self} in {self.curpos} -> "
                #       f"is_fixed is False [(4,{D.x}) is : {locations.locations[(4, D.x)]}]")
                return False
        else:
            return False


g0 = {(1, 2): A, (2, 2): A, (3, 2): A, (4, 2): A,
      (1, 4): B, (2, 4): B, (3, 4): B, (4, 4): B,
      (1, 6): C, (2, 6): C, (3, 6): C, (4, 6): C,
      (1, 8): D, (2, 8): D, (3, 8): D, (4, 8): D}

goals = [g0]
goals_lists = []
for g_ in goals:
    locs = {v: [] for v in g_.values() if v is not None}
    for i, j in g_.items():
        if j is not None:
            locs[j] += [i]
    goals_lists.append(locs)


class Locations:

    def __init__(self, list_of_pods):
        self.locations = {sp: None for sp in spots}
        self.cost = 0
        self.heur = 0
        for pod in list_of_pods:
            self.locations[pod.curpos] = pod
        self._update_pods()
        self._set_id()

    def generate_new(self, old_spot, new_spot):
        pods = [copy.deepcopy(pod) for pod in self.locations.values() if pod is not None]
        new_loc = Locations(pods)
        new_loc.cost = self.cost
        assert new_loc.locations[new_spot] is None, \
            f"New location is expected to be None ! {new_loc.locations[new_spot]}"
        new_loc.locations[new_spot] = new_loc.locations[old_spot]
        new_loc.locations[old_spot] = None
        new_loc.locations[new_spot].curpos = new_spot
        new_loc.cost += distance_pod(old_spot, new_spot) * new_loc.locations[new_spot].energy
        # print(f"[GENERATE_NEW] old, new -> {old_cost}, {new_cost}")
        new_loc._update_pods()
        new_loc._set_id()
        return new_loc

    def _update_pods(self):
        for pod in self.locations.values():
            if pod is not None:
                pod.update_asp()
                pod.update_next_asp(self)
        self.set_heuristic()

    def rep_ok(self):
        pass

    def exist_path(self, start, stop):
        # print(f"Exist_path {start},{stop},{hash(self)}")
        if self.locations[stop] is not None:
            # need to go to an empty spot
            return False
        if start == stop:
            # we are there
            return True
        elif start[0] == 4:
            # si on est en (4, X), on doit forcément aller vers le dessus. On repart du 3
            new_start = (3, start[1])
        elif 3 >= start[0] >= 1:
            # si on est en (i, X) {(1, X) ou (2, X) ou (3, X)}: soit vers (i-1, X), soit vers (i+1, X)
            if start[1] == stop[1]:
                # we are in the same column -> only 0 axis counts:
                assert start[0] != stop[0], "at this point, start[0] and stop[0] should be different"
                if start[0] > stop[0]:
                    new_start = (start[0] - 1, start[1])
                elif start[0] < stop[0]:
                    new_start = (start[0] + 1, start[1])
                else:
                    raise RuntimeError("I have no idea what I'm doint here: I expect same column and different height")
            else:
                # start is not in the same column as stop => need to go up !
                new_start = (start[0] - 1, start[1])


        elif start[0] == 0:
            # on est en (0, X) et on doit aller en (1, X), (2,X), (0, Y), (1, Y), (2, Y), (0, Z), (1, Z), (2, Z):
            if (stop[0] > 0) and (start[1] == stop[1]):
                # si on est en (0, X) et qu'on doit aller en (i, X) => seulement mouvement vertical vers le "bas"
                new_start = (1, start[1])
            elif stop[1] > start[1]:
                # si on est en (0, X), et qu'on doit aller en (?, Z) avec Z > X
                new_start = (0, start[1] + 1)
            elif stop[1] < start[1]:
                # si on est en (0, X), et qu'on doit aller en (?, Y) avec Y < X
                new_start = (0, start[1] - 1)
            else:
                raise RuntimeError(f"Unexpected \"else\" reached: start = {start}, stop = {stop}")
        else:
            raise RuntimeError(f"Unexpected \"else\" => start = {start}, stop = {stop}")

        if self.locations[new_start] is not None:
            return False
        else:
            return self.exist_path(new_start, stop)

    def is_goal_state(self):
        return all([isinstance(self.locations[(x, 2)], A) for x in range(0 + 1, 4 + 1)]) and \
               all([isinstance(self.locations[(x, 4)], B) for x in range(0 + 1, 4 + 1)]) and \
               all([isinstance(self.locations[(x, 6)], C) for x in range(0 + 1, 4 + 1)]) and \
               all([isinstance(self.locations[(x, 8)], D) for x in range(0 + 1, 4 + 1)])

    def get_pods_that_can_move(self):
        # returns a list of pods that can move
        pods = []
        for pod in self.locations.values():
            if (pod is not None) and (len(pod.next_asp[pod.curpos]) > 0):
                pods.append(pod)
        return pods

    def get_next_states(self):
        # only one pod change at a time
        pods = self.get_pods_that_can_move()

        new_states = []
        for pod in pods:
            l_ = list(pod.next_asp[pod.curpos])
            new_locs = [self.generate_new(pod.curpos, new_pos) for new_pos in l_]
            new_states.append(new_locs)

        flattened = [state for states in new_states for state in states]
        # with open('debug.out', 'a+') as f:
        #     f.write("---------------------------------------\n")
        #     f.write("next states {} = {}".format(len(flattened), flattened))
        return flattened

    def set_heuristic(self):
        # return the minimal cost to reach goal (heuristic)
        locs = {type(v): [] for v in self.locations.values() if v is not None}
        for i, j in self.locations.items():
            if j is not None:
                locs[type(j)] += [i]
        # print("locs = {}".format(locs))
        mgl = []
        for gl in goals_lists:
            mg = 0
            for cl in [A, B, C, D]:
                mg += min([
                    distance_pod(locs[cl][0], gl[cl][i]) +
                    distance_pod(locs[cl][1], gl[cl][j]) +
                    distance_pod(locs[cl][2], gl[cl][k]) +
                    distance_pod(locs[cl][3], gl[cl][l])
                    for i, j, k, l in itertools.permutations([0, 1, 2, 3], 4)
                ]) * cl.energy
            mgl += [mg]

        heur = min(mgl)
        self.heur = heur

    def can_be_filled(self, row):
        d_types = {2: A, 4: B, 6: C, 8: D}
        if isinstance(row, int):
            if all([isinstance(self.locations[(ind, d_types[row].x)], type(d_types[row])) or None
                    for ind in [1, 2, 3, 4]]):
                return True
        elif row in d_types.values():
            if all([isinstance(self.locations[(ind, row.x)], type(row)) or None
                    for ind in [1, 2, 3, 4]]):
                return True
        return False

    def __repr__(self):
        l0 = [c for c in '#############\n']
        l1 = [c for c in '#...........#\n']
        l2 = [c for c in '###.#.#.#.###\n']
        l3 = [c for c in '   .#.#.#.   \n']
        l4 = [c for c in '   .#.#.#.   \n']
        l5 = [c for c in '   .#.#.#.   \n']
        l6 = [c for c in '   #######   \n']
        L = [l0, l1, l2, l3, l4, l5, l6]
        for k in self.locations.keys():
            if self.locations[k] is not None:
                coord = self.locations[k].curpos
                L[coord[0] + 1][coord[1] + 1] = str(self.locations[k])
        return '\n' + ''.join([''.join(l) for l in L]) + f'\tcost:{self.cost}, heur:{self.heur}, id:{self.id}\n'
        # return "loc"

    def _set_id(self):
        locs = [(it[0], str(it[1])) for it in self.locations.items() if it[1] is not None]
        locs.sort(key=lambda x: x[0])
        # print(tuple(locs))
        self.id = hash(tuple(locs))

    def is_blocked(self):
        pod_is_blocked = any([pod.is_blocked() for pod in self.locations.values() if pod is not None])
        cond = pod_is_blocked
        return cond


class PP:
    def __init__(self):
        self.loc = None
        self.hashset = dict()

    def add_loc(self, locations: Locations):
        # print(f"[add_loc] {locations}")
        h_ = locations.id
        # print("Location = {}, hash={}".format(locations, h_))
        if h_ in self.hashset.values():
            # print("Introducing loops => None")
            return None  # contains loop
        elif locations.is_blocked():
            # print("Is Locked => None")
            return None
        else:
            new_pp = PP()
            new_pp.loc = copy.deepcopy(locations)
            new_pp.hashset = copy.deepcopy(self.hashset)

            assert new_pp.loc.cost == locations.cost, "the costs should be equal, no ?"
            return new_pp

    def __repr__(self):
        return f"<PP, {len(self.hashset.items())} steps >"


def simplify_queue(qs):
    # print("[SQ] qs length = {}".format([pp.loc.cost+pp.loc.heur for pp in qs]))
    cleaned_qs = qs

    for p, q in itertools.combinations(qs, 2):
        cost_last_p = p.loc.cost
        hash_last_p = p.loc.id
        # print(f"hash last p : {hash_last_p}")
        # print(f"hashes q    : {q.hashset.values()}")
        if (p in cleaned_qs) and (hash_last_p in q.hashset.values()):
            q_cost = list(q.hashset.keys())[list(q.hashset.values()).index(hash_last_p)]
            if (cost_last_p >= q_cost):
                print("Removing {}:{} > {} from queue".format(cost_last_p, hash_last_p, q_cost))
                print("\tp hashset {} ".format(p.hashset))
                print("\tq hashset {} ".format(q.hashset))
                cleaned_qs.remove(p)
                continue
                # print("Removing p from queue : {}".format(p.loc))

    return cleaned_qs


def game():
    Ltest = [
        B((1, 2)), D((2, 2)), D((3, 2)), A((4, 2)),
        C((1, 4)), C((2, 4)), B((3, 4)), D((4, 4)),
        B((1, 6)), B((2, 6)), A((3, 6)), C((4, 6)),
        D((0, 10)), A((2, 8)), C((3, 8)), A((4, 8))
    ]
    L = [
        C((1, 2)), D((2, 2)), D((3, 2)), D((4, 2)),
        A((1, 4)), C((2, 4)), B((3, 4)), C((4, 4)),
        B((1, 6)), B((2, 6)), A((3, 6)), A((4, 6)),
        D((1, 8)), A((2, 8)), C((3, 8)), B((4, 8))
    ]
    # L = [
    #     A((0, 0)), A((2, 2)), A((3, 2)), A((4, 2)),
    #     B((1, 4)), B((2, 4)), B((3, 4)), B((4, 4)),
    #     C((1, 6)), C((2, 6)), C((3, 6)), C((4, 6)),
    #     D((0, 10)), D((2, 8)), D((3, 8)), D((4, 8))
    # ]
    # L = [A((2, 2)), A((2, 8)), B((1, 2)), B((1, 6)), C((1, 4)), C((2, 6)), D((2, 4)), D((0, 9))]
    # L = [A((1, 2)), A((2, 2)), B((1, 6)), B((2, 4)), C((1, 4)), C((2, 6)), D((2, 8)), D((1, 8))]
    init_ = Locations(L)
    print(init_)
    root = PP().add_loc(init_)
    root.hashset[root.loc.cost] = root.loc.id
    queue = [root]
    counter = 0
    Qs = {root.loc.id: 0}
    tnew = 0

    while not (len(queue) == 0 or queue[0].loc.is_goal_state()):
        with open('debug.out', 'a+') as f:
            f.write("[NEW Iteration] queue {} = \n{}\n".format(len(queue), queue[0].loc))
            # f.write("{}\n{}\n".format([pp.loc.cost for pp in queue], [pp.loc.heur for pp in queue]))
            f.write("{}\n".format(queue[0].hashset))
            f.write("---------------------------------------\n")

        # if (len(queue) % 10) == 0:
        print("Iter {}, len queue = {}, Qs = {}, tnew = {}".format(counter, len(queue), len(Qs.keys()), tnew))
        partial_path = queue[0]
        queue = queue[1:]

        assert isinstance(partial_path, PP)
        new_possible_states = []
        next_states = partial_path.loc.get_next_states()
        to_remove_hash = set()
        for loc_ in next_states:
            assert isinstance(loc_, Locations), "this shall be a Location instance"
            next_ = partial_path.add_loc(loc_)
            if next_ is not None:
                assert next_.loc.cost > partial_path.loc.cost, \
                    f"The cost shall increase {next_.loc.cost} > {partial_path.loc.cost} !"

            if (next_ is not None) and \
                    (next_.loc.id not in next_.hashset.values()) and \
                    ((next_.loc.id not in Qs.keys()) or
                     (next_.loc.id in Qs.keys() and Qs[next_.loc.id] > next_.loc.cost)):
                t0 = time.time()
                if next_.loc.id in Qs.keys() and Qs[next_.loc.id] > next_.loc.cost:
                    to_remove_hash.add((next_.loc.id, Qs[next_.loc.id]))
                tnew += (time.time() - t0)

                next_.hashset[next_.loc.cost] = next_.loc.id  # todo: before, was in add_loc
                new_possible_states.append(next_)
                Qs[next_.loc.id] = next_.loc.cost

        #### remove from queue the PP that ends with
        t0 = time.time()
        for hash_, cost_ in to_remove_hash:
            for pp in queue:
                if hash_ in pp.hashset.values():
                    if cost_ in pp.hashset.keys():
                        if hash_ == pp.hashset[cost_]:
                            queue.remove(pp)
                            continue
        t01 = time.time()
        tnew += t01 - t0

        # print("new_possible_states = {}".format(new_possible_states))
        assert all(isinstance(i, PP) for i in new_possible_states), "New_Possible_states should be PPs"
        # assert isinstance(new_possible_states, list), f"is instance of: {type(new_possible_states[0])}"
        queue = queue + new_possible_states
        # if len(queue) > 0:
        #     assert isinstance(queue[0], PP)
        queue.sort(key=lambda x: x.loc.cost + x.loc.heur)
        # if len(queue) > 0:
        #     assert isinstance(queue[0], PP)

        # t0 = time.time()
        # if counter % 1 == 0:
        #     queue = simplify_queue(queue)
        # told += time.time() - t0

        counter += 1


    if len(queue) == 0:
        print(False)
        return False
    elif queue[0].loc.is_goal_state():
        print(queue[0].loc.cost)
        with open('debug.out', 'a+') as f:
            f.write("=========================================")
            f.write("queue (end iteration) {} = {}\n".format(len(queue), [pp.loc for pp in queue]))
            f.write("---------------------------------------\n")
        return queue[0].loc.cost


def assess_locations():
    # a1 = A((2, 2))
    # a2 = A((2, 8))
    # b1 = B((1, 2))
    # b2 = B((1, 6))
    # c1 = C((1, 4))
    # c2 = C((2, 6))
    # d1 = D((2, 4))
    # d2 = D((1, 8))
    # L = [a1, a2, b1, b2, c1, c2, d1, d2]
    # loc = Locations(L)
    #
    # print(loc)
    # print(loc.is_goal_state())
    # print(loc.get_pods_that_can_move())
    # # print(loc.get_next_states())
    #
    # a1 = A((1, 8))
    # a2 = A((2, 8))
    # b1 = B((1, 4))
    # b2 = B((2, 4))
    # c1 = C((1, 6))
    # c2 = C((2, 6))
    # d1 = D((2, 2))
    # d2 = D((1, 2))
    # loc = Locations([a1, a2, b1, b2, c1, c2, d1, d2])
    #
    # print(loc)
    # print(loc.is_goal_state())
    # print(loc.get_pods_that_can_move())
    #
    # a1 = A((1, 2))
    # a2 = A((2, 2))
    # b1 = B((0, 3))
    # b2 = B((0, 5))
    # c1 = C((2, 6))
    # c2 = C((2, 4))
    # d1 = D((2, 8))
    # d2 = D((1, 8))
    # loc = Locations([a1, a2, b1, b2, c1, c2, d1, d2])
    #
    # print(loc)
    # print("is blocked ? {}".format(loc.is_blocked()))
    Ltest = [
        B((1, 2)), D((2, 2)), D((3, 2)), A((4, 2)),
        C((1, 4)), C((2, 4)), B((3, 4)), D((4, 4)),
        B((1, 6)), B((2, 6)), A((3, 6)), C((4, 6)),
        D((0, 10)), A((2, 8)), C((3, 8)), A((4, 8))
    ]
    loc = Locations(Ltest)
    #
    print(loc)
    print("is blocked ? {}".format(loc.is_blocked()))
    print("exist path ? {}".format(loc.exist_path((1, 2), (1, 8))))
    print("exist path ? {}".format(loc.exist_path((1, 2), (0, 0))))
    print("exist path ? {}".format(loc.exist_path((1, 2), (0, 7))))
    print("exist path ? {}".format(loc.exist_path((1, 2), (0, 9))))

    L2 = [
        C((1, 2)), D((2, 2)), D((3, 2)), D((4, 2)),
        A((0, 0)), A((0, 1)), D((0, 3)), A((0, 10)),
        B((2, 4)), B((3, 4)), B((4, 4)),
        C((3, 6)), C((4, 6)),
        A((2, 8)), C((3, 8)), B((4, 8))
    ]
    loc = Locations(L2)

    print(loc)
    print("is blocked ? {}".format(loc.is_blocked()))


def assess_goal():

    Ltest = [
        A((0, 0)), A((2, 2)), A((3, 2)), A((4, 2)),
        B((1, 4)), B((2, 4)), B((3, 4)), B((4, 4)),
        C((1, 6)), C((2, 6)), C((3, 6)), C((4, 6)),
        D((0, 10)), D((2, 8)), D((3, 8)), D((4, 8))
    ]
    loc = Locations(Ltest)
    print("0,10 to 1,2 => {}".format(loc.locations[(0,10)]._can_move_to((1, 2))))
    print("0,10 to 1,8 => {}".format(loc.locations[(0,10)]._can_move_to((1, 8))))
    print("0,0 to 1,2 => {}".format(loc.locations[(0,0)]._can_move_to((1, 2))))
    print("0,0 to 1,8 => {}".format(loc.locations[(0,0)]._can_move_to((1, 8))))

    print(loc)
    print("is blocked ? {}".format(loc.is_blocked()))
    print("is goal state ? {}".format(loc.is_goal_state()))
    print("is fixed D  ? {}".format(loc.locations[(0,10)].is_fixed(loc)))
    print("is fixed A  ? {}".format(loc.locations[(0,0)].is_fixed(loc)))




def assess_hash():
    L = [A((1, 2)), A((2, 2)), B((1, 6)), B((2, 4)), C((1, 4)), C((2, 6)), D((2, 8)), D((1, 8))]
    init_ = Locations(L)
    print(init_.id)
    L = [A((2, 2)), A((1, 2)), B((1, 6)), B((2, 4)), C((1, 4)), C((2, 6)), D((1, 8)), D((2, 8))]
    init_ = Locations(L)
    print(init_.id)


def assess_distance():
    print(f"5 -> {distance_pod((0, 0), (5, 0))}")
    print(f"1 -> {distance_pod((0, 1), (0, 2))}")
    print(f"4 -> {distance_pod((0, 1), (1, 2))}")


if __name__ == '__main__':
    # assess_goal()
    # assess_locations()
    # assess_hash()
    # assess_distance()
    # ok, but about 6 hours to find solution :p (lol)
    cProfile.run('game()')
