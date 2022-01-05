import math

from aoc_utils.source import *


class SnailFishNumber:
    def __init__(self, left=0, right=0, parent=None):
        self.left = left
        self.right = right
        self.parent = parent

    def add(self, number):
        res = SnailFishNumber()
        res.left = self
        res.right = number
        self.parent = res
        number.parent = res
        return res.reduce()

    def reduce(self):
        # print("[reduce] self = {}".format(self))
        t = self.get_leftmost_nested()
        u = self.get_leftmost_ten_or_greater()

        if t or u:
            # print("[reduce] self.get_leftmost_nested() = {}".format(t))
            # print("[reduce] self.contains_ten_or_greater() = {}".format(u))
            if t:
                # print(f"is nested: explode pair {t}")
                t.explode()
                self.reduce()
            if u:
                # print(f"contains ten or greater {u}")
                u.split()
                self.reduce()

        return self

    def is_nested(self):
        if self.parent is not None:
            if self.parent.parent is not None:
                if self.parent.parent.parent is not None:
                    if self.parent.parent.parent.parent is not None:
                        return self
        return False

    def contains_ten_or_greater(self):
        res1 = res2 = res3 = res4 = False
        if isinstance(self.left, int):
            if self.left >= 10:
                res1 = self
        if isinstance(self.left, SnailFishNumber):
            res2 = self.left.contains_ten_or_greater()
        if isinstance(self.right, int):
            if self.right >= 10:
                res3 = self
        if isinstance(self.right, SnailFishNumber):
            res4 = self.right.contains_ten_or_greater()

        return res1 or res2 or res3 or res4

    def explode(self):
        # find first regular number on the left
        p_left = self.find_first_regular("left")
        if p_left:
            if isinstance(p_left.right, int):
                p_left.right += self.left
            else:
                p_left.left += self.left
        # find first regular number on the right
        p_right = self.find_first_regular("right")
        if p_right:
            if isinstance(p_right.left, int):
                p_right.left += self.right
            else:
                p_right.right += self.right

        if self == self.parent.left:
            self.parent.left = 0
        else:
            self.parent.right = 0
        # if no number, do nothing. Else, add self.right to this number

    def find_first_regular(self, direction):
        p = self.parent

        if direction == "left":
            if self == p.right:
                if isinstance(p.left, int):
                    # print(f"Found int {p.left} Left of {self}")
                    return p
                elif isinstance(p.left, SnailFishNumber):
                    return p.left._find_regular_child(direction="rightmost")
            elif self == p.left:
                gp = p.parent
                if p == gp.right:
                    res1 = res2 = False
                    if isinstance(gp.left, int):
                        # print(f"Found int {gp.left} Left of {self}")
                        return gp
                    elif isinstance(gp.left, SnailFishNumber):
                        res1 = gp.left._find_regular_child(direction="rightmost")
                    else:
                        res2 = p.find_first_regular(direction)
                    return res1 or res2
                else:
                    # p is left; gp is left
                    ggp = gp.parent
                    if gp == ggp.right:
                        res1 = res2 = False
                        if isinstance(ggp.left, int):
                            # print(f"Found int {ggp.left} Left of {self}")
                            return ggp
                        elif isinstance(ggp.left, SnailFishNumber):
                            res1 = ggp.left._find_regular_child(direction="rightmost")
                        else:
                            res2 = ggp.find_first_regular(direction)
                        return res1 or res2
                    else:
                        if ggp.parent is not None:
                            return p.find_first_regular(direction)
                        else:
                            return False

        elif direction == "right":
            if self == p.left:
                if isinstance(p.right, int):
                    # print(f"Found int {p.right} Right of {self}")
                    return p
                elif isinstance(p.right, SnailFishNumber):
                    return p.right._find_regular_child(direction="leftmost")
            elif self == p.right:
                gp = p.parent
                if p == gp.left:
                    res1 = res2 = False
                    if isinstance(gp.right, int):
                        # print(f"Found int {gp.right} Right of {self}")
                        return gp
                    elif isinstance(gp.right, SnailFishNumber):
                        res1 = gp.right._find_regular_child(direction="leftmost")
                    else:
                        res2 = p.find_first_regular(direction)
                    return res1 or res2
                else:
                    # p is left; gp is left
                    ggp = gp.parent
                    if gp == ggp.left:
                        res1 = res2 = False
                        if isinstance(ggp.right, int):
                            # print(f"Found int {ggp.right} Right of {self}")
                            return ggp
                        elif isinstance(ggp.right, SnailFishNumber):
                            res1 = ggp.right._find_regular_child(direction="leftmost")
                        else:
                            res2 = ggp.find_first_regular(direction)
                        return res1 or res2
                    else:
                        if ggp.parent is not None:
                            return p.find_first_regular(direction)
                        else:
                            return False
        else:
            raise ValueError("direction not understood : {}".format(direction))

    def _find_regular_child(self, direction):
        q = [self]
        qp = [self.parent]
        while len(q) > 0 and not (isinstance(q[0], int)):
            c = q[0]
            q = q[1:]
            qp = qp[1:]  # remembers the parents of leftmost / rightmost
            if direction == "rightmost":
                new_c = [c.right, c.left]
                new_cp = [c, c]
            elif direction == "leftmost":
                new_c = [c.left, c.right]
                new_cp = [c, c]
            else:
                raise ValueError("direction in [leftmost, rightmost], found {}".format(direction))
            q = new_c + q
            qp = new_cp + qp
        if isinstance(q[0], int):
            # print(f"Found int {q[0]}")
            return qp[0]
        else:
            return False

    def get_leftmost_nested(self):
        q = [self]
        while len(q) > 0 and not (q[0].is_nested()):
            c = q[0]
            q = q[1:]
            # qp = qp[1:] # remembers the parents of leftmost / rightmost
            new_c = []
            if isinstance(c.left, SnailFishNumber):
                new_c.append(c.left)
            if isinstance(c.right, SnailFishNumber):
                new_c.append(c.right)

            q = new_c + q

        if (len(q) > 0) and (q[0].is_nested()):
            return q[0]
        else:
            return False

    def get_leftmost_ten_or_greater(self):
        return self.contains_ten_or_greater()

    def split(self):
        elem = self.get_leftmost_ten_or_greater()
        # print("split: found greater parent element: {}".format(elem))
        if elem:
            if isinstance(elem.left, int) and elem.left >= 10:
                val = elem.left
                new = SnailFishNumber(left=int(math.floor(val / 2)),
                                      right=int(math.ceil(val / 2)))
                elem.left = new
                new.parent = elem
            elif isinstance(elem.right, int) and elem.right >= 10:
                val = elem.right
                new = SnailFishNumber(left=int(math.floor(val / 2)),
                                      right=int(math.ceil(val / 2)))
                elem.right = new
                new.parent = elem
            else:
                raise ValueError('one of the children is expected to be an int >= 10')

    def get_magnitude(self):
        if isinstance(self.left, int) and isinstance(self.right, int):
            return self.left*3 + self.right*2
        elif isinstance(self.left, int) and isinstance(self.right, SnailFishNumber):
            return self.left*3 + self.right.get_magnitude()*2
        elif isinstance(self.left, SnailFishNumber) and isinstance(self.right, int):
            return self.left.get_magnitude()*3 + self.right*2
        else:
            return self.left.get_magnitude()*3 + self.right.get_magnitude()*2

    def __repr__(self):
        return f"[{self.left.__repr__()},{self.right.__repr__()}]"


def constructor(nested_list):
    new = SnailFishNumber()
    if not isinstance(nested_list, list):
        raise ValueError('nested list unknown : {}'.format(nested_list))

    if isinstance(nested_list[0], int) and isinstance(nested_list[1], int):
        new = SnailFishNumber(nested_list[0], nested_list[1], parent=None)
    elif isinstance(nested_list[0], int) and isinstance(nested_list[1], list):
        # [1, [2, 3]]
        child_right = constructor(nested_list[1])
        child_right.parent = new
        new.right = child_right
        new.left = nested_list[0]
    elif isinstance(nested_list[0], list) and isinstance(nested_list[1], int):
        new = SnailFishNumber()
        child_left = constructor(nested_list[0])
        child_left.parent = new
        new.left = child_left
        new.right = nested_list[1]
    elif isinstance(nested_list[0], list) and isinstance(nested_list[1], list):
        new = SnailFishNumber()
        child_left = constructor(nested_list[0])
        child_left.parent = new

        child_right = constructor(nested_list[1])
        child_right.parent = new

        new.left = child_left
        new.right = child_right
    else:
        raise ValueError('nested list unknown : {}'.format(nested_list))

    return new


def test_find_regular():
    sn = constructor([[[[1, 2], [3, [4, 5]]], [[6, 7], [8, 9]]], [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]])
    print("starting with sn :{}".format(sn))
    ts = [sn.left.left.left, sn.left.left.right, sn.left.left.right.right, sn.left.right.left, sn.left.right.right,
          sn.right.left.left, sn.right.left.right, sn.right.right.left, sn.right.right.right]
    for t in ts:
        print("child of interest: self = {}".format(t))
        print("is candidate nested:{}".format(t.is_nested()))
        if t.is_nested():
            ans = t.find_first_regular("left")
            print("Left = ", ans)
            ans = t.find_first_regular("right")
            print("Right = ", ans)
        print("---------------------------------")

    sn = constructor([0, [[[1, 2], 3], 4]])
    print("starting with sn :{}".format(sn))
    ts = [sn.right.left.left]
    for t in ts:
        print("child of interest: self = {}".format(t))
        print("is candidate nested:{}".format(t.is_nested()))
        if t.is_nested():
            ans = t.find_first_regular("left")
            print("Left = ", ans)
            ans = t.find_first_regular("right")
            print("Right = ", ans)
        print("---------------------------------")


def test_explode():
    Ls = (
        [[[[[9, 8], 1], 2], 3], 4],
        [7, [6, [5, [4, [3, 2]]]]],
        [[6, [5, [4, [3, 2]]]], 1],
        [[3, [2, [1, [7, 3]]]], [6, [5, [4, [3, 2]]]]],
        [[3, [2, [8, 0]]], [9, [5, [4, [3, 2]]]]],
        [[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]]
    )

    ans = []
    for L in Ls:
        sn = constructor(L)
        t = sn.get_leftmost_nested()
        if t:
            t.explode()
            ans.append(sn)
        else:
            ans.append("yippee")
    correct_ans = (
        [[[[0, 9], 2], 3], 4],
        [7, [6, [5, [7, 0]]]],
        [[6, [5, [7, 0]]], 3],
        [[3, [2, [8, 0]]], [9, [5, [4, [3, 2]]]]],
        [[3, [2, [8, 0]]], [9, [5, [7, 0]]]],
        [[[[0,7],4],[7,[[8,4],9]]],[1,1]]
    )

    for i, j in zip(ans, correct_ans):
        print("{} =?= {}".format(i, j))


def debug_regular():
    L = [[6, [5, [4, [3, 2]]]], 1]
    sn = constructor(L)
    t = sn.get_leftmost_nested()
    a = t.find_first_regular(direction="right")
    print("found: {}".format(a))


def test_split():
    L = [[[[0, 7], 4], [15, [0, 13]]], [1, 1]]
    sn = constructor(L)
    print(sn)
    # a = sn.contains_ten_or_greater()
    # print(a)
    sn.split()
    print(sn)
    sn.split()
    print(sn)

def test_addition():
    L1 = [[[[4,3],4],4],[7,[[8,4],9]]]
    L2 = [1,1]

    sn1 = constructor(L1)
    sn2 = constructor(L2)
    sn3 = sn1.add(sn2)
    print(sn3)

def test_magnitude():
    Ls = (
        [[1,2],[[3,4],5]],
        [[[[0,7],4],[[7,8],[6,0]]],[8,1]],
        [[[[1,1],[2,2]],[3,3]],[4,4]],
        [[[[3,0],[5,3]],[4,4]],[5,5]],
        [[[[5,0],[7,4]],[5,5]],[6,6]],
        [[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]]
    )
    Ans = (
        143,
        1384,
        445,
        791,
        1137,
        3488
    )

    a = []
    for L in Ls:
        sn = constructor(L)
        a.append(sn.get_magnitude())

    for i, j in zip(a, Ans):
        print("{} =?= {}".format(i, j))
