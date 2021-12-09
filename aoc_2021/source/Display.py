class Display:
    def __init__(self):
        self.segments = {"x0": {"a", "b", "c", "d", "e", "f", "g"},
                         "x1": {"a", "b", "c", "d", "e", "f", "g"},
                         "x2": {"a", "b", "c", "d", "e", "f", "g"},
                         "x3": {"a", "b", "c", "d", "e", "f", "g"},
                         "x4": {"a", "b", "c", "d", "e", "f", "g"},
                         "x5": {"a", "b", "c", "d", "e", "f", "g"},
                         "x6": {"a", "b", "c", "d", "e", "f", "g"}}

    def process_code(self, code):
        cs = {c for c in code}
        if len(code) == 2:
            # digit = 2
            self.segments["x1"] = self.segments["x1"].intersection(cs)
            self.segments["x4"] = self.segments["x4"].intersection(cs)
        elif len(code) == 3:
            self.segments["x0"] = self.segments["x0"].intersection(cs)
            self.segments["x1"] = self.segments["x1"].intersection(cs)
            self.segments["x4"] = self.segments["x4"].intersection(cs)
        elif len(code) == 4:
            self.segments["x3"] = self.segments["x3"].intersection(cs)
            self.segments["x2"] = self.segments["x2"].intersection(cs)
            self.segments["x1"] = self.segments["x1"].intersection(cs)
            self.segments["x4"] = self.segments["x4"].intersection(cs)
        elif len(code) == 5:
            self.segments["x0"] = self.segments["x0"].intersection(cs)
            self.segments["x2"] = self.segments["x2"].intersection(cs)
            self.segments["x5"] = self.segments["x5"].intersection(cs)
        elif len(code) == 6:
            self.segments["x0"] = self.segments["x0"].intersection(cs)
            self.segments["x3"] = self.segments["x3"].intersection(cs)
            self.segments["x4"] = self.segments["x4"].intersection(cs)
            self.segments["x5"] = self.segments["x5"].intersection(cs)
        self._ensure_consistency()

    def decode(self, code):
        if len(code) == 2: return 1
        elif len(code) == 3: return 7
        elif len(code) == 4: return 4
        elif len(code) == 7: return 8
        else:
            segs = self._get_lit(code)
            if len(code) == 5:
                if (("x1" in segs) and ("x4" in segs)) \
                        or (len(self.segments["x3"]) == 1 and (not ("x3" in segs)) and
                            (len(self.segments["x6"]) == 1) and (not ("x6" in segs))):
                    return 3
                elif ("x1" in segs) or ("x6" in segs) \
                        or (len(self.segments["x3"]) == 1 and (not ("x3" in segs)) and
                            (len(self.segments["x4"]) == 1) and (not ("x4" in segs))):
                    return 2
                elif ("x4" in segs) or ("x3" in segs) \
                        or (len(self.segments["x1"]) == 1 and (not ("x1" in segs)) and
                            (len(self.segments["x6"]) == 1) and (not ("x6" in segs))):
                    return 5
                else:
                    print("unknown: \n segments lit = {}\n code = {}\n database = {}".format(segs, code, self.segments))

            elif len(code) == 6:
                if len({"x0", "x1", "x2", "x3"}.intersection(segs)) == 4 \
                        or ((len(self.segments["x6"]) == 1) and (not ("x6" in segs))):
                    return 9
                elif len({"x2", "x4", "x5", "x6"}.intersection(segs)) == 4 \
                        or ((len(self.segments["x1"]) == 1) and (not ("x1" in segs))):
                    return 6
                else:
                    return 0

    def _get_lit(self, code):
        # return all the segments that are certainly lit
        cs = {c for c in code}
        lits = []
        for seg in self.segments.keys():
            if len(self.segments[seg].difference(cs)) == 0: lits += [seg]
        return set(lits)

    def _ensure_consistency(self):
        diff_x0_x1 = self.segments["x0"].difference(self.segments["x1"])
        diff_x0_x4 = self.segments["x0"].difference(self.segments["x4"])
        # if X1 and X4, and if x0
        if (len(diff_x0_x1) == 1) and (diff_x0_x4 == diff_x0_x1):
            self.segments["x0"] = diff_x0_x4

        # redundant with next, but should decrease number of ops
        if len(self.segments["x1"]) < 3:
            self.segments["x3"] = self.segments["x3"].difference(self.segments["x1"])
            self.segments["x2"] = self.segments["x2"].difference(self.segments["x1"])

        if len(self.segments["x3"]) < 5 and len(self.segments["x1"]) == 3:
            self.segments["x3"] = self.segments["x3"].difference(self.segments["x1"])
            self.segments["x2"] = self.segments["x2"].difference(self.segments["x1"])

        for seg in self.segments.keys():
            if len(self.segments[seg]) == 1:
                self._remove_from_candidates(self.segments[seg])

    def _remove_from_candidates(self, unique_val):
        for seg in self.segments.keys():
            if len(self.segments[seg]) > 1:
                self.segments[seg] = self.segments[seg].difference(unique_val)
