from optimization.CollisionChecker import CollisionChecker


class Optimizer:
    def __init__(self, paths: dict[int, list[(int, int)]]):
        """
        @param paths Dict with the paths of the robots as follow: {0: [(1,1), (1,2), (2,2)], 1: [...], ...}
        """
        self.paths: dict[int, list[(int, int)]] = paths
        self._optimized_paths: list[dict[int, list[(int, int)]]] = []

    def remove_loops(self) -> dict[int, list[(int, int)]]:
        removed = {}
        for robot in self.paths.keys():
            removed[robot] = self._remove_loops(self.paths[robot])
        return removed

    def _remove_loops(self, path: list[(int, int)]) -> list[(int, int)]:
        for i in range(len(path)):
            for j in range(len(path) - 1, 0, -1):
                if i != j and path[i] == path[j]:
                    return self._remove_loops(path[:i] + path[j:])
        return path

    def remove_collisions_by_halt(self, paths: dict[int, list[(int, int)]]) -> dict[int, list[(int, int)]]:
        cpaths = paths.copy()
        collisions = CollisionChecker(paths).check_collisions()
        for collision in collisions:
            cpaths[collision['r1']], cpaths[collision['r2']] = self._solve_collision(cpaths[collision['r1']],
                                                                                   cpaths[collision['r2']],
                                                                                   collision['t'])
        return cpaths

    @staticmethod
    def _solve_collision(path1: list[(int, int)], path2: list[(int, int)], t: int) -> (
    list[(int, int)], list[(int, int)]):
        if len(CollisionChecker({0: path1, 1: path2}).check_collisions()) == 0:
            return path1, path2

        for i in range(1, t+1):
            if t-i >= len(path1):
                continue
            cpath1 = path1[:max(t - i, 0)] + ([path1[max(t - i, 0)]] * i) + path1[max(t - i, 0):]
            if len(CollisionChecker({0: cpath1, 1: path2}).check_collisions()) == 0:
                return cpath1, path2

        for i in range(1, t+1):
            if t-i >= len(path2):
                continue
            cpath2 = path2[:max(t - i, 0)] + ([path2[max(t - i, 0)]] * i) + path2[max(t - i, 0):]
            if len(CollisionChecker({0: path1, 1: cpath2}).check_collisions()) == 0:
                return path1, cpath2
        return path1, path2

    def _best(self, paths: [dict[int, list[(int, int)]]]) -> dict[int, list[(int, int)]]:
        best: dict[int, list[(int, int)]] = None
        b_col: int = -1
        b_len = -1
        for p in paths:
            col = len(CollisionChecker(p).check_collisions())
            p_len = sum([len(a) for a in p.values()])
            if best is None or col < b_col or (col == b_col and p_len <= b_len):
                best = p
                b_col = col
                b_len = p_len
        return best

    def get_optimized(self) -> dict[int, list[(int, int)]]:
        self._optimized_paths += [self.remove_loops()]
        for i in range(len(self._optimized_paths)):
            self._optimized_paths += [self.remove_collisions_by_halt(self._optimized_paths[i])]

        #self._optimized_paths += [self.remove_collisions_by_halt(self.paths)]
        return self._best(self._optimized_paths)


if __name__ == '__main__':
    paths = {0: [(8, 10), (8, 11), (7, 11), (6, 11), (6, 10)],
             1: [(9, 11), (8, 11), (7, 11), (6, 11)]}

    print(Optimizer(paths).get_optimized())
