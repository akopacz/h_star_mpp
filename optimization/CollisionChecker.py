class CollisionChecker:
    def __init__(self, paths: dict[any, list[(int, int)]]) -> None:
        self.paths: dict[any, list[(int, int)]] = paths
        self.collisions: list[dict[str, any]] = []

    def check_collisions(self) -> list[dict[str, any]]:
        """
        Method that looks for collisions between paths. It returns a list with the collisions. Each collision is a dict
        that stores the time (t), the robot 1 (r1) and the robot 2 (r2).

        @return dictionary with collision detected.
        """
        robots: list[any] = list(self.paths.keys())
        for t in range(max([len(p) for p in self.paths.values()])):
            for i in range(len(robots)):
                for j in range(i):
                    pos_i0: (int, int) = self._get_pos_at_time(t, robots[i])
                    pos_j0: (int, int) = self._get_pos_at_time(t, robots[j])
                    pos_i1: (int, int) = self._get_pos_at_time(t - 1, robots[i])
                    pos_j1: (int, int) = self._get_pos_at_time(t - 1, robots[j])

                    if pos_i0 == pos_j0 or (pos_i1 == pos_j0 and pos_i0 == pos_j1):
                        self.collisions += [{
                            't': t,
                            'r1': robots[i],
                            'r2': robots[j]
                        }]
        return self.collisions

    def _get_pos_at_time(self, t: int, robot: any) -> (int, int):
        return self.paths[robot][t if t < len(self.paths[robot]) else -1]
