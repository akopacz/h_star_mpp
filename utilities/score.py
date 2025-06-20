from typing import List, Tuple, Iterable

def get_avg_path_len(paths:Iterable[List[Tuple[int]]]):
    nr = 0
    s = 0
    for p in paths:
        s += len(p)
        nr += 1
    if nr > 0:
        return s / nr
    else:
        return 0.

def get_sum_path_len(paths:Iterable[List[Tuple[int]]]):
    return sum(map(len, paths))

def get_max_path_len(paths:Iterable[List[Tuple[int]]]):
    return max(map(len, paths))