# Hybrid Adaptive Greedy Algorithm Addressing the Multi-Robot Path Planning Problem

**Journal:** IEEE LATIN AMERICAN TRANSACTIONS

**Manuscript ID:** 9386 

**Authors**:
- Anikó Kopacz [1]
- Enol García González [2]
- Camelia Chira [1]
- José R. Villar [2]

**Affiliations:**

[1] Faculty of Mathematics and Computer Science, Babeș-Bolyai University, Romania

[2] Computer Science Department at University of Oviedo, Spain

---

In this repository, the implementation of the H* algorithm is provided for the paper "Hybrid Adaptive Greedy Algorithm Addressing the Multi-Robot Path Planning Problem".

## Requirements

- Python 3.9 or newer
- optional: Docker

## Execution

To execute the H* algorithm for a single input file run:
```
python testing_robots.py -i <coordinates file> --scenery <scenery file> -d <output directory> -e
```

To execute the H* algorithm for multiple files run:
```
python testing_robots.py -i <input directory> -o <output directory> -e
```
The input directory should have a directory for each scenario, each scenario's directory should contain a `scenery.txt`, and another directory with the files containing the coordinates:
```
input
├── scenario1
│   ├── coordinates
│   │   ├── 10coordinates.txt
│   │   ├── 11coordinates.txt
│   │   └── ...
│   └── scenery.txt
│   ...
└── scenario_n
    ├── coordinates
    │   ├── 5coordinates.txt
    │   ├── 6coordinates.txt
    │   └── ...
    └── scenery.txt
```
The file containing the scenery should be named `scenery.txt`, the names of the files with the coordinates and the names of the directories are arbitrary.

## Files

| Script or module | Description |
| --- | --- |
| `astar/` | Implements the A* algorithm for 1 robot |
| `draw.py` | Draws the robot paths for a given scenery and robot initial and target coordinates |
| `envs/` | Wraps a scenery and implements helper functions for simulating robot movements |
| `exact_policy/` | Implements a policy that schedules robots and decides robot movements; used in the `votes` module |
| `multi_astar/` | Plans multiple robot routes simultaneously using A*; implements the greedy path search (first step of the H* algorithm) |
| `optimization/` | Implements the local optimization operator (second step of the H* algorithm) |
| `testing_robots.py` | Plan routes with H*; run `python testing_robots.py -h` to list all run-time arguments|
| `test/` | Unit tests |
| `utilities/` | Helper functions |
| `votes/` | Implements the movement selection mechanism for the greedy path search |

## License

MIT license