import argparse
import csv
import logging
from argparse import RawTextHelpFormatter
from itertools import cycle
from os import listdir, makedirs, path
from time import time
from optimization import Optimizer

import numpy as np
from multi_astar.greedy_controller import GreedyController
from multi_astar.graph_routes import ShortestPathAfterSwitch
from multi_astar.graph_predictor import DistanceGraph
from multi_astar.observe_paths import ObserveManyPaths
from utilities import load_obstacle_matrix, load_robot_coordinates, show_routes
from envs.env import Agent, Env, reverse_dict
from utilities.paths import return_path
from utilities.score import get_max_path_len, get_sum_path_len

from torch.utils.tensorboard import SummaryWriter

def get_input_files(config):
    if config.get("input"):
        input_path = config["input"]
        if path.isdir(input_path):
            for scenery in listdir(input_path):
                sc_dir = path.join(input_path, scenery)
                sc_map = path.join(sc_dir, "scenery.txt")
                if path.isdir(sc_dir) and path.exists(sc_map):
                    for coords_part in listdir(sc_dir):
                        if coords_part == "scenery.txt":
                            continue
                        coords = path.join(sc_dir, coords_part)
                        if path.isdir(coords):
                            for f in listdir(coords):
                                yield sc_map, path.join(coords, f)
                        else:
                            yield sc_map, coords
        elif config.get("scenery"):
            yield config["scenery"], config["input"]   


all_envs = {
    "static": Env
}

def rl_test(config):
    if not path.exists(config['output_dir']):
        makedirs(config['output_dir'])

    logging.basicConfig(
        filename=f"{config['output_dir']}/train.log",
        format='%(asctime)s %(levelname)-6s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d'
    )
    logging.info(
        f"Collision penalty for agents: {config['collision_penalty']}")

    # init map
    input_files_iter = cycle(get_input_files(config))
    input_hint = None

    EnvClass = all_envs.get('static', Env)

    start_time = time()
    predictor = ShortestPathAfterSwitch(max_depth=100)
    observer = ObserveManyPaths(predictor)
    votes_size, nr_of_actions = (5, 5)
    controller = GreedyController(
        preprocess_obs=lambda x:x,
        nr_of_actions=nr_of_actions, 
        vote_size=votes_size
    )

    end_time = time()
    logging.info(f"Model setup: {end_time - start_time} s")

    # simulation
    train_policy = not config["eval"]
    epsilon = config["eps_start"]
    verbose = config.get("verbose", False)
    f = None
    training_statistics = np.zeros((4, config["train_rounds"]), dtype=float)
    writer = None
    if controller.training_policy:
        writer = SummaryWriter(log_dir=f"{config['output_dir']}/tb")
        controller.training_policy.set_writer(writer)

    start_time = time()
    for t in range(config["train_rounds"]):
        acc_reward = 0
        input_files = next(input_files_iter)
        if input_files != input_hint:
            matrix = load_obstacle_matrix(input_files[0])
            coordinates = load_robot_coordinates(input_files[1])
            input_hint = input_files
        if not coordinates or not isinstance(matrix, np.ndarray) or matrix.size == 0:
            # input files not found. skip this iteration
            logging.warn(f"{t} training iteration: input files not found, skip training")
            continue
        # init robots
        robots = [
            Agent(i, start, end)
            for i, (start, end) in enumerate(coordinates)
        ]
        env = EnvClass(
            maze=matrix,
            agents=robots,
            obs_builder=observer,
            distance_map=DistanceGraph(robots, *matrix.shape),
            collision_penalty=config["collision_penalty"]
        )
        obs = env.obs_builder.get_many()
        speed_dict = {a: 1 for a in range(env.nr_agents)}
        info = {
            "speed": speed_dict,
            "action_required": {ag.handle: ag.position != ag.target for ag in env.agents}
        }
        controller.reset(
            env.nr_agents,
            env.width,
            initial_obs=None
        )
        controller.prev_obs = controller.get_processed_obs(obs, info)
        paths = {a.handle: [a.initial_position] for a in env.agents}

        if (t) % config["checkpoints"] == 0 and verbose:
            f = open(f"{config['output_dir']}/simulation_step_{t}.csv", "w")
            csv_writer = csv.writer(f)
            csv_writer.writerow(["step", "agent", "x", "y", "finished",
                                "action_chosen", "action_taken", "reward", "votes"])

        s = env.max_steps
        episode_time = time()
        for i in range(env.max_steps):
            # action
            info = {
                "speed": speed_dict,
                "action_required": {ag.handle: ag.position != ag.target for ag in env.agents}
            }
            action_dict, processed_obs = controller.plan_action_combined_policy(
                obs=obs,
                info=info,
                train_policy=train_policy,
                choose_baseline_act=epsilon,
                preprocess_obs=True,
                epsilon=0
            )

            obs, reward = env.step(action_dict)
            done = {ag.handle: ag.finished for ag in env.agents}
            controller.train_rl_policy(
                actions=action_dict,
                rewards=reward,
                obs=processed_obs,
                done=done,
                action_required=info["action_required"]
            )
            acc_reward += sum(reward.values())
            if (t) % config["checkpoints"] == 0 and verbose:
                for ind, agent in enumerate(env.agents):
                    action_taken = reverse_dict[agent.position[0] -
                                                agent.prev_position[0], agent.position[1] - agent.prev_position[1]]
                    act = action_dict.get(ind)
                    logging.info(
                        f"  ind: {ind}, pos: {agent.position}, " +
                        f"selected act:{act}, taken act: {action_taken}, " +
                        f"reward:{reward[ind]}, votes:{processed_obs.get(ind, None)}")
                    csv_writer.writerow(
                        [i, ind, agent.position[0], agent.position[1],
                            agent.finished, act, action_taken, reward[ind],
                            *(processed_obs.get(ind, []))])
            for ind in range(env.nr_agents):
                if info["action_required"].get(ind, False):
                    paths[ind].append(env.agents[ind].position)

            if all(done.values()):
                if s == env.max_step:
                    logging.info("all agents arrived")
                    s = i + 1
        episode_time = time() - episode_time
        sc = get_sum_path_len(paths.values())
        logging.info(
            f"episode: {t}, execution time: {episode_time}, steps: {s}, nr of agents: {env.nr_agents}, accumulated reward: {acc_reward}, path length sum: {sc}")

        training_statistics[:, t] = [sc, get_max_path_len(
            paths.values()), acc_reward, episode_time]

        epsilon *= config["eps_decay"]
        if (t) % config["checkpoints"] == 0:
            if verbose:
                f.close()
            logging.info(f"paths: {paths}")
            controller.save_tr_policy(
                f"{config['output_dir']}/model{t}.pkl", t+1)
            print_statistics_for_paths(
                paths, matrix, coordinates, f"{config['output_dir']}/routes_{t}.png")
    optimized_paths = Optimizer(paths).get_optimized()
    run_time = time() - start_time
    show_routes(matrix, optimized_paths.values(), coordinates, filename=f"{config['output_dir']}/paths.png")

    logging.info(f"Training done in: {run_time} s (~{run_time // 60} min)")

    with open(f"{config['output_dir']}/stats_lengths.csv", "a") as f:
        f.write(f"{config['input']};{run_time};{[len(optimized_paths[r]) for r in optimized_paths.keys()]}\n")

    controller.save_tr_policy(
        f"{config['output_dir']}/model_end.pkl", config["train_rounds"])

    # print statistics to csv file
    np.savetxt(
        f"{config['output_dir']}/train_stats.csv",
        training_statistics.T,
        delimiter=',',
        fmt='%.4f',
        header="Columns: episode_path_length_sum, episode_path_length_max, episode_reward, episode_time"
    )

    # writer.add_hparams(dict(train_params._asdict()), {'hparams/score': self.score / self.episode_ind})
    if writer: writer.close()


def print_statistics_for_paths(paths, maze, coordinates, filename):
    show_routes(maze, paths.values(), coordinates, filename=filename)

    for a, r in paths.items():
        logging.info(f"{a}: {return_path(r, maze)}")


if __name__ == "__main__":
    pars = argparse.ArgumentParser(
        description="Plan robot paths using A* - first step",
        formatter_class=RawTextHelpFormatter)

    pars.add_argument('-i', '--input', type=str, default="./input/",
                      help='Path of the file(s) containing the information about the start and destination points of the robots. '
                      'Expected file structure: the input is either a file with the coordinates, or a directory, that contains scenarios, '
                      'where each scenario has a scenery.txt and multiple files with the coordinates.')
    pars.add_argument('--scenery', type=str,
                      default='./input/scenario1/scenery.txt', help='Route for the scenery file')
    pars.add_argument(
        "-d", "--output-dir", type=str, default="outputs",
        help="Relative path to output directory")
    pars.add_argument("--train-rounds", type=int, default=1,
                      help="number of restarts, default is 1")
    pars.add_argument("-c", "--checkpoints", type=int, default=1,
                      help="number of restarts to save model after")
    pars.add_argument("--eps-start", type=float, default=0.99,
                      help="starting probability of random exploration of actions during training")
    pars.add_argument("--eps-decay", type=float, default=0.98,
                      help="decay of the action exploration probability during training")
    pars.add_argument("-p", "--collision-penalty", type=int, default=-2,
                      help="(negative) reward received by agents when colliding, default is -2")
    pars.add_argument(
        "-e", "--eval", default=False, action="store_true",
        help="Evaluate model without any training"
    )
    pars.add_argument(
        "-v", "--verbose", default=False, action="store_true",
        help="Save paths to file for debugging"
    )
    
    args = pars.parse_args()

    rl_test(vars(args))
