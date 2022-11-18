"""Redo the data_collection_script using the original RL environment.

Because I think the gym environment should be setup in a way that I don't need
to edit anything even when I am replacing an agent with a human.

1. Initiate joystick
2. Initiate a gym environment
3. start a real-time stepping and save data
4. Log the data to disk
"""


import argparse
import glob
import os
import pickle
import sys
import time
from typing import Dict, Optional

import numpy as np
import pygame
from dotenv import load_dotenv

from src.async_arch.envs.aircraft.agent_defs import AcclerationAgent, AcclerationAgent2
from src.async_arch.envs.aircraft.aircraft_gym import AircraftEnv
from src.async_arch.envs.aircraft.aircraft_params import add_aircraft_env_args


def get_joy_vals(joystick_no: int = 0):
    joystick = pygame.joystick.Joystick(joystick_no)
    joystick.init()
    axis0 = joystick.get_axis(0)
    axis1 = -joystick.get_axis(1)
    axis2 = joystick.get_axis(2)
    axis3 = -joystick.get_axis(3)
    return [axis0, axis1, axis2, axis3]


def get_latest_run_id(log_path: Optional[str] = None, log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :return: latest run number
    """
    max_run_id = -1
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if (
            log_name == "_".join(file_name.split("_")[:-1])
            and ext.isdigit()
            and int(ext) > max_run_id
        ):
            max_run_id = int(ext)
    return max_run_id


def dump_trajectories(states: Dict, actions: Dict, rews: Dict, terminals: Dict):
    logs_folder = "data"
    log_name = "eps"
    run_id = get_latest_run_id(logs_folder, log_name) + 1
    save_path = f"{logs_folder}/{log_name}_{run_id}"

    # all of the data is of form Dict[step, Dict/array/int/bool]
    # so I can dump it as one Dict[step, tuple(s, a, r, t)]
    ds = [states, actions, rews, terminals]
    d = {}
    for k in states.keys():
        d[k] = tuple(d[k] for d in ds)

    with open(save_path, "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


def main():
    load_dotenv()
    pygame.joystick.init()

    # Data variables
    states_dict = {}
    actions_dict = {}
    rewards_dict = {}
    terminals_dict = {}

    # initialise a parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )
    add_aircraft_env_args("Aircraft_Env", parser)
    args = parser.parse_args()

    # make render true and agent interaction steps 1 (gives more data)
    args.agent_interaction_steps = 1
    args.no_render = False

    agent = AcclerationAgent()
    agent2 = AcclerationAgent2()  # because I want different initial conditions

    # var = 0.1 * (np.random.rand() - 0.5)
    # agent2.initial_conditions[c.ic_long_gc_deg] = 1.5501 + var
    # agent2.initial_conditions[c.ic_lat_geod_deg] = 43.711 + var
    env = AircraftEnv(agent, agent2, args)

    # We should reset environment each episode, also I need to understand how I
    # am meant to store transitions (meaning at step 0 should it be state, action, terminal)
    # so the state from reset
    state = env.reset()

    terminal = False
    initial_time = time.time()
    print(f"initial time: {initial_time}")
    frame_duration = env._aggressor._simulation.jsbsim_exec.get_delta_t()

    env._target._simulation.jsbsim_exec.do_trim(0)

    while not terminal:
        current_time = time.time()
        actual_elapsed_time = current_time - initial_time
        sim_lag_time = actual_elapsed_time - env._aggressor._simulation.get_sim_time()

        for _ in range(int(sim_lag_time / frame_duration)):
            action = np.array(get_joy_vals())

            step = env._timestep
            states_dict[step] = state
            actions_dict[step] = action

            state, reward, terminal, _ = env.step({"aggressor": action})
            env.render()

            rewards_dict[step] = reward
            terminals_dict[step] = terminal

            # print(state["aggressor"][2])
            # print(terminal)
            print(state["aggressor"][-1])
            # print(state["target"][-1])
            # TODO: print distance and track angle stats to make it easier to find opponent

            if terminal:
                # I don't want to waste last state so early in data collection
                step += 1
                states_dict[step] = state
                actions_dict[step] = []
                rewards_dict[step] = 0
                terminals_dict[step] = True
                break

    print(
        f"States: {len(states_dict)}, Actions:{len(actions_dict)}, Rewards: {len(rewards_dict)}, Terminals: {len(terminals_dict)}"
    )

    dump_trajectories(states_dict, actions_dict, rewards_dict, terminals_dict)
    print("Successfully saved to disk")


if __name__ == "__main__":
    sys.exit(main())
