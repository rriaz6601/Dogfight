"""Offline dataset from pickle files."""

import pickle

import numpy as np
from d3rlpy.dataset import MDPDataset
from navpy import angle2dcm

from src.async_arch.envs.aircraft.aircraft_utils import calculate_relative_position


def state_aggregation(observation):

    # Relative position and velocity
    ned_rel_pos = calculate_relative_position(
        observation["target"], observation["aggressor"]
    )
    target_rot_mat = angle2dcm(
        observation["target"][5],
        observation["target"][3],
        observation["target"][4],
        input_unit="deg",
    )
    target_vel_ned = np.dot(target_rot_mat, observation["target"][6:9])

    aggressor_rot_mat = angle2dcm(
        observation["aggressor"][5],
        observation["aggressor"][3],
        observation["aggressor"][4],
        input_unit="deg",
    )
    aggressor_vel_ned = np.dot(aggressor_rot_mat, observation["aggressor"][6:9])
    rel_velo = aggressor_vel_ned - target_vel_ned - ned_rel_pos

    obs_deltas = []

    obs_deltas[:3] = ned_rel_pos[:]  # Relative position
    obs_deltas[3:6] = rel_velo[:]  # relative velocity
    obs_deltas[6:9] = observation["aggressor"][3:6]  # own euler angles
    obs_deltas[9:18] = observation["aggressor"][
        9:18
    ]  # own rot vel, and 6 accelerations
    obs_deltas[18:21] = observation["aggressor"][
        23:26
    ]  # own alpha and beta and opp health
    obs_deltas.append(1.0)  #  my health

    assert len(obs_deltas) == 22

    # Normalising the observations
    # Subtract the middle then divide by (high - low) /2
    # TODO: There should be a better way to do this normalisation
    obs_deltas = np.array(obs_deltas)
    middle = np.array(
        [
            1500,
            1500,
            1500,
            700,
            700,
            700,
            0,
            0,
            180,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.5,
            0.5,
        ]
    )
    difference = np.array(
        [
            1500,
            1500,
            1500,
            700,
            700,
            700,
            90,
            90,
            180,
            2 * np.pi,
            2 * np.pi,
            2 * np.pi,
            4,
            4,
            4,
            (8 / 180) * np.pi,
            (8 / 180) * np.pi,
            (8 / 180) * np.pi,
            90,
            180,
            0.5,
            0.5,
        ]
    )

    obs_deltas -= middle
    normalised_obs = np.divide(obs_deltas, difference)

    return normalised_obs


if __name__ == "__main__":

    observations = []
    actions = []
    rewards = []
    terminals = []

    for eps in range(76):
        with open(f"data/eps_{eps}", "rb") as input_file:
            data = pickle.load(input_file)

        observations_list = []
        actions_list = []
        rewards_list = []
        terminal_list = []

        # print(data[0][0])
        # obs = state_aggregation(data[0][0])
        # print(obs)

        last = sorted(data.keys())[-1]
        del data[last]

        for t in data.keys():
            observations_list.append(state_aggregation(data[t][0]))
            actions_list.append(data[t][1])
            rewards_list.append(data[t][2])
            terminal_list.append(data[t][3])

        assert (
            len(observations_list)
            == len(actions_list)
            == len(rewards_list)
            == len(terminal_list)
        )

        observations.extend(observations_list)
        actions.extend(actions_list)
        rewards.extend(rewards_list)
        terminals.extend(terminal_list)

    print(len(observations))
    print(observations[:3])

    observations = np.array(observations)
    actions = np.array(actions)
    print(observations.shape)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    # automatically splitted into d3rlpy.dataset.Episode objects
    dataset.episodes

    # each episode is also splitted into d3rlpy.dataset.Transition objects
    episode = dataset.episodes[0]
    episode[0].observation
    episode[0].action
    episode[0].reward
    episode[0].next_observation
    episode[0].terminal

    # d3rlpy.dataset.Transition object has pointers to previous and next
    # transitions like linked list.
    transition = episode[0]
    while transition.next_transition:
        transition = transition.next_transition

    # save as HDF5
    dataset.dump("dataset.h5")

    # load from HDF5
    # new_dataset = MDPDataset.load("dataset.h5")
