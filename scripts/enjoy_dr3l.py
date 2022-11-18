"""This script to view the training of the agent trained with behaviour cloning."""

import argparse
import time

import d3rlpy
import numpy as np
from d3rlpy.models.encoders import VectorEncoderFactory
from dotenv import load_dotenv

from src.async_arch.envs.aircraft.agent_defs import AcclerationAgent, AcclerationAgent2
from src.async_arch.envs.aircraft.aircraft_gym import AircraftEnv
from src.async_arch.envs.aircraft.aircraft_params import add_aircraft_env_args
from src.async_arch.envs.aircraft.wrappers import FullStateAggregation


def load_model(environment):

    encoder_factory = VectorEncoderFactory(
        hidden_units=[512, 512, 512, 512, 512, 512], activation="tanh"
    )

    # bc = d3rlpy.algos.BC(encoder_factory=encoder_factory)
    # bc.build_with_env(environment)
    # bc.from_json("d3rlpy_logs/BC_20220927115049/params.json")
    # bc.load_model("d3rlpy_logs/BC_20220927115049/model_70000.pt")

    sac = d3rlpy.algos.SAC(
        actor_encoder_factory=encoder_factory, critic_encoder_factory=encoder_factory
    )
    sac.build_with_env(environment)
    sac.from_json("d3rlpy_logs/CQL_20220929110301/params.json")
    sac.load_model("d3rlpy_logs/CQL_20220929110301/model_240000.pt")

    return sac


def main():

    load_dotenv()

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

    env = AircraftEnv(agent, agent2, args)
    env = FullStateAggregation(env)

    # Initialise and load the algorithm
    model = load_model(env)
    state = env.reset()

    terminal = False
    initial_time = time.time()
    frame_duration = env._aggressor._simulation.jsbsim_exec.get_delta_t()

    env._target._simulation.jsbsim_exec.do_trim(0)

    while not terminal:
        current_time = time.time()
        actual_elapsed_time = current_time - initial_time
        sim_lag_time = actual_elapsed_time - env._aggressor._simulation.get_sim_time()

        for _ in range(int(sim_lag_time / frame_duration)):

            state = np.reshape(state, newshape=(1, 22))

            action = model.predict(state)
            action = np.reshape(action, (4,))
            print(action)

            state, reward, terminal, _ = env.step({"aggressor": action})
            env.render()

        if terminal:
            print(f"it took {time.time()-initial_time} seconds")
            break


if __name__ == "__main__":
    main()
