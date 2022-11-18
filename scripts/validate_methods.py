"""
Script for validating common methods.

Run this script: python -m scripts.validate_methods

The method for testing.
1. Initiate the environment.
2. Do an outer loop for environment.
3. Do an inner loop for timestep.
4. Add asserts for confirming the calculations are correct.
"""

import sys
import time

from src.async_arch.envs.aircraft.env_rule import EnvRuleBasedOpponent

DEFAULT_ACT = [0.0, -0.5, 0.0, 0.8]


def main():
    """Run 10 episodes."""
    print("Started the function.")
    # env = EnvBothRule()
    env = EnvRuleBasedOpponent()
    rews = []

    for out_l in range(10):
        state = env.reset()
        print("Environment reset")

        eps_rew = 0.0
        for in_l in range(1501):
            state, reward, done, _ = env.step(DEFAULT_ACT)
            # print(state)
            time.sleep(0.01)
            eps_rew += reward

            if done:
                print(f"Episode ending at timestep: {env.timestep}, reward: {eps_rew}")
                rews.append(eps_rew)
                break

    print(rews)


if __name__ == "__main__":
    print("Hello!")
    sys.exit(main())
    print("Hello There")
