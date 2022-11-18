import random

import numpy as np


class PolicyManager:
    """Implements the mapping between agents in envs and their policies.

    We just pick a random policy from the population for every agent at the
    beginning of the episode.

    Methods of this class can potentially be overloaded to provide a more
    clever mapping, e.g. we can minimise the number of different policies per
    rollout worker thus minimising the amount of communication required.

    Parameters
    ----------

    Returns
    -------

    Attributes
    ----------
    resample_env_policy: int
        Number of episodes after which resample the policy.
    Methods
    -------
    get_policy_for_agent
    """

    def __init__(self, cfg, num_agents):
        self.rng = np.random.RandomState(seed=random.randint(0, 2**32 - 1))

        self.num_agents = num_agents
        self.num_policies = cfg.num_policies
        self.mix_policies_in_one_env = cfg.pbt_mix_policies_in_one_env

        self.resample_env_policy_every = 10  # episodes
        self.env_policies = dict()
        self.env_policy_requests = dict()

    def get_policy_for_agent(self, agent_idx: int, env_idx: int) -> int:
        """Get policy given agent and env.

        A new policy is sampled after every (agents * 10) requests.
        So every 10 episodes we can sample for a single agent environment.

        For multi-agent systems thought it would be re-sampling after more
        requests for each environment, i-e for 2 agents after 20 requests for
        the environment, but for the second agent it would be after 40 requests
        (20 episodes), TODO: verify if sampling is done at different times for
        these agents.

        Parameters
        ----------
        agent_idx :

        env_idx :

        agent_idx: int :

        env_idx: int :


        Returns
        -------


        """
        num_requests = self.env_policy_requests.get(env_idx, 0)
        if num_requests % (self.num_agents * self.resample_env_policy_every) == 0:
            if self.mix_policies_in_one_env:
                self.env_policies[env_idx] = [
                    self._sample_policy() for _ in range(self.num_agents)
                ]
            else:
                policy = self._sample_policy()
                self.env_policies[env_idx] = [policy] * self.num_agents

        self.env_policy_requests[env_idx] = num_requests + 1
        return self.env_policies[env_idx][agent_idx]

    def _sample_policy(self):
        """Random mapping currently."""
        return self.rng.randint(0, self.num_policies)
