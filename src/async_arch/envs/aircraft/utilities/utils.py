import csv
import glob
import os
from typing import Optional

import gym


def share_among_lists(e, sh):
    """

    Parameters
    ----------
    e :

    sh :


    Returns
    -------

    """
    n = len(e)
    for i in range(n):
        k = i + 1 if i < (n - 1) else 0
        m = len(e[i])
        for j in range(n - 1, 1, -1):
            e[i][m - (sh * j) : m - (sh * (j - 1))] = e[k][:sh]
            k = k + 1 if k < (n - 1) else 0
        e[i][-sh:] = e[k][:sh]

    return e


def get_latest_run_id(log_path: Optional[str] = None, log_name: str = "") -> int:
    """Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :return: latest run number

    Parameters
    ----------
    log_path: Optional[str] :
         (Default value = None)
    log_name: str :
         (Default value = "")

    Returns
    -------

    """
    max_run_id = 0
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


class RecordObservationWrapper(gym.Wrapper):
    """ """

    def __init__(
        self, env, save_freq=10, logs_folder="logs/", eps_log_name="run", filename="eps"
    ):
        super().__init__(env)
        self.run_id = 0
        self.save_freq = save_freq
        self.filename = filename
        self._elapsed_episodes = 0
        self.run_id = get_latest_run_id(logs_folder, eps_log_name) + 1
        self.save_path = f"{logs_folder}/{eps_log_name}_{self.run_id}"
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def step(self, action):
        """

        Parameters
        ----------
        action :


        Returns
        -------

        """
        observation, reward, done, info = self.env.step(action)
        if done:
            self._elapsed_episodes += 1

        if self._elapsed_episodes % self.save_freq == 0:
            path = os.path.join(
                self.save_path, f"{self.filename}_{self._elapsed_episodes}"
            )

            with open(path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(observation)

        return observation, reward, done, info
