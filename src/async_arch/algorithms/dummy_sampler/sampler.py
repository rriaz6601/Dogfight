"""
    This fake algorithm finds the upper bound of the performance for an
    RL algorithm, because it just samples the environment.
"""

import ctypes
import json
import multiprocessing
import os
import signal
import time
from collections import deque
from multiprocessing.sharedctypes import RawValue
from sys import platform

import numpy as np
import psutil
from _queue import Empty

from src.async_arch.algorithms.algorithm import AlgorithmBase
from src.async_arch.envs.create_env import create_env
from src.async_arch.utils.env_utils import make_env_func
from src.async_arch.utils.gpu_utils import set_global_cuda_envvars, set_gpus_for_process
from src.async_arch.utils.timing import Timing
from src.async_arch.utils.utils import (
    AttrDict,
    cfg_file,
    experiment_dir,
    init_file_logger,
    log,
    memory_consumption_mb,
    set_process_cpu_affinity,
    str2bool,
)

if os.name == "nt":
    pass  # TODO: The coded queue in the 'nt' system
else:
    # faster_fifo queue is a faster replacement of multiprocessing.Queue()
    # https://github.com/alex-petrenko/faster-fifo
    from faster_fifo import Queue as MpQueue


class DummySampler(AlgorithmBase):
    """ """

    @classmethod
    def add_cli_args(cls, parser):
        """

        Parameters
        ----------
        parser :


        Returns
        -------

        """
        super().add_cli_args(parser)

        parser.add_argument(
            "--num_workers",
            default=multiprocessing.cpu_count(),
            type=int,
            help="Number of processes to use to sample the environment.",
        )
        parser.add_argument(
            "--num_envs_per_worker",
            default=1,
            type=int,
            help="Number of envs on a single CPU sampled sequentially.",
        )

        parser.add_argument(
            "--sample_env_frames",
            default=int(2e9),
            type=int,
            help="Stop after sampling this many env frames (this takes frameskip into account)",
        )
        parser.add_argument(
            "--sample_env_frames_per_worker",
            default=int(1e9),
            type=int,
            help="Stop after sampling this many env frames per worker (this takes frameskip into account)",
        )
        parser.add_argument(
            "--timeout_seconds",
            default=600,
            type=int,
            help="Stop after sampling for this many seconds",
        )

        parser.add_argument(
            "--set_workers_cpu_affinity",
            default=True,
            type=str2bool,
            help="Whether to assign workers to specific CPU cores or not. The \
                logic is beneficial for most workloads because prevents a lot \
                of context switching.However for some environments it can be \
                better to disable it, to allow one worker to use all cores \
                some of the time. This can be the case for some DMLab \
                environments with very expensive episode reset that can use \
                parallel CPU cores for level generation.",
        )

        parser.add_argument(
            "--sampler_worker_gpus",
            default=[],
            type=int,
            nargs="*",
            help="By default, workers only use CPUs. Changes this if e.g. you need GPU-based rendering on the actors",
        )

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        set_global_cuda_envvars(self.cfg)

        self.processes = []
        self.terminate = RawValue(ctypes.c_bool, False)

        self.start_event = multiprocessing.Event()
        self.start_event.clear()

        self.report_queue = MpQueue()
        self.report_every_sec = 1.0
        self.last_report = 0

        self.avg_stats_intervals = (1, 60, 300, 600)
        self.fps_stats = deque([], maxlen=max(self.avg_stats_intervals))

    def _cfg_dict(self):
        """ """
        if isinstance(self.cfg, dict):
            return self.cfg
        else:
            return vars(self.cfg)

    def _save_cfg(self):
        """ """
        cfg_dict = self._cfg_dict()
        with open(cfg_file(self.cfg), "w") as json_file:
            json.dump(cfg_dict, json_file, indent=2)

    def initialize(self):
        """ """
        # creating an environment in the main process tends to fix some very weird issues further down the line
        # https://stackoverflow.com/questions/60963839/importing-opencv-after-importing-pytorch-messes-with-cpu-affinity
        # do not delete this unless you know what you're doing
        tmp_env = create_env(self.cfg.env, cfg=self.cfg, env_config=None)
        tmp_env.close()

        self._save_cfg()
        # save_git_diff(experiment_dir(cfg=self.cfg))
        init_file_logger(experiment_dir(self.cfg))

        for i in range(self.cfg.num_workers):
            p = multiprocessing.Process(target=self.sample, args=(i,))
            self.processes.append(p)

    def sample(self, proc_idx):
        """

        Parameters
        ----------
        proc_idx :


        Returns
        -------

        """
        # Workers should ignore Ctrl+C because the termination is handled in the event loop
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if self.cfg.sampler_worker_gpus:
            set_gpus_for_process(
                proc_idx,
                num_gpus_per_process=1,
                process_type="sampler_proc",
                gpu_mask=self.cfg.sampler_worker_gpus,
            )

        timing = Timing()

        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api=None):
            if self.cfg.set_workers_cpu_affinity:
                set_process_cpu_affinity(proc_idx, self.cfg.num_workers)

            initial_cpu_affinity = (
                psutil.Process().cpu_affinity()
                if platform != "darwin"
                else "MacOS - None"
            )
            psutil.Process().nice(10)

            with timing.timeit("env_init"):
                envs = []
                env_key = ["env" for _ in range(self.cfg.num_envs_per_worker)]

                for env_idx in range(self.cfg.num_envs_per_worker):
                    global_env_id = proc_idx * self.cfg.num_envs_per_worker + env_idx
                    env_config = AttrDict(
                        worker_index=proc_idx, vector_idx=env_idx, env_id=global_env_id
                    )

                    env = make_env_func(cfg=self.cfg, env_config=env_config)
                    log.debug(
                        "CPU affinity after create env: %r",
                        psutil.Process().cpu_affinity()
                        if platform != "darwin"
                        else "MacOS - None",
                    )
                    env.seed(global_env_id)  # TODO: seed() method deprecated for gymEnv
                    envs.append(env)

                    # this is to track the performance of individual DMLab levels TODO: get rid of
                    if hasattr(env.unwrapped, "level_name"):
                        env_key[env_idx] = env.unwrapped.level_name

                episode_length = [0 for _ in envs]
                episode_lengths = [deque([], maxlen=20) for _ in envs]

            # sample a lot of random actions at once otherwise it is pretty slow in Python
            total_random_actions = 500
            actions = [
                [env.action_space.sample() for _ in range(env.num_agents)]
                for _ in range(total_random_actions)
            ]
            action_i = 0

            try:
                with timing.timeit("first_rest"):
                    for env_idx, env in enumerate(envs):
                        env.reset()
                        log.info(
                            "Process %d finished resetting %d/%d envs",
                            proc_idx,
                            env_idx + 1,
                            len(envs),
                        )

                    self.report_queue.put(dict(proc_idx=proc_idx, finished_reset=True))

                self.start_event.wait()

                with timing.timeit("work"):
                    last_report = last_report_frames = total_env_frames = 0
                    while (
                        not self.terminate.value
                        and total_env_frames < self.cfg.sample_env_frames_per_worker
                    ):
                        for env_idx, env in enumerate(envs):
                            with timing.add_time(f"{env_key[env_idx]}.step"):
                                _, _, dones, infos = env.step(actions[action_i])
                                action_i = (action_i + 1) % total_random_actions

                            num_frames = sum(
                                [info.get("num_frames", 1) for info in infos]
                            )  # dict.get(key, default) if key not in dict default
                            total_env_frames += num_frames
                            episode_length[env_idx] += num_frames

                            if all(dones):
                                episode_lengths[env_idx].append(
                                    episode_length[env_idx] / env.num_agents
                                )
                                episode_length[env_idx]

                        with timing.add_time("report"):
                            now = time.time()
                            if now - last_report > self.report_every_sec:
                                last_report = now
                                frames_since_last_report = (
                                    total_env_frames - last_report_frames
                                )
                                last_report_frames = total_env_frames
                                self.report_queue.put(
                                    dict(
                                        proc_idx=proc_idx,
                                        env_frames=frames_since_last_report,
                                    )
                                )

                                if proc_idx == 0:
                                    log.debug(
                                        "Memory usage: %.4f Mb", memory_consumption_mb()
                                    )

                # Extra check to make sure cpu affinity is preserved throughout the execution.
                # I observed weird effect when some environments tried to alter affinity of the current process, leading
                # to decreased performance.
                # This can be caused by some interactions between deep learning libs, OpenCV, MKL, OpenMP, etc.
                # At least user should know about it if this is happening.
                cpu_affinity = (
                    psutil.Process().cpu_affinity() if platform != "darwin" else None
                )
                assert initial_cpu_affinity == cpu_affinity, (
                    f"Worker CPU affinity was changed from {initial_cpu_affinity} to {cpu_affinity}!"
                    f"This can significantly affect performance!"
                )

            except:
                log.exception("Unknown exception")
                log.error("Unknown exception in worker %d, terminating...", proc_idx)
                self.report_queue.put(dict(proc_idx=proc_idx, crash=True))

            time.sleep(proc_idx * 0.01 + 0.01)
            log.info("Process %d finished sampling. Timing: %s", proc_idx, timing)

            for env_idx, env in enumerate(envs):
                if len(episode_lengths[env_idx]) > 0:
                    log.warning(
                        "Level %s avg episode len %d",
                        env_key[env_idx],
                        np.mean(episode_lengths[env_idx]),
                    )

            for env in envs:
                env.close()

    def report(self, env_frames):
        """Reports the frames and avg FPS to logger

        Parameters
        ----------
        env_frames : _type_
            _description_

        Returns
        -------

        """
        now = time.time()
        self.last_report = now

        self.fps_stats.append((now, env_frames))
        if len(self.fps_stats) <= 1:
            return

        fps = []
        for avg_interval in self.avg_stats_intervals:
            past_moment, past_frames = self.fps_stats[
                max(0, len(self.fps_stats) - 1 - avg_interval)
            ]
            fps.append((env_frames - past_frames) / (now - past_moment))

        fps_str = []
        for interval, fps_value in zip(self.avg_stats_intervals, fps):
            fps_str.append(
                f"{int(interval * self.report_every_sec)} sec: {fps_value:.1f}"
            )
        fps_str = f'({", ".join(fps_str)})'
        log.info("Sampling FPS: %s. Total frames collected: %d", fps_str, env_frames)

    def run(self):
        """The main run loop. The following things in order:
        1. Starts the already initialized(defined) processes
        2. Makes sure all the environments are reset
        3. Then allows the workers to sample.
        4. Ends the sampling based on 6 conditions:
            (i) collected required env_frames (ii) max time reached (iii) KeyboardInterrupt
            (iv) any worker crashes  (v) any process frozen (5mins) (vi) any process died unexpectedly
        5. Prints final logs (frames per second)

        Parameters
        ----------

        Returns
        -------

        """
        for p in self.processes:
            time.sleep(0.3)
            p.start()

        finished_reset = np.zeros([self.cfg.num_workers], dtype=bool)
        while not all(finished_reset):
            try:
                msg = self.report_queue.get(timeout=0.1)
                if "finished_reset" in msg:
                    finished_reset[msg["proc_idx"]] = True
                    log.debug(
                        "Process %d finished reset! Status %r",
                        msg["proc_idx"],
                        finished_reset,
                    )

            except Empty:
                pass

        log.debug("All workers finished reset!")
        time.sleep(2)
        self.start_event.set()  # Tells all the child processes that it is time to start working (resets are done)

        start = time.time()
        env_frames = 0
        last_process_report = [time.time() for _ in self.processes]

        while not self.terminate.value:
            try:
                try:
                    msgs = self.report_queue.get_many(
                        timeout=self.report_every_sec * 1.5
                    )
                    for msg in msgs:
                        last_process_report[msg["proc_idx"]] = time.time()

                        if "crash" in msg:
                            self.terminate.value = True
                            log.error(
                                "Terminating due to process %d crashing...",
                                msg["proc_idx"],
                            )
                            break

                        env_frames += msg["env_frames"]

                    if env_frames >= self.cfg.sample_env_frames:
                        log.warning("Desired number of frames reached")
                        self.terminate.value = True

                    if time.time() - start > self.cfg.timeout_seconds:
                        log.warning("Terminated by timer")
                        self.terminate.value = True

                except Empty:
                    pass

            except KeyboardInterrupt:
                self.terminate.value = True
                log.error("Keyboard interrupt in main loop! Terminating...")
                break

            if time.time() - self.last_report > self.report_every_sec:
                self.report(env_frames)

            for proc_idx, p in enumerate(self.processes):
                delay = time.time() - last_process_report[proc_idx]
                if delay > 600:
                    # killing the script is the best way to know that some of the processes froze
                    log.error(
                        "Process %d had not responded in %.1f s!!! Terminating...",
                        proc_idx,
                        delay,
                    )
                    self.terminate.value = True

            for p in self.processes:
                if not p.is_alive():
                    self.terminate.value = True
                    log.error("Process %r died! terminating...", p)

        total_time = time.time() - start
        log.info(
            "Collected %d frames in %.1f s, avg FPS: %.1f",
            env_frames,
            total_time,
            env_frames / total_time,
        )
        log.debug("Done sampling...")

    def finalize(self) -> None:
        """This function does the final clean up of the processes"""
        try:
            self.report_queue.get_many_nowait()
        except Empty:
            pass

        log.debug("Joining worker processes")
        for p in self.processes:
            p.join()
        log.debug("Done joining!")
