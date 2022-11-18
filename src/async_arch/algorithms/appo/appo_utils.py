import threading
from collections import OrderedDict, deque
from typing import Dict

import numpy as np
import torch

from src.async_arch.utils.utils import log, memory_consumption_mb

CUDA_ENVVAR = "CUDA_VISIBLE_DEVICES"


class TaskType:
    """ """

    (
        INIT,
        TERMINATE,
        RESET,
        ROLLOUT_STEP,
        POLICY_STEP,
        TRAIN,
        INIT_MODEL,
        PBT,
        UPDATE_ENV_STEPS,
        EMPTY,
    ) = range(10)


def iterate_recursively(d):
    """Generator for a dictionary that can potentially include other dictionaries.
    Yields tuples of (dict, key, value), where key, value are "leaf" elements of the "dict".

    Parameters
    ----------
    d :


    Returns
    -------


    """
    for k, v in d.items():
        if isinstance(v, (dict, OrderedDict)):
            yield from iterate_recursively(v)
        else:
            yield d, k, v


def copy_dict_structure(d):
    """Copy dictionary layout without copying the actual values (populated with Nones).

    Parameters
    ----------
    d :


    Returns
    -------


    """
    d_copy = type(d)()
    _copy_dict_structure_func(d, d_copy)
    return d_copy


def _copy_dict_structure_func(d, d_copy):
    """

    Parameters
    ----------
    d :

    d_copy :


    Returns
    -------


    """
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            d_copy[key] = type(value)()
            _copy_dict_structure_func(value, d_copy[key])
        else:
            d_copy[key] = None


def iter_dicts_recursively(d1: Dict, d2: Dict):
    """Iterate through two dictionaries where both contain all keys in `d1`.

    Assuming structure of d1 is strictly included into d2. I.e. each key at
    each recursion level is also present in d2. This is also true when d1 and
    d2 have the same structure.

    Parameters
    ----------
    d1 :

    d2 :

    d1: Dict :

    d2: Dict :


    Returns
    -------


    """
    for k, v in d1.items():
        assert k in d2

        if isinstance(v, (dict, OrderedDict)):
            yield from iter_dicts_recursively(d1[k], d2[k])
        else:
            yield d1, d2, k, d1[k], d2[k]


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """

    Parameters
    ----------
    list_of_dicts :


    Returns
    -------


    """
    dict_of_lists = dict()

    for d in list_of_dicts:
        for key, x in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []

            dict_of_lists[key].append(x)

    return dict_of_lists


def extend_array_by(x, extra_len):
    """Assuming the array is currently not empty.

    Parameters
    ----------
    x :

    extra_len :


    Returns
    -------


    """
    if extra_len <= 0:
        return x

    last_elem = x[-1]
    tail = [last_elem] * extra_len
    tail = np.stack(tail)
    return np.append(x, tail, axis=0)


def memory_stats(process, device):
    """

    Parameters
    ----------
    process :

    device :


    Returns
    -------


    """
    memory_mb = memory_consumption_mb()
    stats = {f"memory_{process}": memory_mb}
    if device.type != "cpu":
        gpu_mem_mb = torch.cuda.memory_allocated(device) / 1e6
        gpu_cache_mb = torch.cuda.memory_reserved(device) / 1e6
        stats.update(
            {f"gpu_mem_{process}": gpu_mem_mb, f"gpu_cache_{process}": gpu_cache_mb}
        )

    return stats


def tensor_batch_size(tensor_batch):
    """

    Parameters
    ----------
    tensor_batch :


    Returns
    -------


    """
    for _, _, v in iterate_recursively(tensor_batch):
        return v.shape[0]


class TensorBatcher:
    """ """

    def __init__(self, batch_pool):
        self.batch_pool = batch_pool

    def cat(self, dict_of_arrays, macro_batch_size, use_pinned_memory, timing):
        """Here 'macro_batch' is the overall size of experience per iteration.
        Macro-batch = mini-batch * num_batches_per_iteration

        Parameters
        ----------
        dict_of_arrays :

        macro_batch_size :

        use_pinned_memory :

        timing :


        Returns
        -------


        """

        tensor_batch = self.batch_pool.get()

        if tensor_batch is not None:
            old_batch_size = tensor_batch_size(tensor_batch)
            if old_batch_size != macro_batch_size:
                # this can happen due to PBT changing batch size during the experiment
                log.warning(
                    "Tensor macro-batch size changed from %d to %d!",
                    old_batch_size,
                    macro_batch_size,
                )
                log.warning("Discarding the cached tensor batch!")
                del tensor_batch
                tensor_batch = None

        if tensor_batch is None:
            tensor_batch = copy_dict_structure(dict_of_arrays)
            log.info("Allocating new CPU tensor batch (could not get from the pool)")

            for _, cache_d, key, arr, _ in iter_dicts_recursively(
                dict_of_arrays, tensor_batch
            ):
                cache_d[key] = torch.from_numpy(arr)
                if use_pinned_memory:
                    cache_d[key] = cache_d[key].pin_memory()
        else:
            with timing.add_time("batcher_mem"):
                # this is slower than the older version where we copied trajectories to the cached tensor one-by-one
                # this will only affect CPU-based envs with pixel observations, and won't matter after
                # Sample Factory 2.0 is released. Tradeoff is that the batching code is a lot more simple and
                # easier to modify.
                for _, cache_d, key, arr, cache_t in iter_dicts_recursively(
                    dict_of_arrays, tensor_batch
                ):
                    t = torch.as_tensor(arr)
                    cache_t.copy_(t)

        return tensor_batch


class ObjectPool:
    """ """

    def __init__(self, pool_size=10):
        self.pool_size = pool_size
        self.pool = deque([], maxlen=self.pool_size)
        self.lock = threading.Lock()

    def get(self):
        """ """
        with self.lock:
            if len(self.pool) <= 0:
                return None

            obj = self.pool.pop()
            return obj

    def put(self, obj):
        """

        Parameters
        ----------
        obj :


        Returns
        -------


        """
        with self.lock:
            self.pool.append(obj)

    def clear(self):
        """ """
        with self.lock:
            self.pool = deque([], maxlen=self.pool_size)
