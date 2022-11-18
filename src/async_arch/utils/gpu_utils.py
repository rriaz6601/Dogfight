"""These utilities are for gpu/cuda, it can also work as a script to get total available devices"""

import os
import sys

import torch

from src.async_arch.utils.utils import log

CUDA_ENVVAR = "CUDA_VISIBLE_DEVICES"


def get_gpus_without_triggering_pytorch_cuda_initialization(envvars=None):
    """

    Parameters
    ----------
    envvars :
         (Default value = None)

    Returns
    -------

    """
    if envvars is None:
        envvars = os.environ

    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "src.async_arch.utils.gpu_utils"],
        capture_output=True,
        env=envvars,
    )
    text_output = out.stdout.decode()
    err_output = out.stderr.decode()
    returncode = out.returncode

    from src.async_arch.utils.utils import log

    if returncode:
        log.error(
            "Querying available GPUs... return code %d, error: %s, stdout: %s",
            returncode,
            err_output,
            text_output,
        )

    log.debug("Queried available GPUs: %s", text_output)
    return text_output


def set_global_cuda_envvars(cfg):
    """Set CUDA environment variables.

    Parameters
    ----------
    cfg :


    Returns
    -------


    """
    if cfg.device == "cpu":
        available_gpus = ""
    else:
        available_gpus = get_gpus_without_triggering_pytorch_cuda_initialization(
            os.environ
        )

    if CUDA_ENVVAR not in os.environ:
        os.environ[CUDA_ENVVAR] = available_gpus
    os.environ[f"{CUDA_ENVVAR}_backup_"] = os.environ[CUDA_ENVVAR]
    os.environ[CUDA_ENVVAR] = ""


def get_available_gpus():
    """ """
    orig_visible_devices = os.environ[f"{CUDA_ENVVAR}_backup_"]
    available_gpus = [int(g) for g in orig_visible_devices.split(",") if g]
    return available_gpus


def set_gpus_for_process(
    process_idx, num_gpus_per_process, process_type, gpu_mask=None
):
    """Set gpu for current process.

    Parameters
    ----------
    process_idx :

    num_gpus_per_process :

    process_type :

    gpu_mask :
         (Default value = None)

    Returns
    -------

    """
    available_gpus = get_available_gpus()
    if gpu_mask is not None:
        assert len(available_gpus) >= len(available_gpus)
        available_gpus = [available_gpus[g] for g in gpu_mask]
    num_gpus = len(available_gpus)
    gpus_to_use = []

    if num_gpus == 0:
        os.environ[CUDA_ENVVAR] = ""
        log.debug("Not using GPUs for %s process %d", process_type, process_idx)
    else:
        first_gpu_idx = process_idx * num_gpus_per_process
        for i in range(num_gpus_per_process):
            index_mod_num_gpus = (first_gpu_idx + i) % num_gpus
            gpus_to_use.append(available_gpus[index_mod_num_gpus])

        os.environ[CUDA_ENVVAR] = ",".join([str(g) for g in gpus_to_use])
        log.info(
            "Set environment var %s to %r for %s process %d",
            CUDA_ENVVAR,
            os.environ[CUDA_ENVVAR],
            process_type,
            process_idx,
        )
        log.debug("Visible devices: %r", torch.cuda.device_count())

    return gpus_to_use


def cuda_envvars_for_policy(policy_id, process_type):
    """

    Parameters
    ----------
    policy_id :

    process_type :


    Returns
    -------

    """
    set_gpus_for_process(policy_id, 1, process_type)


def main():
    """This function prints the index of available gpus so if it prints 0 it
    means there is 1 gpu numbered 0

    Parameters
    ----------

    Returns
    -------
    int
        Just a 0 int to show correct execution

    """
    import torch

    device_count = torch.cuda.device_count()
    available_gpus = ",".join(str(g) for g in range(device_count))
    print(available_gpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
