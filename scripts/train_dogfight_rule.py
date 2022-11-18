"""
First training script use this to run an experiment with the rule based agent
Run using the following command from the root of the directory:
python -m scripts.train_dogfight_rule --algo=APPO --with_wandb=True --env=aircraft_env_v1 --experiment=Env2 --save_milestones_sec=1200

For debugging purposes just do:
python -m scripts.train_dogfight_rule  --algo=APPO --env=aircraft_env_v1 --experiment=Debug1 --train_in_background_thread=False --num_workers=1 --worker_num_splits=1
"""

import sys

from dotenv import load_dotenv

from src.async_arch.algorithms.utils.arguments import arg_parser, parse_args
from src.async_arch.envs.aircraft.aircraft_params import (
    add_aircraft_env_args,
    aircraft_override_defaults,
    make_aircraft_env,
)
from src.async_arch.envs.env_registry import global_env_registry
from src.async_arch.run_algorithm import run_algorithm


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix="aircraft_env_",
        make_env_func=make_aircraft_env,
        add_extra_params_func=add_aircraft_env_args,
        override_default_params_func=aircraft_override_defaults,
    )


def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.

    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for
    evaluating the policy (see the enjoy_ script).

    """
    parser = arg_parser(argv, evaluation=evaluation)

    # add custom args here
    parser.add_argument(
        "--my_custom_arg",
        type=int,
        default=42,
        help="Any custom arguments users might define",
    )

    # parse_args function does some additional processing (see comments there)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    """Script entry point."""
    load_dotenv()

    register_custom_components()
    cfg = custom_parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
