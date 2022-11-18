import sys

from scripts.train_gym_env import custom_parse_args, register_custom_components
from src.async_arch.algorithms.appo.enjoy_appo import enjoy


def main():
    """Script entry point."""
    register_custom_components()
    cfg = custom_parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
