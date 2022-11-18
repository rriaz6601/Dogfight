"""Use the trained dogfight rule agent.

Run from the root directory.
python -m scripts.enjoy_dogfight_rule --algo=APPO --env=aircraft_env_v1 --experiment=rnn4

"""

import sys

from dotenv import load_dotenv

from scripts.train_dogfight_rule import custom_parse_args, register_custom_components
from src.async_arch.algorithms.appo.enjoy_appo import enjoy


def main():
    """Script entry point."""
    load_dotenv()

    register_custom_components()
    cfg = custom_parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
