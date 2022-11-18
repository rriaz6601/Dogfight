import sys

from src.async_arch.algorithms.utils.arguments import (
    get_algo_class,
    maybe_load_from_checkpoint,
    parse_args,
)


def run_algorithm(cfg) -> int:
    """

    Parameters
    ----------
    cfg :


    Returns
    -------

    """
    cfg = maybe_load_from_checkpoint(cfg)

    algo = get_algo_class(cfg.algo)(cfg)
    algo.initialize()
    status = algo.run()
    algo.finalize()
    return status


def main():
    """Main function, it is this script's entry point"""

    cfg = parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
