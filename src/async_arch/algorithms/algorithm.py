import os
from abc import ABC, abstractmethod
from os.path import join

from src.async_arch.utils.utils import str2bool


class AlgorithmBase(ABC):
    """ """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @classmethod
    def add_cli_args(cls, parser):
        """

        Parameters
        ----------
        parser :


        Returns
        -------

        """

        parser.add_argument(
            "--train_dir",
            default=join(os.getcwd(), "train_dir"),
            type=str,
            help="Root for all experiments",
        )
        parser.add_argument(
            "--device",
            default="gpu",
            type=str,
            choices=["gpu", "cpu"],
            help="CPU training is only recommended for smaller e.g. MLP policies",
        )

    @abstractmethod
    def initialize(self):
        """ """
        return 1

    @abstractmethod
    def run(self) -> int:
        """ """
        return 1

    @abstractmethod
    def finalize(self):
        """ """
        return 1


class ReinforcementLearningAlgorithm(AlgorithmBase):
    """Abstract class as a structure for all algorithms in this project."""

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
            "--seed", default=None, type=int, help="Set a fixed seed value"
        )

        # Checkpointing args
        parser.add_argument(
            "--save_every_sec", default=120, type=int, help="Checkpointing rate"
        )
        parser.add_argument(
            "--keep_checkpoints",
            default=3,
            type=int,
            help="Number of model checkpoints to keep",
        )
        parser.add_argument(
            "--save_milestones_sec",
            default=-1,
            type=int,
            help="Save intermediate checkpoints in a separate folder for later evaluation (default=never)",
        )

        parser.add_argument(
            "--stats_avg",
            default=100,
            type=int,
            help="How many episodes to average to measure performance (avg. reward etc)",
        )

        # Learning Rate customization
        parser.add_argument("--learning_rate", default=1e-4, type=float, help="LR")
        parser.add_argument(
            "--lr_schedule",
            default="constant",
            choices=["constant", "kl_adaptive_minibatch", "kl_adaptive_epoch"],
            type=str,
            help=(
                "Learning rate schedule to use. Constant keeps constant learning rate throughout training."
                "kl_adaptive* schedulers look at --lr_schedule_kl_threshold and if KL-divergence with behavior policy"
                "after the last minibatch/epoch significantly deviates from this threshold, lr is apropriately"
                "increased or decreased"
            ),
        )
        parser.add_argument(
            "--lr_schedule_kl_threshold",
            default=0.008,
            type=float,
            help="Used with kl_adaptive_* schedulers",
        )

        # Max training args
        parser.add_argument(
            "--train_for_env_steps",
            default=int(1e10),
            type=int,
            help="Stop after all policies are trained for this many env steps",
        )
        parser.add_argument(
            "--train_for_seconds",
            default=int(1e6),
            type=int,
            help="Stop training after this many seconds",
        )

        # Observation Preprocessing
        parser.add_argument(
            "--obs_subtract_mean",
            default=0.0,
            type=float,
            help="Observation preprocessing, mean value to subtract from observation (e.g. 128.0 for 8-bit RGB)",
        )
        parser.add_argument(
            "--obs_scale",
            default=1.0,
            type=float,
            help="Observation preprocessing, divide observation tensors by this scalar (e.g. 128.0 for 8-bit RGB)",
        )

        # RL
        parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
        parser.add_argument(
            "--reward_scale",
            default=1.0,
            type=float,
            help=(
                "Multiply all rewards by this factor before feeding into RL algorithm."
                "Sometimes the overall scale of rewards is too high which makes value estimation a harder regression task."
                "Loss values become too high which requires a smaller learning rate, etc."
            ),
        )
        parser.add_argument(
            "--reward_clip",
            default=10.0,
            type=float,
            help="Clip rewards between [-c, c]. Default [-10, 10] virtually means no clipping for most envs",
        )

        # Policy Model configurations
        parser.add_argument(
            "--encoder_type",
            default="conv",
            type=str,
            help="Type of the encoder. Supported: conv, mlp, resnet (feel free to define more)",
        )
        parser.add_argument(
            "--encoder_subtype",
            default="convnet_simple",
            type=str,
            help="Specific encoder design (see model.py)",
        )
        parser.add_argument(
            "--encoder_custom",
            default=None,
            type=str,
            help="Use custom encoder class from the registry (see model_utils.py)",
        )
        parser.add_argument(
            "--encoder_extra_fc_layers",
            default=1,
            type=int,
            help='Number of fully-connected layers of size "hidden size" to add after the basic encoder (e.g. convolutional)',
        )
        parser.add_argument(
            "--hidden_size",
            default=512,
            type=int,
            help="Size of hidden layer in the model, or the size of RNN hidden state in recurrent model (e.g. GRU)",
        )
        parser.add_argument(
            "--nonlinearity",
            default="elu",
            choices=["elu", "relu", "tanh"],
            type=str,
            help="Type of nonlinearity to use",
        )
        parser.add_argument(
            "--policy_initialization",
            default="orthogonal",
            choices=["orthogonal", "xavier_uniform", "torch_default"],
            type=str,
            help="NN weight initialization",
        )
        parser.add_argument(
            "--policy_init_gain",
            default=1.0,
            type=float,
            help="Gain parameter of PyTorch initialization schemas (i.e. Xavier)",
        )
        parser.add_argument(
            "--actor_critic_share_weights",
            default=True,
            type=str2bool,
            help="Whether to share the weights between policy and value function",
        )

        # TODO: Right now this only applies to custom encoders. Make sure generic policies also factor in this arg
        parser.add_argument(
            "--use_spectral_norm",
            default=False,
            type=str2bool,
            help="Use spectral normalization to smoothen the gradients and stabilize training. Only supports fully connected layers",
        )

        # TODO: Need to understand what does this mean
        parser.add_argument(
            "--adaptive_stddev",
            default=True,
            type=str2bool,
            help="Only for continuous action distributions, whether stddev is state-dependent or just a single learned parameter",
        )
        parser.add_argument(
            "--initial_stddev",
            default=1.0,
            type=float,
            help="Initial value for non-adaptive stddev. Only makes sense for continuous action spaces",
        )
