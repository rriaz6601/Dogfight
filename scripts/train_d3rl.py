import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory

dataset = MDPDataset.load("dataset.h5")
print(len(dataset))

# encoder factory
encoder_factory = VectorEncoderFactory(
    hidden_units=[512, 512, 512, 512, 512, 512], activation="tanh"
)

# prepare algorithm
sac = d3rlpy.algos.CQL(
    actor_encoder_factory=encoder_factory,
    critic_encoder_factory=encoder_factory,
    use_gpu=True,
)

# train offline
sac.fit(dataset, n_steps=1000000, tensorboard_dir="train_offline")

sac.save_model("model_6layer_cql.pt")
