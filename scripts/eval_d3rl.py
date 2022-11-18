from d3rlpy.algos import SAC
from d3rlpy.dataset import MDPDataset
from d3rlpy.ope import FQE

dataset = MDPDataset.load("dataset.h5")

# prepare algorithm
sac = SAC()
sac.build_with_dataset(dataset)
sac.from_json("d3rlpy_logs/SAC_20220925205545/params.json")
sac.load_model("d3rlpy_logs/SAC_20220925205545/model_990000.pt")

fqe = FQE(algo=sac)

# metrics to evaluate with
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer, soft_opc_scorer

# train estimators to evaluate the trained policy
fqe.fit(
    dataset.episodes,
    n_epochs=1,
    eval_episodes=dataset.episodes,
    scorers={
        "init_value": initial_state_value_estimation_scorer,
        "soft_opc": soft_opc_scorer(return_threshold=600),
    },
)
