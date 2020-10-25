import ray
import os
from ray import tune
from ray.rllib.models import ModelCatalog

from dqn import DQNTrainer, DQNModel

if __name__ == "__main__":
    ray.init(num_cpus=2)
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    tune.run(
        DQNTrainer,
        # checkpoint_freq=10,
        checkpoint_at_end=True,
        stop={"episodes_total": 10000},
        config={
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            "rollout_fragment_length": 50,
            "env": "CartPole-v1",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 0.0005,  # tune.grid_search([0.0005, 0.001]),
            "ex_buf_len": 10000,
            "disc": 0.95,
            "eps": 0.5,
            "ex_buf_sample_size": 2000,

            "dqn_model": {
                "custom_model": "DQNModel",
                "custom_model_config": {
                    "layers": [
                        {
                            "type": "linear",
                            "input": 4,
                            "output": 32
                        },
                        {
                            "type": "relu",
                        },
                        {
                            "type": "linear",
                            "input": 32,
                            "output": 64
                        },
                        {
                            "type": "relu",
                        },
                        {
                            "type": "linear",
                            "input": 64,
                            "output": 32
                        },
                        {
                            "type": "relu",
                        },
                        {
                            "type": "linear",
                            "input": 32,
                            "output": 2
                        }
                    ]
                },
            }
        }
    )
