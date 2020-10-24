from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer

from dqn.dqn_policy import DQNPolicy

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = with_common_config({
    ########################################
    # Parameters Agent
    ########################################
    "lr": 0,
    "ex_buf_len": 2000,
    "disc": 1,
    "eps": 10,
    "ex_buf_sample_size": 100,

    "dqn_model": {
        "custom_model": "?",
        "custom_model_config": {
            "layers": [
                {
                    "type": "linear",
                    "input": 4,
                    "outputs": 2
                },
                {
                    "type": "relu",
                },
                {
                    "type": "linear",
                    "input": 2,
                    "output": 2
                }
            ]
        },  # extra options to pass to your model
    }
})

DQNTrainer = build_trainer(
    name="DQNAlgorithm",
    default_policy=DQNPolicy,
    default_config=DEFAULT_CONFIG)
