from torch import nn, cat
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gym.spaces import Discrete, Box


class DQNModel(nn.Module, TorchModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.obs_space = obs_space
        self.action_space = action_space
        self.model_config = model_config
        self.name = name

        if isinstance(self.obs_space, Box):
            self.obs_shape = obs_space.shape[0]
        else:
            self.obs_shape = self.obs_space

        self.layers = nn.Sequential()

        conf_layers = model_config["custom_model_config"]["layers"]
        lin_cnt = 0
        relu_cnt = 0
        l_relu_cnt = 0

        for layer in conf_layers:
            l_type = layer["type"]
            if l_type == "linear":
                name = "linear_" + str(lin_cnt)
                lin_cnt += 1
                self.layers.add_module(name, nn.Linear(layer["input"], layer["output"]))
            elif l_type == "relu":
                name = "relu_" + str(relu_cnt)
                relu_cnt += 1
                self.layers.add_module(name, nn.ReLU())
            elif l_type == "l_relu":
                name = "leaky_relu_" + str(l_relu_cnt)
                l_relu_cnt += 1
                self.layers.add_module(name, nn.LeakyReLU())
            else:
                print("ERR: couldn't find layer named: " + l_type)

    @override(TorchModelV2)
    def forward(self, obs):
        return self.layers(obs)
