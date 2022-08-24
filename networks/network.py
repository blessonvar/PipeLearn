import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, config, model_type="server"):
        """
        Build a neural network
        :param config: network config
        :param model_type: "complete": complete model; "server": server-side model; "client": client-side model.
        """
        super(Network, self).__init__()
        self.model_config = config.MODEL_CONFIG[config.MODEL_NAME]
        assert config.LAYER_NUM_ON_CLIENT <= len(self.model_config), \
            "Layer number on client is larger than total layer number."
        # By default all layers are deployed on clients.
        self.layer_num_on_client = len(self.model_config) if config.LAYER_NUM_ON_CLIENT == -1 \
            else config.LAYER_NUM_ON_CLIENT
        self.model_type = model_type
        # Make layers.
        self.convolutional_layers, self.dense_layers = self._make_layers()
        self._initialize_weights()

    def forward(self, x):
        if len(self.convolutional_layers) > 0:
            x = self.convolutional_layers(x)
        if len(self.dense_layers) > 0:
            x = torch.flatten(x, start_dim=1)
            x = self.dense_layers(x)
        return x

    def _make_layers(self):
        if self.model_type == "server":
            model_config = self.model_config[self.layer_num_on_client:]
        elif self.model_type == "client":
            model_config = self.model_config[:self.layer_num_on_client]
        elif self.model_type == "complete":
            model_config = self.model_config
        else:
            raise Exception('Parameter "type" can only be one of "complete", "server" and "client".')
        convolutional_layers = []
        dense_layers = []
        for layer_config in model_config:
            layers = layer_config.make_layers()
            if layer_config.layer_type != 'Dense':
                convolutional_layers.append(layers)
            else:
                dense_layers.append(layers)
        return nn.Sequential(*convolutional_layers), nn.Sequential(*dense_layers)

    def _initialize_weights(self):
        pass
