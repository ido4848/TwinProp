import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

# non spiking model: (model_input_dim) -> (model_output_dim)
# TODO: support leaky relu and other options
class FullyConnected(nn.Module):
    def __init__(self, ds_input_shape, ds_output_shape, bias=True, hidden_layers=None, count_hidden_layers=None, hidden_layer_size=None,
     batch_norm=False):
        super(FullyConnected, self).__init__()
        self.ds_input_shape = ds_input_shape
        self.ds_input_dim = np.prod(ds_input_shape)

        self.ds_output_shape = ds_output_shape
        self.ds_output_dim = np.prod(ds_output_shape)
        
        self.enable_last_layer = True

        if hidden_layers is None:
            if hidden_layer_size is None or count_hidden_layers is None:
                raise ValueError("hidden_layers or hidden_layer_size and count_hidden_layers must be provided")
            self.hidden_layers = [hidden_layer_size] * count_hidden_layers
        else:
            self.hidden_layers = hidden_layers

        self.hidden_layers_list = []
        for i, layer in enumerate(self.hidden_layers):
            if i == 0:
                self.hidden_layers_list.append(nn.Linear(self.ds_input_dim, layer, bias=bias))
            else:
                self.hidden_layers_list.append(nn.Linear(self.hidden_layers[i-1], layer, bias=bias))

        self.hidden_layers_list.append(nn.Linear(self.hidden_layers[-1], self.ds_output_dim, bias=bias))

        self.hidden_layers_list = nn.ModuleList(self.hidden_layers_list)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norms = [nn.BatchNorm1d(layer) for layer in self.hidden_layers]
            self.batch_norms.append(nn.BatchNorm1d(self.ds_output_dim))
            self.batch_norms = nn.ModuleList(self.batch_norms)

        else:
            self.batch_norms = None

    def disable_last_layer(self):
        self.enable_last_layer = False
        return (self.ds_input_dim,)

    def forward(self, x):
        # TODO:?
        x = x.float()

        batch_size = x.shape[0]

        x = x.reshape(batch_size, self.ds_input_dim)

        for layer_ind, layer in enumerate(self.hidden_layers_list):
            if layer_ind == len(self.hidden_layers_list) - 1:
                if self.enable_last_layer:
                    x = layer(x)
                    if self.batch_norm:
                        x = self.batch_norms[layer_ind](x)
                    x = torch.softmax(x, dim=1)
                else:
                    pass
            else:
                x = layer(x)
                if self.batch_norm:
                    x = self.batch_norms[layer_ind](x)
                x = torch.relu(x)

        return x

    def get_model_shape(self):
        if self.enable_last_layer:
            return ((self.ds_input_shape), False, (self.ds_output_dim,), 0, 0)
        else:
            return ((self.ds_input_shape), False, (self.ds_input_dim,), 0, 0)

    def parameters(self):
        return self.hidden_layers_list.parameters()
