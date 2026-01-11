import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F
import logging

from utils.utils import setup_logger

logger = logging.getLogger(__name__)

# This class implements a constrained linear layer, which can be used as a drop-in replacement for nn.Linear
# BaseConstrainedLinear is the base class, which implements the basic functionality,
# and ConstrainedLinear is a wrapper around it, which adds some additional functionality, such as enforcing Dale's law.
class BaseConstrainedLinear(nn.Module):
      def __init__(self, in_features, out_features, bias=True,\
             diagonals_only=False, positive_weight=False, grad_abs=False, positive_by_sigmoid=False, positive_by_softplus=False,\
                   weight_init_mean=0, weight_init_bound=None, weight_init_sparsity=None,\
                        zero_smaller_than=None, keep_max_k_from_input=None, keep_max_k_to_output=None,\
                              keep_weight_mean=None, keep_weight_std=None, keep_weight_max=None,\
                                          enforce_every_in_train_epochs=None, enforce_every_in_train_batches=None):
                  
            super(BaseConstrainedLinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.diagonals_only = diagonals_only
            self.positive_weight = positive_weight
            self.grad_abs = grad_abs
            self.positive_by_sigmoid = positive_by_sigmoid
            self.positive_by_softplus = positive_by_softplus
            self.use_bias = bias
            self.weight_init_mean = weight_init_mean
            self.weight_init_bound = weight_init_bound
            self.weight_init_sparsity = weight_init_sparsity
            self.zero_smaller_than = zero_smaller_than
            self.keep_max_k_from_input = keep_max_k_from_input
            self.keep_max_k_to_output = keep_max_k_to_output
            self.keep_weight_mean = keep_weight_mean
            self.keep_weight_std = keep_weight_std
            self.keep_weight_max = keep_weight_max
            self.enforce_every_in_train_epochs = enforce_every_in_train_epochs
            self.enforce_every_in_train_batches = enforce_every_in_train_batches

            self.current_train_epoch = None
            self.current_train_batch = None

            if self.positive_weight and self.use_bias:
                  raise ValueError("Cannot have positive weights and bias right now")

            if self.positive_by_sigmoid and self.positive_by_softplus:
                  raise ValueError("Cannot have positive weights by sigmoid and softplus")

            if self.zero_smaller_than is not None and self.use_bias:
                  raise ValueError("Cannot have zero_smaller_than and bias right now")

            if self.keep_max_k_from_input is not None and self.use_bias:
                  raise ValueError("Cannot have keep_max_k_from_input and bias right now")

            if self.keep_max_k_to_output is not None and self.use_bias:
                  raise ValueError("Cannot have keep_max_k_to_output and bias right now")

            if self.keep_weight_mean is not None and self.use_bias:
                  raise ValueError("Cannot have keep_weight_mean and bias right now")

            if self.keep_weight_std is not None and self.use_bias:
                  raise ValueError("Cannot have keep_weight_std and bias right now")

            if self.keep_weight_max is not None and self.use_bias:
                  raise ValueError("Cannot have keep_weight_max and bias right now")                  

            if self.keep_max_k_from_input is not None and self.diagonals_only:
                  raise ValueError("Cannot have keep_max_k_from_input and diagonals_only right now")

            if self.keep_max_k_to_output is not None and self.diagonals_only:
                  raise ValueError("Cannot have keep_max_k_to_output and diagonals_only right now")

            if self.keep_weight_mean is not None and self.diagonals_only:
                  raise ValueError("Cannot have keep_weight_mean and diagonals_only right now")

            if self.keep_weight_std is not None and self.diagonals_only:
                  raise ValueError("Cannot have keep_weight_std and diagonals_only right now")      

            if self.diagonals_only and self.use_bias:
                  raise NotImplementedError("Diagonals only and bias not supported (buggy?)")

            if self.weight_init_sparsity is not None:
                  if self.weight_init_sparsity < 0:
                        if self.keep_max_k_to_output is not None:
                              self.weight_init_sparsity = (-self.weight_init_sparsity) * self.keep_max_k_to_output / self.in_features
                              logger.info(f"Detected negative weight_init_sparsity, auto calculating weight_init_sparsity to be (-weight_init_sparsity) * keep_max_k_to_output / in_features = {self.weight_init_sparsity}")
                        else:
                              raise ValueError("Cannot have negative weight_init_sparsity without keep_max_k_to_output")

            if self.diagonals_only:
                  if self.out_features % self.in_features != 0:
                        raise ValueError("In the case of diagonals only, out_features must be a multiple of in_features")
                  self.k_factor = self.out_features // self.in_features

                  init_weights = []
                  for _ in range(self.k_factor):
                        param = self.weight_init_mean + nn.Linear(in_features, 1, bias=False).weight.reshape(-1)
                        if self.weight_init_bound is not None:
                              with torch.no_grad():
                                    torch.nn.init.uniform_(param,\
                                          self.weight_init_mean - self.weight_init_bound, self.weight_init_mean + self.weight_init_bound)

                        if self.positive_weight:                              
                              if not (self.positive_by_sigmoid or self.positive_by_softplus):
                                    with torch.no_grad():
                                          param = torch.abs(param)

                        if self.keep_weight_max is not None:
                              with torch.no_grad():
                                    param = param.clamp(max=self.keep_weight_max)                                    

                        if self.weight_init_sparsity is not None:
                              with torch.no_grad():
                                    param = param * (torch.rand_like(param) < self.weight_init_sparsity)

                        if self.zero_smaller_than is not None:
                              with torch.no_grad():
                                    param = param * (param > self.zero_smaller_than)

                        init_weights.append(param)

                  self.weights = nn.ParameterList([nn.Parameter(w) for w in init_weights])

                  if self.use_bias:
                        self.bias = nn.Linear(1, out_features, bias=True).bias
            else:
                  self.base_linear_layer = nn.Linear(in_features, out_features, bias=self.use_bias)
                  if self.weight_init_bound is not None:
                        nn.init.uniform_(self.base_linear_layer.weight,\
                               self.weight_init_mean - self.weight_init_bound, self.weight_init_mean + self.weight_init_bound)
                  elif self.weight_init_mean != 0:
                        with torch.no_grad():
                              self.base_linear_layer.weight.copy_(self.base_linear_layer.weight + self.weight_init_mean)

                  # not doing keep_weight_mean and keep_weight_std here, because we use the initial weight_init_mean and weight_init_bound

                  if self.positive_weight:
                        if not (self.positive_by_sigmoid or self.positive_by_softplus):
                              with torch.no_grad():
                                    self.base_linear_layer.weight.copy_(self.base_linear_layer.weight.clamp(min=0))

                  if self.keep_weight_max is not None:
                        with torch.no_grad():
                              self.base_linear_layer.weight.copy_(self.base_linear_layer.weight.clamp(max=self.keep_weight_max))

                  if self.weight_init_sparsity is not None:
                        with torch.no_grad():
                              w = self.base_linear_layer.weight
                              sparse_w = w * (torch.rand_like(w) < self.weight_init_sparsity)
                              self.base_linear_layer.weight.copy_(sparse_w)

                  if self.zero_smaller_than is not None:
                        with torch.no_grad():
                              w = self.base_linear_layer.weight
                              zeroed_w = w * (w > self.zero_smaller_than)
                              self.base_linear_layer.weight.copy_(zeroed_w)

                  if self.keep_max_k_from_input is not None:
                        with torch.no_grad():
                              w = self.base_linear_layer.weight
                              k = self.keep_max_k_from_input
                              if k > w.shape[1]:
                                    k = w.shape[1]
                              for i in range(w.shape[0]):
                                    top_k = torch.topk(w[i], k, largest=True, sorted=False).indices
                                    mask = torch.zeros_like(w[i])
                                    mask.view(-1)[top_k.view(-1)] = 1
                                    self.base_linear_layer.weight[i] = w[i] * mask

                  if self.keep_max_k_to_output is not None:
                        with torch.no_grad():
                              w = self.base_linear_layer.weight
                              k = self.keep_max_k_to_output
                              if k > w.shape[0]:
                                    k = w.shape[0]
                              for i in range(w.shape[1]):
                                    top_k = torch.topk(w[:, i], k, largest=True, sorted=False).indices
                                    mask = torch.zeros_like(w[:, i])
                                    mask.view(-1)[top_k.view(-1)] = 1
                                    self.base_linear_layer.weight[:, i] = w[:, i] * mask

      def update_train_epoch(self, train_epoch):
            self.current_train_epoch = train_epoch

      def update_train_batch(self, train_batch):
            self.current_train_batch = train_batch

      def get_weight_shape(self):
            return (self.in_features, self.out_features)

      def should_enforce_train_epoch(self):
            if self.current_train_epoch is None:
                  return True

            if self.enforce_every_in_train_epochs is None:
                  return True

            if self.current_train_epoch % self.enforce_every_in_train_epochs == 0:
                  return True

            return False
      
      def should_enforce_train_batch(self):
            if self.current_train_batch is None:
                  return True

            if self.enforce_every_in_train_batches is None:
                  return True

            if self.current_train_batch % self.enforce_every_in_train_batches == 0:
                  return True

            return False

      def should_enforce(self):
            # TODO: right now always enforcing on non train situations (validation, test, eval, etc.)
            return self.should_enforce_train_epoch() and self.should_enforce_train_batch()

      def get_weights(self):
            if self.diagonals_only:
                  for i, w in enumerate(self.weights):
                        if self.positive_weight:
                              if not (self.positive_by_sigmoid or self.positive_by_softplus):
                                    if self.grad_abs:
                                          # TODO:?
                                          # weights = torch.abs(weights)
                                          self.weights[i].copy_(w.clamp(min=0))
                                    else:
                                          with torch.no_grad():
                                                self.weights[i].copy_(w.clamp(min=0))

                        if self.keep_weight_max is not None and self.should_enforce():
                              with torch.no_grad():
                                    self.weights[i].copy_(w.clamp(max=self.keep_weight_max))                                          

                        if self.zero_smaller_than is not None and self.should_enforce():
                              # TODO: grad through?
                              with torch.no_grad():
                                    self.weights[i].copy_(w * (w > self.zero_smaller_than))
                  
                  out_weights = torch.stack([torch.diag_embed(w) for w in self.weights])
                  out_weights = out_weights.view(self.out_features, self.in_features)

                  if self.positive_weight:
                        if self.positive_by_sigmoid:
                              out_weights = torch.sigmoid(out_weights)
                        elif self.positive_by_softplus:
                              out_weights = F.softplus(out_weights)

                  return out_weights
            else:
                  if (self.keep_weight_mean is not None or self.keep_weight_std is not None) and self.should_enforce():
                        # TODO: grad through?
                        with torch.no_grad():
                              w = self.base_linear_layer.weight
                              w_mean = torch.mean(w)
                              w_std = torch.std(w)
                              desired_mean = self.keep_weight_mean if self.keep_weight_mean is not None else w_mean
                              desired_std = self.keep_weight_std if self.keep_weight_std is not None else w_std
                              w = (w - w_mean) / w_std
                              w = w * desired_std + desired_mean
                              self.base_linear_layer.weight.copy_(w)

                  # this is actually keep_weight_min
                  if self.positive_weight:
                        if not (self.positive_by_sigmoid or self.positive_by_softplus):
                              if self.grad_abs:
                                    self.base_linear_layer.weight.copy_(self.base_linear_layer.weight.clamp(min=0))
                              else:
                                    with torch.no_grad():
                                          self.base_linear_layer.weight.copy_(self.base_linear_layer.weight.clamp(min=0))

                  if self.keep_weight_max is not None and self.should_enforce():
                        # TODO: grad through?
                        with torch.no_grad():
                              self.base_linear_layer.weight.copy_(self.base_linear_layer.weight.clamp(max=self.keep_weight_max))

                  if self.zero_smaller_than is not None and self.should_enforce():
                        # TODO: grad through?
                        with torch.no_grad():
                              self.base_linear_layer.weight.copy_(self.base_linear_layer.weight * (self.base_linear_layer.weight > self.zero_smaller_than)) 
                  
                  if self.keep_max_k_from_input is not None and self.should_enforce():
                        # TODO: grad through?
                        with torch.no_grad():
                              w = self.base_linear_layer.weight
                              k = self.keep_max_k_from_input
                              if k > w.shape[1]:
                                    k = w.shape[1]
                              for i in range(w.shape[0]):
                                    top_k = torch.topk(w[i], k, largest=True, sorted=False).indices
                                    mask = torch.zeros_like(w[i])
                                    mask.view(-1)[top_k.view(-1)] = 1
                                    self.base_linear_layer.weight[i] = w[i] * mask

                  if self.keep_max_k_to_output is not None and self.should_enforce():
                        # TODO: grad through?
                        with torch.no_grad():
                              w = self.base_linear_layer.weight
                              k = self.keep_max_k_to_output
                              if k > w.shape[0]:
                                    k = w.shape[0]
                              for i in range(w.shape[1]):
                                    top_k = torch.topk(w[:, i], k, largest=True, sorted=False).indices
                                    mask = torch.zeros_like(w[:, i])
                                    mask.view(-1)[top_k.view(-1)] = 1
                                    self.base_linear_layer.weight[:, i] = w[:, i] * mask

                  out_weights = self.base_linear_layer.weight

                  if self.positive_weight:
                        if self.positive_by_sigmoid:
                              out_weights = torch.sigmoid(out_weights)
                        elif self.positive_by_softplus:
                              out_weights = F.softplus(out_weights)

                  return out_weights

      def forward(self, x):
            weights = self.get_weights()
            if self.diagonals_only:
                  if self.use_bias:
                        return F.linear(x, weights, self.bias)
                  else:
                        return F.linear(x, weights)
            else:
                  return self.base_linear_layer(x)

class ConstrainedLinear(nn.Module):
      def __init__(self, in_features, out_features, dales_law=False, **kwargs):
            super(ConstrainedLinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.dales_law = dales_law
            self.current_train_epoch = None
            self.current_train_batch = None

            if self.dales_law:
                  # for every key of kwargs, take the first value of the tuple
                  # TODO: support arbitrary numbers of excitatory and inhibitory neurons
                  self.exc_kwargs = {k: v[0] if isinstance(v, tuple) else v for k, v in kwargs.items()}
                  self.in_features_exc = in_features[0] if isinstance(in_features, tuple) else in_features // 2
                  self.out_features_exc = out_features[0] if isinstance(out_features, tuple) else out_features // 2
                  self.exc_constrained_linear = BaseConstrainedLinear(self.in_features_exc, self.out_features_exc, **self.exc_kwargs)

                  # for every key of kwargs, take the second value of the tuple
                  self.inh_kwargs = {k: v[1] if isinstance(v, tuple) else v for k, v in kwargs.items()}
                  self.in_features_inh = in_features[1] if isinstance(in_features, tuple) else in_features // 2
                  self.out_features_inh = out_features[1] if isinstance(out_features, tuple) else out_features // 2
                  self.inh_constrained_linear = BaseConstrainedLinear(self.in_features_inh, self.out_features_inh, **self.inh_kwargs)
            else:
                  self.base_constrained_layer = BaseConstrainedLinear(in_features, out_features, **kwargs)                           

      def update_train_epoch(self, train_epoch):
            if self.dales_law:
                  self.exc_constrained_linear.update_train_epoch(train_epoch)
                  self.inh_constrained_linear.update_train_epoch(train_epoch)
            else:
                  self.base_constrained_layer.update_train_epoch(train_epoch)

      def update_train_batch(self, train_batch):
            if self.dales_law:
                  self.exc_constrained_linear.update_train_batch(train_batch)
                  self.inh_constrained_linear.update_train_batch(train_batch)
            else:
                  self.base_constrained_layer.update_train_batch(train_batch)

      def get_weight_shape(self):
            if self.dales_law:
                  return (self.in_features_exc + self.in_features_inh, self.out_features_exc + self.out_features_inh)
            else:
                  return (self.in_features, self.out_features)

      def get_weights(self):
            if self.dales_law:
                  exc_weights = self.exc_constrained_linear.get_weights()
                  inh_weights = self.inh_constrained_linear.get_weights()
                  # weights = (E     0)
                  #           (0     I)
                  weights = torch.zeros((self.out_features_exc + self.out_features_inh, self.in_features_exc + self.in_features_inh))
                  weights[:self.out_features_exc, :self.in_features_exc] = exc_weights
                  weights[self.out_features_exc:, self.in_features_exc:] = inh_weights
                  return weights
            else:
                  return self.base_constrained_layer.get_weights()

      def forward(self, x):
            if self.dales_law:
                  exc_x = x[:, :, :self.in_features_exc]
                  inh_x = x[:, :, self.in_features_exc:]
                  exc_out = self.exc_constrained_linear(exc_x)
                  inh_out = self.inh_constrained_linear(inh_x)
                  total_out = torch.cat((exc_out, inh_out), dim=2)
                  return total_out
            else:
                  return self.base_constrained_layer(x)
