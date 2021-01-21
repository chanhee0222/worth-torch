import copy
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as tfunc

from models.global_worth_manager import get_worth_manager
from utils.estimator import Estimatable
from utils.summary_helper import SummarizableModule
from utils.scope import name_scope


class FFNN(Estimatable, SummarizableModule):
    def __init__(self, params):
        super().__init__()

        self._logger = logging.getLogger(__name__)

        self.params = params

        self.layers = nn.ModuleList()

        # Can't have both batch norm and layer norm.
        assert not (self.params["batch_norm"] and self.params["layer_norm"]), \
            "Can't have both batch norm and layer norm."

        assert (self.params["ort_alpha"] <= 1.0) and (self.params["ort_alpha"] >= 0.0)

        use_bias = (not self.params["batch_norm"]) and self.params["use_bias"]

        worth_manager = get_worth_manager()
        worth_manager.device = params["device"]
        worth_manager.worth_sample_rate = params["worth_sample_rate"]

        # +1 for the output (softmax) layer.
        num_layers = len(self.params["ffnn_widths"]) + 1

        # Worth loss setting for each axis in each weight.
        worth_loss = [None, None]
        for axis in range(2):
            if params["ort_loss_a%d" % axis]:
                worth_loss[axis] = copy.copy(params["ort_norm_type"])

        worth_loss_types = [copy.copy(worth_loss) for _ in range(num_layers)]
        worth_loss_types[-1][1] = None

        self.activations = [nn.ReLU() for _ in range(num_layers - 1)]
        self.activations.append(None)

        if self.params["layer_norm"]:
            raise NotImplementedError
        elif self.params["batch_norm"]:
            raise NotImplementedError
        else:
            self.normalizations = None

        widths = self.params["ffnn_widths"] + [10]

        input_dim = 28 * 28
        weights = []
        for output_dim, worth_loss_type in zip(widths, worth_loss_types):
            weight = worth_manager.new_weight(
                shape=[input_dim, output_dim],
                # weight_norm_type=params["weight_norm_type"],
                worth_loss_type=worth_loss_type,
                cos_loss_type=None,
                # do_reduce=params["reduce_weights"],
            )
            weight.dim_lower_bound = 10
            weights.append(weight)
            input_dim = output_dim

        # Output dimension of the output layer are treated differently.
        weights[-1].cos_loss_type[1] = params["output_norm_type"]
        weights[-1].worth_loss_type[1] = None

        # Don't reduce the input dimension of the input layer.
        weights[0].do_reduce_weight[0] = False
        # Don't reduce the output dimension of the output layer.
        weights[-1].do_reduce_weight[1] = False

        # Link adjacent weights for feature space reduction.
        for i in range(1, len(weights)-1, 1):
            weights[i-1].adjacent_weights[1] = weights[i]
            weights[i].adjacent_weights[0] = weights[i-1]
            weights[i+1].adjacent_weights[0] = weights[i]
            weights[i].adjacent_weights[1] = weights[i+1]

        for w in weights:
            self.layers.append(WorthDense(w, use_bias))

        self.loss_fn = nn.CrossEntropyLoss()

        self._prev_epoch = 1
        self._compression_rate = 0.0

    def optimize_weight(self, engine):
        changed = False
        worth_manager = get_worth_manager()
        # Optimize weights at the end of each epoch.
        # if (engine is not None) and (engine.state.epoch != self._prev_epoch) and self.training:
        if (engine is not None) and (engine.state.iteration % self.params["steps_per_opt"] == 0) and self.training:
            num_of_params = 0
            for param in self.parameters():
                if param.requires_grad:
                    num_of_params += np.prod(param.shape)
            if self._prev_epoch == 1:
                self._init_num_params = num_of_params
            self._compression_rate = 1.0 - (num_of_params / self._init_num_params)
            self.summary.add_scalar("compression_rate", self._compression_rate)
            worth_manager.compression_rate = self._compression_rate

            changed = worth_manager.reduce_weights()
            #     cos_th=self.params["reduction_cos_th"],
            #     euc_th=self.params["reduction_euc_th"],
            #     remove_longer=self.params["remove_longer"],
            #     symmetric_reduction=self.params["symmetric_reduction"]
            # )

            self._prev_epoch = engine.state.epoch

        if changed:
            self.loss_fn = nn.CrossEntropyLoss()
            self._logger.info(str(self))

        return changed

    def forward(self, inputs, engine=None):
        try:
            current_epoch = engine.state.epoch
            current_step = engine.state.iteration
        except Exception as e:
            self._logger.warning("Can't get the current iteration. Defaulting to 1. Msg=%s" %str(e))
            current_epoch = 1
            current_step = 1

        worth_manager = get_worth_manager()
        is_training = self.training
        self.losses = []

        ort_alpha_method = self.params["ort_alpha_method"].lower()
        if ort_alpha_method == "static":
            worth_alpha = self.params["ort_alpha"]
        elif ort_alpha_method == "alternate":
            if current_step % self.params["ort_alternate_steps"] == 0:
                worth_alpha = self.params["ort_alpha"]
            else:
                worth_alpha = 1.0
        else:
            raise ValueError("Unknown ort_alpha_method: %s" %self.params["ort_alpha_method"])

        if self.training:
            self.summary.add_scalar("worth_alpha", worth_alpha)

        with name_scope("ffnn"):
            layer_out = inputs
            layer_out = torch.reshape(layer_out, (layer_out.shape[0], np.prod(layer_out.shape[1:])))

            for layer_idx, (layer, activation) in enumerate(zip(self.layers, self.activations), start=1):
                with name_scope("dense_layer_%d" %layer_idx):
                    # Write summary of row/column magnitudes.
                    if not is_training:
                        for axis in range(2):
                            magnitude = layer.worth_weight.reduced_weight.pow(2.0).sum(1 - axis).sqrt()
                            self.summary.add_histogram("weight_mag_a%d" %axis, magnitude)
                            self.summary.add_scalar("weight_mag_a%d" %axis, magnitude.mean())

                    # residual = layer_out

                    # Wx+b (or just Wx when not using bias).
                    layer_out = layer(layer_out)

                    if self.params["layer_norm"]:
                        norm = self.normalizations[layer_idx - 1]
                        if norm is not None:
                            layer_out = norm(layer_out)

                    if not is_training:
                        self.summary.add_histogram("pre-activations_a0", torch.mean(layer_out, 0))
                        self.summary.add_histogram("pre-activations_a1", torch.mean(layer_out, 1))

                    # Non-linearity.
                    if activation is not None:
                        layer_out = activation(layer_out)

                        # Dropout.
                        def get_drop_rate():
                            dropout_type = self.params["dropout_type"].lower()
                            if dropout_type == "static":
                                drop_rate = self.params["drop_rate"]
                            elif dropout_type == "linear":
                                drop_rate = self.params["drop_rate"] * layer.worth_weight.output_dim / layer.worth_weight.original_shape[1]
                            else:
                                raise ValueError("Unknown dropout type: %s" % self.params["dropout_type"])

                            self.summary.add_scalar("drop_rate", drop_rate)
                            return drop_rate

                        if not self.params["do_after_ci"]:
                            if is_training and self.params["drop_rate"] > 0.0:
                                drop_rate = get_drop_rate()
                                layer_out = tfunc.dropout(layer_out, drop_rate, is_training)
                        if self.params["center_inputs"]:
                            # Center the features. (make mean 0)
                            l_mean = torch.mean(layer_out, dim=1, keepdim=True)
                            layer_out = layer_out - l_mean

                        # Dropout.
                        if self.params["do_after_ci"]:
                            if is_training and self.params["drop_rate"] > 0.0:
                                drop_rate = get_drop_rate()
                                layer_out = tfunc.dropout(layer_out, drop_rate, is_training)

                    if not is_training:
                        self.summary.add_histogram("activations_a0", torch.mean(layer_out, 0))
                        self.summary.add_histogram("activations_a1", torch.mean(layer_out, 1))

                    if self.params["center_weight_axis"] is not None:
                        raise NotImplementedError

                    if self.params["length_norm_axis"] is not None:
                        raise NotImplementedError

                    # if layer.worth_weight.input_dim == layer.worth_weight.output_dim:
                    #     layer_out = layer_out + residual

        worth_loss = worth_manager.make_worth_loss(worth_alpha, self.params["ort_weighting_method"])
        if worth_loss is not None:
            # Linearly increase ort weight.
            try:
                # See if "ort_loss_weight" is a list.
                init_weight, final_weight = self.params["ort_loss_weight"]

                # Linearly increase/decrease ort weight.
                worth_loss_weight = current_step / self.params["total_steps"] * (final_weight - init_weight) + init_weight

            except TypeError:
                # Do not increase ort loss weight.
                worth_loss_weight = self.params["ort_loss_weight"]

            # if (self.params["target_comp_rate"] > 0.0): # and not is_orthogonalizing:
            #     worth_loss_weight = worth_loss_weight * max((self.params["target_comp_rate"] - self._compression_rate) / self.params["target_comp_rate"], 0.0)

            if self.training:
                self.summary.add_scalar("ort_loss_weight", worth_manager.worth_loss_weight)
                self.losses.append(worth_loss)

        # L2 weight decay.
        with name_scope("l2_weight_decay"):
            l2_loss = worth_manager.make_l2_decay_loss(1.0)
            self.summary.add_scalar("l2_loss", l2_loss)
            if self.params["l2_regularization_weight"] > 0.0:
                self.losses.append(l2_loss * self.params["l2_regularization_weight"])

        # with name_scope("l2_weight_decay"):
        #     l2_losses = []
        #     for layer in self.layers:
        #         l2_loss = layer.worth_weight.reduced_weight.pow(2.0).mean()
        #         l2_losses.append(l2_loss)
        #     l2_loss = sum(l2_losses) / len(l2_losses)
        #     self.summary.add_scalar("l2_loss", l2_loss)
        #     if self.params["l2_regularization_weight"] > 0.0:
        #         self.losses.append(l2_loss * self.params["l2_regularization_weight"])

        if is_training and (current_epoch % 100 == 0):
            # Dump pairwise cosine similarity to a csv file.
            data_dir = os.path.join(self.params["model_dir"], "data")
            os.makedirs(data_dir, exist_ok=True)
            bucket_array = np.linspace(-1.0, 1.0, 41)

            for layer_idx, layer in enumerate(self.layers, start=1):
                for axis in range(2):
                    cos_sim = layer.worth_weight.pairwise_cos_sim(axis)
                    mask = torch.ones_like(cos_sim).tril(-1).bool()
                    masked_cos_sim = torch.masked_select(cos_sim, mask)
                    file_name = "epoch_%05d_layer_%d_cos_sim_a%d.csv" %(current_epoch, layer_idx, axis)
                    data = masked_cos_sim.cpu().detach().numpy()
                    df = pd.DataFrame({
                        "data": data
                    })
                    df["data"] = pd.cut(df["data"], bucket_array)
                    a = df.groupby('data').size()
                    with open(os.path.join(data_dir, file_name), "w") as out_fh:
                        np.savetxt(out_fh, np.expand_dims(a.values, 0), delimiter="\t") #, header="\t".join(a.index.values))

        return layer_out

    def compute_loss(self, logits, targets):
        # Cross-entropy loss.
        loss = self.loss_fn(logits, targets)

        if self.training:
            self.summary.add_scalar("xentropy_loss", loss.item())

        if len(self.losses) > 0:
            loss += sum(self.losses)
            self.losses = []

        return loss


class WorthDense(SummarizableModule):
    def __init__(self, worth_weight,
                 use_bias,
                 bias_initializer=(nn.init.constant_, {"val":0}),
                 ):

        super(WorthDense, self).__init__()

        self._logger = logging.getLogger(__name__)

        self.worth_weight = worth_weight
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer

        self._build()

    def _build(self):
        self.device = self.worth_weight.device
        output_dim = self.worth_weight.output_dim

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty([output_dim], device=self.device, requires_grad=True))
            self.bias_initializer[0](self.bias, **self.bias_initializer[1])
        else:
            self.bias = None

    def forward(self, inputs, save_summary=True):
        self.worth_weight.save_summaries = save_summary

        outputs = nn.functional.linear(inputs, self.worth_weight.weight.t(), self.bias)

        # Summaries.
        if save_summary and (not self.training):
            if self.bias is not None:
                self.summary.add_histogram("bias", self.bias)

        return outputs

    def __repr__(self):
        if self.use_bias:
            repr_str = "WorthDense(W:%s, B:%s)" %(str(self.worth_weight), str(list(self.bias.shape)))
        else:
            repr_str = "WorthDense(W:%s, B:None)" % (str(self.worth_weight))

        return repr_str


