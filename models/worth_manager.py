import json
import logging
import math
import random

import numpy as np
import torch
from torch import nn as nn
from torch import _weight_norm, norm_except_dim
from torch.nn import functional as tfunc
from torch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out, uniform_

from utils.fairseq_dist_utils import all_gather_list, item
from utils.summary_helper import SummarizableModule
from utils.model_helper import pairwise_cos_sim, merge_rows, merge_cols, triu_argmax, merge_feature_space
from utils.scope import name_scope

USE_DEFAULT = "34987v5n23V$%&YV%$23tc84320mcx72p348t"


class WorthManager(SummarizableModule):
    def __init__(self, device=None):
        super().__init__()

        self._logger = logging.getLogger(__name__)
        self.device = device
        # self._worth_weights = nn.ModuleList()
        self._worth_weights_dict = nn.ModuleDict()
        # self._worth_weight_names = set()

        # Default values.
        self.weight_norm_type = None
        self.worth_loss_type = None
        self.cos_loss_type = None
        self._worth_loss_weight = 1.0
        self.worth_alpha = None
        self.worth_weighting_method = None
        self.red_cos_th = None
        self.red_euc_th = None
        self.remove_longer = True
        self.symmetric_reduction = True
        self.l2_loss_weight = 1.0
        self.worth_sample_rate = 1.0
        self.do_reduce_weight = True
        self.do_reduce_features = True
        self.target_comp_rate = 0.0

        # This needs to be set to use target_comp_rate.
        self.compression_rate = None

        # Optimizer instance. We update the optimizer's parameter list when parameters are changed.
        self.optimizer = None

        # root_model is required if the training is distributed.
        # Backward hook that collects gradients across devices needs to be registered to new parameters.
        self.root_model = None
        self._is_distributed_training = False

        # Fix random seed to have consistency across distributed training.
        self._random = random.Random(1)
        self._sampled_weights = None

        self.worth_loss_scaling_factor = 1.0

    @property
    def is_distributed_training(self):
        return self._is_distributed_training

    @is_distributed_training.setter
    def is_distributed_training(self, val):
        self._is_distributed_training = val
        for weight_inst in self._worth_weights:
            weight_inst.is_distributed_training = val

    @property
    def _worth_weight_names(self):
        return self._worth_weights_dict.keys()

    @property
    def _worth_weights(self):
        return self._worth_weights_dict.values()

    @property
    def worth_loss_weight(self):
        weight = self._worth_loss_weight
        if self.target_comp_rate > 0.0:
            if self.compression_rate is None:
                self._logger.warning("Can't get current compression rate.")
            else:
                assert 0.0 <= self.compression_rate < 1.0
                scale = max(1.0 - pow((self.compression_rate / self.target_comp_rate), 5.0), 0.0)
                weight *= scale

        return weight

    @worth_loss_weight.setter
    def worth_loss_weight(self, weight):
        self._worth_loss_weight = weight

    def load_hparams(self, hparams):
        self._hparams = hparams

        key_attr_map = {
            "weight_norm_type": "weight_norm_type",
            "ort_norm_type": "worth_loss_type",
            "output_norm_type": "cos_loss_type",
            "ort_loss_weight": "_worth_loss_weight",
            "ort_alpha": "worth_alpha",
            "ort_weighting_method": "worth_weighting_method",
            "reduction_cos_th": "red_cos_th",
            "reduction_euc_th": "red_euc_th",
            "remove_longer": "remove_longer",
            "symmetric_reduction": "symmetric_reduction",
            "l2_regularization_weight": "l2_loss_weight",
            "worth_sample_rate": "worth_sample_rate",
            "reduce_weights": "do_reduce_weight",
            "reduce_features": "do_reduce_features",
            "target_comp_rate": "target_comp_rate",
        }

        self._logger.info("Loaded parameters to WorthManager:")
        for hparam, attr in key_attr_map.items():
            setattr(self, attr, hparams[hparam])
            self._logger.info("\t%s=%s" %(hparam, str(hparams[hparam])))

        # self.weight_norm_type = hparams["weight_norm_type"]
        # self.worth_loss_type = hparams["ort_norm_type"]
        # self.cos_loss_type = hparams["output_norm_type"]
        # self._worth_loss_weight = hparams["ort_loss_weight"]
        # self.worth_alpha = hparams["ort_alpha"]
        # self.worth_weighting_method = hparams["ort_weighting_method"]
        # self.red_cos_th = hparams["reduction_cos_th"]
        # self.red_euc_th = hparams["reduction_euc_th"]
        # self.remove_longer = hparams["remove_longer"]
        # self.symmetric_reduction = hparams["symmetric_reduction"]
        # self.l2_loss_weight = hparams["l2_regularization_weight"]
        # self.worth_sample_rate = hparams["worth_sample_rate"]
        # self.do_reduce = hparams["reduce_weights"]
        # self.target_comp_rate = hparams["target_comp_rate"]

    def new_weight(self, shape,
                   weight_norm_type=USE_DEFAULT,
                   worth_loss_type=USE_DEFAULT,
                   cos_loss_type=USE_DEFAULT,
                   do_reduce_weight=USE_DEFAULT,
                   do_reduce_features=USE_DEFAULT,
                   name="worth_weight", device=USE_DEFAULT,
                   kernel_initializer=(nn.init.kaiming_normal_, {"nonlinearity":"relu", "mode": "fan_out"}),
                   ):

        weight_norm_type = self.weight_norm_type if weight_norm_type == USE_DEFAULT else weight_norm_type
        worth_loss_type = self.worth_loss_type if worth_loss_type == USE_DEFAULT else worth_loss_type
        cos_loss_type = self.cos_loss_type if cos_loss_type == USE_DEFAULT else cos_loss_type
        device = self.device if device == USE_DEFAULT else device
        do_reduce_weight = self.do_reduce_weight if do_reduce_weight == USE_DEFAULT else do_reduce_weight
        do_reduce_features = self.do_reduce_features if do_reduce_features == USE_DEFAULT else do_reduce_features

        # Create a new unique name.
        idx = 1
        while name + "_%d" %idx in self._worth_weight_names:
            idx += 1
        name = name + "_%d" %idx

        # Create new weight.
        weight = WorthWeight(shape, weight_norm_type, worth_loss_type, cos_loss_type, do_reduce_weight, do_reduce_features,
                             name=name, device=device, kernel_initializer=kernel_initializer)
        # self._worth_weight_names.add(weight.name)

        weight.is_distributed_training = self.is_distributed_training

        # Save the new weight as one of the weights.
        self._worth_weights_dict[name] = weight

        return weight

    def clear_caches(self):
        for weight in self._worth_weights:
            weight.clear_cache()

        self._sampled_weights = None

    def clear_caches_hook(self, module, inputs):
        self.clear_caches()

    def sample_weights(self):
        if self._sampled_weights is None:
            # Sample a subset of weights for faster training.
            if self.worth_sample_rate < 1.0:
                num_sample = round(len(self._worth_weights) * self.worth_sample_rate)
                assert num_sample >= 1, "worth_sample_rate is too small."
                worth_weights = self._random.sample(list(self._worth_weights), num_sample)
            else:
                worth_weights = self._worth_weights

            sampled_weight_names = sorted([w.name for w in worth_weights])
            # self._logger.debug("Sampled weights: " + ", ".join(sampled_weight_names))

            self._sampled_weights = worth_weights

        return self._sampled_weights

    def reduce_weights(self, cos_th=USE_DEFAULT, euc_th=USE_DEFAULT, remove_longer=USE_DEFAULT, symmetric_reduction=USE_DEFAULT):
        if (self.worth_sample_rate == 0.0) or (self.worth_loss_weight * self.worth_loss_scaling_factor == 0.0):
            return False

        cos_th = self.red_cos_th if cos_th == USE_DEFAULT else cos_th
        euc_th = self.red_euc_th if euc_th == USE_DEFAULT else euc_th
        remove_longer = self.remove_longer if remove_longer == USE_DEFAULT else remove_longer
        symmetric_reduction = self.symmetric_reduction if symmetric_reduction == USE_DEFAULT else symmetric_reduction

        any_changed = False

        # Sample a subset of weights for faster training.
        worth_weights = self.sample_weights()
        for weight_inst in worth_weights:
            this_inst_changed = False
            for axis in range(2):
                # If symmetric_reduction is True, only reduce the larger dimension.
                if (not symmetric_reduction) or (weight_inst.reduced_weight.shape[axis] >= weight_inst.reduced_weight.shape[1 - axis]):
                    changed = weight_inst.reduce_weight(axis, cos_th, euc_th, remove_longer=remove_longer)
                    any_changed = any_changed or changed
                    this_inst_changed = this_inst_changed or changed

            # If there was any change, and we're using distributed training,
            # backward hook needs to be added to the new parameter for proper training.
            if this_inst_changed and self.is_distributed_training:
                if self.root_model is None:
                    self._logger.warning("Can't attach backward hook to the new parameter. " +
                                         "This is likely to cause problems in gradient synchronization.")
                else:
                    self.root_model._register_grad_hook(weight_inst.reduced_weight)

            with name_scope(weight_inst.name):
                self.summary.add_scalar("input_dim", weight_inst.reduced_weight.shape[0])
                self.summary.add_scalar("output_dim", weight_inst.reduced_weight.shape[1])
                if weight_inst.weight2out_mat is not None:
                    self.summary.add_scalar("feature_dim", weight_inst.weight2out_mat.shape[1])

        # Update the optimizer.
        if self.optimizer is None:
            self._logger.warning("WorthManager's optimizer is not set. Parameters will not be updated!")
        elif any_changed:
            self._update_optimizer_params()

        return any_changed

    def make_l2_decay_loss(self, loss_weight=USE_DEFAULT):
        if self.l2_loss_weight == 0.0:
            return 0.0

        loss_weight = self.l2_loss_weight if loss_weight == USE_DEFAULT else loss_weight
        # L2 weight decay.
        l2_losses = []
        for weight_inst in self._worth_weights:
            with name_scope(weight_inst.name):
                l2_loss = weight_inst.reduced_weight.pow(2.0).mean()
                self.summary.add_scalar("l2_loss", l2_loss)
            l2_losses.append(l2_loss)
        l2_loss = sum(l2_losses) / len(l2_losses)
        self.summary.add_scalar("losses/l2_loss", l2_loss)
        return l2_loss * loss_weight

    def make_worth_loss(self, worth_alpha=USE_DEFAULT, weighting_method=USE_DEFAULT):
        self.summary.add_scalar("stats/worth_loss_weight", self.worth_loss_weight * self.worth_loss_scaling_factor)
        if self.worth_loss_weight * self.worth_loss_scaling_factor == 0.0:
            return 0.0

        worth_alpha = self.worth_alpha if worth_alpha == USE_DEFAULT else worth_alpha
        weighting_method = self.worth_weighting_method if weighting_method == USE_DEFAULT else weighting_method

        # Sample a subset of weights for faster training.
        worth_weights = self.sample_weights()

        ort_losses_and_weights = []
        for weight_idx, weight_inst in enumerate(worth_weights, start=1):
            ort_losses = []
            for a in range(2):
                # Axis 0 = input, axis 1 = output.

                # Don't normalize along an axis if it's reached the size lower bound.
                if weight_inst.reduced_weight.shape[a] <= weight_inst.dim_lower_bound:
                    weight_inst.worth_loss_type[a] = None

                # Make the loss to generate summaries, even the loss itself is not used.
                ort_loss = weight_inst.make_worth_loss(a, worth_alpha)

                if weight_inst.worth_loss_type[a] is not None:
                    ort_losses.append(ort_loss)

                if weight_inst.cos_loss_type[a] is not None:
                    # Minimize the pairwise cosine similarity of output embeddings.
                    ort_loss = weight_inst.make_cos_loss(a)
                    ort_losses.append(ort_loss)

            if len(ort_losses) > 0:
                ort_loss = sum(ort_losses) / len(ort_losses)
                if weighting_method == "fan_avg":
                    ort_weight = sum(weight_inst.reduced_weight.shape) / 2.0
                elif weighting_method == "fan_prod":
                    ort_weight = np.prod(weight_inst.reduced_weight.shape)
                elif weighting_method == "fan_max":
                    ort_weight = max(weight_inst.reduced_weight.shape)
                elif weighting_method == "uniform":
                    ort_weight = 1
                else:
                    raise ValueError("Unknown weighting method: %s" % weighting_method)

                ort_losses_and_weights.append((ort_loss, ort_weight))

        if len(ort_losses_and_weights) > 0:
            weight_sum = 0
            ort_loss = []
            for single_ort_loss, loss_weight in ort_losses_and_weights:
                weight_sum += loss_weight
                ort_loss.append(single_ort_loss * loss_weight)

            ort_loss = sum(ort_loss) / weight_sum

        else:
            ort_loss = None

        self.summary.add_scalar("losses/total_worth_loss", ort_loss)

        ort_loss = ort_loss * self.worth_loss_scaling_factor * self.worth_loss_weight

        return ort_loss

    def _update_optimizer_params(self):
        param_groups = self.optimizer.param_groups
        assert isinstance(param_groups, list)

        self._logger.debug("Updating optimizer parameters.")

        def check_worth_param(param):
            is_worth = hasattr(param, "is_worth_param") and param.is_worth_param
            # self._logger.debug("Is worth:%s, shape=%s" %(is_worth, param.shape))
            # if is_worth:
            #     self._logger.debug(param.worth_name)
            return is_worth

        for param_group in param_groups:
            assert isinstance(param_group, dict)
            for key in param_group:
                if isinstance(param_group[key], list):
                    new_param_list = []
                    for param in param_group[key]:

                        # Replace all worth params in the optimizer with the new ones.
                        if check_worth_param(param):
                            new_param = self._worth_weights_dict[param.worth_name].reduced_weight
                            new_param_list.append(new_param)

                            # Remove the old param from the optimizer's state dict.
                            del self.optimizer.state[param]
                        else:
                            new_param_list.append(param)
                    param_group[key] = new_param_list

                elif isinstance(param_group[key], torch.Tensor):
                    param = param_group[key]
                    # Replace all worth params in the optimizer with the new ones.
                    if check_worth_param(param):
                        new_param = self._worth_weights_dict[param.worth_name].reduced_weight
                        param_group[key] = new_param

                        # Remove the old param from the optimizer's state dict.
                        del self.optimizer.state[param]

    def save_model_architecture(self, filename=None):
        if filename is not None:
            # Dump the shape of worth weights into a json file, for model load.
            self._logger.info("Saving Worth architecture to file: %s" %filename)
            out_file = filename + ".json"
        else:
            # Dump the shape of worth weights into a dictionary, for model sync.
            self._logger.info("Saving Worth architecture to a dictionary.")
            out_file = None

        param_shape_dict = {}
        for name, weight_inst in self._worth_weights_dict.items():
            assert name not in param_shape_dict
            param_shape_dict[name] = list(weight_inst.reduced_weight.shape)
            if weight_inst.in2weight_mat is not None:
                param_shape_dict[name + "/in2weight_mat"] = list(weight_inst.in2weight_mat.shape)
            if weight_inst.weight2out_mat is not None:
                param_shape_dict[name + "/weight2out_mat"] = list(weight_inst.weight2out_mat.shape)

        if out_file is not None:
            with open(out_file, "w") as out_fh:
                json.dump(param_shape_dict, out_fh, indent=2, sort_keys=True)

        return param_shape_dict

    def load_model_architecture(self, filename=None, shape_dict=None):
        # Restore the shape of each worth weight.
        assert (filename is None) != (shape_dict is None)

        if filename is not None:
            self._logger.info("Restoring Worth architecture from file: %s" %filename)
            in_file = filename + ".json"
            with open(in_file, "r") as in_fh:
                param_shape_dict = json.load(in_fh)
        else:
            self._logger.info("Restoring Worth architecture from a dictionary.")
            param_shape_dict = shape_dict

        saved_params_set = set()
        for key in param_shape_dict.keys():
            weight_type = key.split('/')[-1]
            if weight_type == "in2weight_mat" or weight_type == "weight2out_mat":
                continue
            saved_params_set.add(key)

        current_params_set = set(self._worth_weights_dict.keys())

        # List of params that are in the save but not in the current model.
        orph_saved_names = saved_params_set - current_params_set
        if len(orph_saved_names) > 0:
            self._logger.warning(
                "%d parameters that are in save file but not in current model:" %len(orph_saved_names))
            for orph_saved_name in sorted(list(orph_saved_names)):
                self._logger.warning("\t%s" %orph_saved_name)

        orph_param_names = current_params_set - saved_params_set
        if len(orph_param_names) > 0:
            self._logger.warning(
                "%d parameters that are in current model but not in save file:" % len(orph_param_names))
            for orph_param_name in sorted(list(orph_param_names)):
                self._logger.warning("\t%s" % orph_param_name)

        dtype = next(iter(self._worth_weights)).reduced_weight.dtype
        device = next(iter(self._worth_weights)).reduced_weight.device

        for name in (saved_params_set & current_params_set):
            worth_inst = self._worth_weights_dict[name]
            if name + "/in2weight_mat" in param_shape_dict:
                shape = param_shape_dict[name + "/in2weight_mat"]
                worth_inst.in2weight_mat = torch.empty(shape, device=device, dtype=dtype)

            if name + "/weight2out_mat" in param_shape_dict:
                shape = param_shape_dict[name + "/weight2out_mat"]
                worth_inst.weight2out_mat = torch.empty(shape, device=device, dtype=dtype)

            worth_inst.reduced_weight = torch.empty(param_shape_dict[name], device=device, dtype=dtype)


class WorthWeight(SummarizableModule):
    def __init__(self, shape, weight_norm_type, worth_loss_type, cos_loss_type, do_reduce_weight, do_reduce_features,
                 device, name, kernel_initializer,
                 ):

        super(WorthWeight, self).__init__()

        self._logger = logging.getLogger(__name__)

        assert len(shape) == 2

        self.device = device
        self.original_shape = shape
        self.weight_norm_type = weight_norm_type

        # Legacy parameter. (It's always None)
        self.bias = None

        if worth_loss_type is None or isinstance(worth_loss_type, str):
            self.worth_loss_type = [worth_loss_type, worth_loss_type]
        else:
            assert len(worth_loss_type) == 2
            self.worth_loss_type = worth_loss_type

        if cos_loss_type is None or isinstance(cos_loss_type, str):
            self.cos_loss_type = [cos_loss_type, cos_loss_type]
        else:
            assert len(cos_loss_type) == 2
            self.cos_loss_type = cos_loss_type

        if isinstance(do_reduce_weight, bool):
            self.do_reduce_weight = [do_reduce_weight, do_reduce_weight]
        else:
            assert len(do_reduce_weight) == 2
            self.do_reduce_weight = do_reduce_weight

        self.do_reduce_features = do_reduce_features

        self.kernel_initializer = kernel_initializer
        self.name = name

        self._in2weight_mat = None
        self._weight2out_mat = None
        self._reduced_weight = None

        # Previous and next weight is used to reduce feature space.
        self.adjacent_weights = [None, None]

        self.save_summaries = True

        # Lower bound of the weight matrix's dimension.
        self.dim_lower_bound = 0

        # self.register_buffer("_weight_cache", None)
        self._weight_cache = None
        self._pairwise_cos_sim = [None, None]

        self.is_distributed_training = False

        self._build()

    @property
    def input_dim(self):
        if self.in2weight_mat is not None:
            input_dim = self.in2weight_mat.shape[0]
        else:
            input_dim = self.reduced_weight.shape[0]

        return input_dim

    @property
    def output_dim(self):
        if self.weight2out_mat is not None:
            output_dim = self.weight2out_mat.shape[1]
        else:
            output_dim = self.reduced_weight.shape[1]

        return output_dim

    @property
    def reduced_weight(self):
        return self._reduced_weight

    @reduced_weight.setter
    def reduced_weight(self, new_tensor):
        assert isinstance(new_tensor, torch.Tensor)
        if self._reduced_weight is not None:
            self._reduced_weight.detach_()

        if isinstance(new_tensor, nn.Parameter):
            if not new_tensor.requires_grad:
                self._logger.warning("New worth_weight (%s) have requires_grad=False." %self.name)
        else:
            new_tensor = nn.Parameter(new_tensor, requires_grad=True)

        self._reduced_weight = new_tensor
        self._reduced_weight.is_worth_param = True
        self._reduced_weight.worth_name = self.name

    @property
    def in2weight_mat(self):
        return self._in2weight_mat

    @in2weight_mat.setter
    def in2weight_mat(self, new_tensor):
        if self._in2weight_mat is not None:
            self._in2weight_mat.detach_()

        if isinstance(new_tensor, nn.Parameter):
            if new_tensor.requires_grad:
                self._logger.warning("New in2weight_mat (%s) have requires_grad=True." %self.name)
        else:
            new_tensor = nn.Parameter(new_tensor, requires_grad=False)

        self._in2weight_mat = new_tensor
        self._in2weight_mat.is_worth_param = True
        self._in2weight_mat.worth_name = self.name + "/in2weight_mat"

    @property
    def weight2out_mat(self):
        return self._weight2out_mat

    @weight2out_mat.setter
    def weight2out_mat(self, new_tensor):
        if self._weight2out_mat is not None:
            self._weight2out_mat.detach_()

        if isinstance(new_tensor, nn.Parameter):
            if new_tensor.requires_grad:
                self._logger.warning("New weight2out_mat (%s) have requires_grad=True." %self.name)
        else:
            new_tensor = nn.Parameter(new_tensor, requires_grad=False)

        self._weight2out_mat = new_tensor
        self._weight2out_mat.is_worth_param = True
        self._weight2out_mat.worth_name = self.name + "/weight2out_mat"

    def pairwise_cos_sim(self, axis):
        if self._pairwise_cos_sim[axis] is None:
            self._pairwise_cos_sim[axis] = pairwise_cos_sim(self.reduced_weight, axis)

        return self._pairwise_cos_sim[axis]

    def clear_cache(self):
        self._weight_cache = None
        self._pairwise_cos_sim = [None, None]

    def _build(self):
        input_dim = self.original_shape[0]
        output_dim = self.original_shape[1]
        device = self.device

        self.reduced_weight = torch.empty([input_dim, output_dim], device=device, requires_grad=True)
        # self.reduced_weight = nn.Parameter(torch.empty([input_dim, output_dim], device=device, requires_grad=True))
        # self.reduced_weight.is_worth_param = True
        # self.reduced_weight.worth_name = self.name

        self.kernel_initializer[0](self.reduced_weight, **self.kernel_initializer[1])
        # std = math.sqrt(2.0 / input_dim)
        # torch.nn.init.normal_(self.reduced_weight, std=std)

        if self.weight_norm_type is not None:
            if self.weight_norm_type == "wn":
                # Weight normalization.
                self.weight_g = nn.Parameter(norm_except_dim(self.reduced_weight, 2, 1).data)
                self.scale = nn.Parameter(torch.ones([1], device=device), requires_grad=False)
            else:
                self.scale = nn.Parameter(torch.ones([1], device=device), requires_grad=True)
                # self.scale = nn.Parameter(torch.ones([1], device=device), requires_grad=False)
                # print("scale is not trainable!!")
        else:
            self.scale = nn.Parameter(torch.ones([1], device=device), requires_grad=False)

        self.update_weight_norm()

    def update_weight_norm(self):
        if (self.weight_norm_type is not None) and (self.weight_norm_type != "wn"):
            weight_norm_args = self.weight_norm_type.lower().split("_")

            if weight_norm_args[1] == "l2":
                init_norm_val = self.reduced_weight.pow(2)
            elif weight_norm_args[1] == "l1":
                init_norm_val = self.reduced_weight.abs()
            else:
                raise ValueError("Unknown norm type: %s" % self.weight_norm_type)

            if weight_norm_args[2] == "mean":
                init_norm_val = init_norm_val.mean()
            elif weight_norm_args[2] == "sum":
                init_norm_val = init_norm_val.sum()
            else:
                raise ValueError("Unknown norm type: %s" % self.weight_norm_type)

            if weight_norm_args[1] == "l2":
                init_norm_val = init_norm_val.sqrt()

            self._init_norm_val = init_norm_val.item()
        else:
            self._init_norm_val = None

    @property
    def weight(self):
        return self(None)

    def forward(self, *not_used):
        if self._weight_cache is None:
            # print(self.name)
            with name_scope(self.name):
                if self.weight_norm_type is None:
                    virtual_weight = self.reduced_weight
                    if self.in2weight_mat is not None:
                        virtual_weight = self.in2weight_mat.matmul(virtual_weight)

                    if self.weight2out_mat is not None:
                        virtual_weight = virtual_weight.matmul(self.weight2out_mat)

                elif self.weight_norm_type == "wn":
                    # Weight normalization.
                    virtual_weight = self.reduced_weight
                    if self.in2weight_mat is not None:
                        virtual_weight = self.in2weight_mat.matmul(virtual_weight)

                    if self.weight2out_mat is not None:
                        virtual_weight = virtual_weight.matmul(self.weight2out_mat)

                    virtual_weight = _weight_norm(virtual_weight, self.weight_g, 1)
                else:
                    weight_norm_args = self.weight_norm_type.lower().split("_")
                    if weight_norm_args[0] not in ["inner", "outer"] \
                            or weight_norm_args[1] not in ["l1", "l2"] \
                            or weight_norm_args[2] not in ["mean", "sum"]:
                        raise ValueError("Unknown norm type: %s" % self.weight_norm_type)

                    virtual_weight = self.reduced_weight
                    if weight_norm_args[0] == "outer":
                        if self.in2weight_mat is not None:
                            virtual_weight = self.in2weight_mat.matmul(virtual_weight)

                        if self.weight2out_mat is not None:
                            virtual_weight = virtual_weight.matmul(self.weight2out_mat)

                    if weight_norm_args[1] == "l2":
                        norm_val = virtual_weight.pow(2)
                    elif weight_norm_args[1] == "l1":
                        norm_val = virtual_weight.abs()

                    if weight_norm_args[2] == "mean":
                        norm_val = norm_val.mean()
                    elif weight_norm_args[2] == "sum":
                        norm_val = norm_val.sum()

                    if weight_norm_args[1] == "l2":
                        norm_val = torch.rsqrt(norm_val + 1e-6)
                    elif weight_norm_args[1] == "l1":
                        norm_val = 1.0 / (norm_val + 1e-6)

                    # print(1.0 / norm_val, self._init_norm_val)
                    # print(norm_val * self._init_norm_val, self.scale)
                    virtual_weight = virtual_weight * norm_val * self._init_norm_val * self.scale

                    if weight_norm_args[0] == "inner":
                        if self.in2weight_mat is not None:
                            virtual_weight = self.in2weight_mat.matmul(virtual_weight)

                        if self.weight2out_mat is not None:
                            virtual_weight = virtual_weight.matmul(self.weight2out_mat)

                # Summaries.
                if self.save_summaries: # and (not self.training):
                    self.summary.add_scalar("scale", self.scale)
                    self.summary.add_histogram("normalized_weight", virtual_weight)

                    # Summaries.
                    self.summary.add_histogram("weight_a0", self.reduced_weight.sum(0))
                    self.summary.add_histogram("weight_a1", self.reduced_weight.sum(1))

            # self.register_buffer("_weight_cache", virtual_weight)
            if isinstance(virtual_weight, nn.Parameter):
                virtual_weight = virtual_weight * 1.0

            self._weight_cache = virtual_weight

        return self._weight_cache

    def make_cos_loss(self, axis):
        norm_type = self.cos_loss_type[axis]
        cos_sim = self.pairwise_cos_sim(1 - axis)

        norm_type = norm_type.lower()
        if norm_type is None:
            pass
        elif norm_type == "raw":
            cos_loss = torch.mean((cos_sim + 1.0) / 2.0)
        elif norm_type == "l2":
            squared_cos_sim = cos_sim.pow(2.0)
            cos_loss = torch.mean(squared_cos_sim)
        elif norm_type == "inf":
            # mask = torch.ones_like(cos_sim).tril(-1).byte()
            # masked_cos_sim = torch.masked_select(cos_sim, mask)
            # cos_loss = torch.max(masked_cos_sim)
            row, col = triu_argmax(cos_sim)
            cos_loss = cos_sim[row][col]
        elif norm_type[:3] == "inf":
            _, num_reduce = norm_type.split("_")
            num_reduce = int(num_reduce)
            mask = torch.ones_like(cos_sim).tril(-1).bool()
            masked_cos_sim = torch.masked_select(cos_sim, mask)
            cos_loss = masked_cos_sim.topk(num_reduce, largest=True, sorted=False)[0].mean()
        elif norm_type == "ang":
            row, col = triu_argmax(cos_sim)
            cos_loss = math.pi - torch.acos(cos_sim[row][col])
        elif norm_type[:3] == "ang":
            _, num_reduce = norm_type.split("_")
            num_reduce = int(num_reduce)
            mask = torch.ones_like(cos_sim).tril(-1).bool()
            masked_cos_sim = torch.masked_select(cos_sim, mask)
            cos_loss = masked_cos_sim.topk(num_reduce, largest=True, sorted=False)[0]
            cos_loss = math.pi - torch.acos(cos_loss)
            cos_loss = cos_loss.mean()
        elif norm_type == "rec_ang_2":
            row, col = triu_argmax(cos_sim)
            cos_loss = torch.acos(cos_sim[row][col]).pow(-2.0)
        else:
            raise ValueError("Unknown norm type for cosine loss: %s" % norm_type)

        return cos_loss

    def make_worth_loss(self, axis, alpha, cos_th=0.1):
        loss_type = self.worth_loss_type[axis]
        cos_sim = self.pairwise_cos_sim(1 - axis)

        with name_scope("%s/worth_loss_a%d" % (self.name, axis)):
            # if not self.training:
            self.summary.add_histogram("cos_sim_a%d" % (1 - axis), cos_sim)

            # Sine similarity.
            squared_cos_sim = cos_sim.pow(2.0)
            squared_sin_sim = 1.0 - squared_cos_sim

            # Ort loss.
            # Multiply constants to make the function's range [0, 1].
            if loss_type is None:
                worth_loss = None
            elif loss_type.lower() == "l2":
                worth_loss = torch.mean((1.0 - alpha) * squared_cos_sim * squared_sin_sim * 4.0
                                          + (alpha * squared_sin_sim))
            elif loss_type.lower() == "l1":
                worth_loss = torch.mean((1.0 - alpha) * torch.sqrt(squared_cos_sim * squared_sin_sim + 1e-6) * 2.0
                                          + (alpha * squared_sin_sim))
            elif loss_type.lower() == "min1":
                mask = torch.ones_like(cos_sim).tril(-1).bool()

                if alpha < 1.0:
                    masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                    squared_sin_cos_loss = masked_squared_sin_cos.mean()
                else:
                    squared_sin_cos_loss = 0.0
                squared_sin_loss = torch.masked_select(squared_sin_sim, mask).min()
                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * squared_sin_loss)
            elif loss_type.lower() == "min2":
                mask = torch.ones_like(cos_sim).tril(-1).bool()
                masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                squared_sin_cos_loss = masked_squared_sin_cos.min()
                squared_sin_loss = torch.masked_select(squared_sin_sim, mask).min()
                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * squared_sin_loss)
            elif loss_type.lower() == "min3":
                mask = torch.ones_like(cos_sim).tril(-1).bool()
                masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                squared_sin_cos_loss = masked_squared_sin_cos.mean()
                num_reduce = max(cos_sim.shape[0] // 100, 1)
                squared_sin_loss = torch.masked_select(squared_sin_sim, mask).topk(num_reduce, largest=False, sorted=False)[0].mean()
                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * squared_sin_loss)
            elif loss_type.lower()[:4] == "min4":
                _, num_reduce = loss_type.split("_")
                num_reduce = int(num_reduce)
                mask = torch.ones_like(cos_sim).tril(-1).bool()
                if alpha < 1.0:
                    masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                    squared_sin_cos_loss = masked_squared_sin_cos.mean()
                else:
                    squared_sin_cos_loss = 0.0
                squared_sin_loss = torch.masked_select(squared_sin_sim, mask).topk(num_reduce, largest=False, sorted=False)[0].mean()
                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * squared_sin_loss)
            elif loss_type.lower() == "min5":
                mask = torch.ones_like(cos_sim).tril(-1).bool()
                masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                squared_sin_cos_loss = masked_squared_sin_cos.mean()

                neg_cos_loss = cos_sim * -0.5 + 0.5
                neg_cos_loss = torch.masked_select(neg_cos_loss, mask).min()
                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * neg_cos_loss)

            elif loss_type.lower() == "min6":
                mask = torch.ones_like(cos_sim).tril(-1).bool()
                masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                squared_sin_cos_loss = masked_squared_sin_cos.mean()
                masked_cos_sim = torch.masked_select(cos_sim, mask)
                cos_sim_pow = masked_cos_sim.pow(8.0)
                cos_mask = (cos_sim_pow > 1e-2)
                squared_sin_sim = torch.masked_select(1.0 - cos_sim_pow, cos_mask)
                squared_sin_loss = squared_sin_sim.min()
                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * squared_sin_loss)

            elif loss_type.lower() == "min7":
                mask = torch.ones_like(cos_sim).tril(-1).bool()
                masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                squared_sin_cos_loss = masked_squared_sin_cos.mean()
                masked_cos_sim = torch.masked_select(squared_cos_sim, mask)
                cos_mask = (torch.masked_select(cos_sim.abs(), mask) > cos_th)
                squared_sin_sim = torch.masked_select(1.0 - masked_cos_sim, cos_mask)
                squared_sin_loss = squared_sin_sim.min()
                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * squared_sin_loss)

            elif loss_type.lower() == "min8":
                mask = torch.ones_like(cos_sim).tril(-1).bool()
                masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                squared_sin_cos_loss = masked_squared_sin_cos.mean()

                row, col = triu_argmax(squared_cos_sim)
                if cos_sim[row][col].abs() > cos_th:
                    squared_sin_loss = 1.0 - squared_cos_sim[row][col]
                    if axis == 1:
                        vec1 = self.reduced_weight[:, row]
                        vec2 = self.reduced_weight[:, col]
                    else:
                        vec1 = self.reduced_weight[row]
                        vec2 = self.reduced_weight[col]

                    vec1_mag = vec1.pow(2).sum().sqrt()
                    vec2_mag = vec2.pow(2).sum().sqrt()
                    cos_val = 1 if cos_sim[row][col] > 0 else -1
                    if vec1_mag < vec2_mag:
                        vec_diff = (vec1 * vec2_mag / vec1_mag) - cos_val * vec2
                    else:
                        vec_diff = vec1 - (cos_val * vec2 * vec1_mag / vec2_mag)
                    vec_diff = vec_diff.pow(2).mean()
                    paral_loss = squared_sin_loss + vec_diff
                else:
                    paral_loss = 0.0

                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * paral_loss)

            elif loss_type.lower() == "min9":
                mask = torch.ones_like(cos_sim).tril(-1).bool()
                masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                squared_sin_cos_loss = masked_squared_sin_cos.mean()

                row, col = triu_argmax(squared_cos_sim)
                squared_sin_loss = 1.0 - squared_cos_sim[row][col]
                if axis == 1:
                    vec1 = self.reduced_weight[:, row]
                    vec2 = self.reduced_weight[:, col]
                else:
                    vec1 = self.reduced_weight[row]
                    vec2 = self.reduced_weight[col]

                vec1_mag = vec1.pow(2).sum().sqrt()
                vec2_mag = vec2.pow(2).sum().sqrt()
                cos_val = 1 if cos_sim[row][col] > 0 else -1
                if vec1_mag < vec2_mag:
                    vec_diff = (vec1 * vec2_mag / vec1_mag) - cos_val * vec2
                else:
                    vec_diff = vec1 - (cos_val * vec2 * vec1_mag / vec2_mag)
                vec_diff = vec_diff.pow(2).mean()
                paral_loss = squared_sin_loss + vec_diff

                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * paral_loss)

            elif loss_type.lower() == "min10":
                mask = torch.ones_like(cos_sim).tril(-1).bool()
                if alpha < 1.0:
                    masked_squared_sin_cos = torch.masked_select(squared_cos_sim * squared_sin_sim * 4.0, mask)
                    squared_sin_cos_loss = masked_squared_sin_cos.mean()
                else:
                    squared_sin_cos_loss = 0.0

                row, col = triu_argmax(squared_cos_sim)
                squared_sin_loss = 1.0 - squared_cos_sim[row][col]
                paral_loss = squared_sin_loss

                worth_loss = (1.0 - alpha) * squared_sin_cos_loss + (alpha * paral_loss)

            else:
                raise ValueError("Unknown worth loss type: %s" %loss_type)

            if worth_loss is not None:
                self.summary.add_scalar("loss", worth_loss)

            # Mean Absolute Cosine similarity.
            mac_sim = cos_sim.abs().sum(0) / cos_sim.shape[0]
            # self.summary.add_histogram("mac_sim", mac_sim)
            self.summary.add_scalar("mac_sim", torch.mean(mac_sim))

        return worth_loss

    def reduce_weight(self, axis, cos_th, euc_th, check_equality=False, remove_longer=True):
        if not self.do_reduce_weight[axis]:
            return False

        if self.reduced_weight.shape[axis] <= self.dim_lower_bound:
            return False

        with name_scope(self.name):
            with torch.no_grad():
                init_output = None
                random_input = None
                changed = False
                prev_shape = self.reduced_weight.shape
                euc_dist_list = []

                # # Only optimize the larger dimension.
                # axis=0 means input dimension, axis=1 means output dimension.
                # cos_sim = pairwise_cos_sim(self.weight, 1 - axis)

                # See how many columns/rows can be optimized, and skip if there isn't any.
                # print(opt_targets.sum(dim=0))
                # total_count = opt_targets.sum()
                # self.summary.add_scalar("opt_targets_a%d" %axis, total_count)

                if self.reduced_weight.shape[axis] <= self.dim_lower_bound:
                    return changed

                cos_sim = self.pairwise_cos_sim(1 - axis)
                size = cos_sim.shape[0]

                abs_cos_sim = cos_sim.abs()
                row, col = triu_argmax(abs_cos_sim)
                max_val = abs_cos_sim[row][col]

                if self.is_distributed_training:
                    row = item(row)
                    col = item(col)
                    max_val = item(max_val)
                    # Check if all processes are reducing the same indices.
                    row_list, col_list, max_val_list = \
                        zip(*all_gather_list(
                            [row, col, max_val],
                        ))

                    if (not (all(r == row_list[0] for r in row_list))
                        or not (all(c == col_list[0] for c in col_list))
                        or not (all(v == max_val_list[0] for v in max_val_list))
                    ):
                        self._logger.warning("Out of sync detected: %s, %s, %s"
                                                 %(str(row_list), str(col_list), str(max_val_list)))
                        raise OutOfSyncException

                if max_val < cos_th:
                    return changed
                else:
                    if check_equality and (init_output is None):
                        test_size = 100
                        random_input = torch.randn([test_size, self.input_dim], dtype=self.reduced_weight.dtype,
                                                   device=self.reduced_weight.device)
                        init_output = self(random_input, save_summary=False)

                    if axis == 0:
                        # Optimize the input dimension.
                        if self.in2weight_mat is None:
                            self.in2weight_mat = self.reduced_weight.new_empty([size, size])
                            nn.init.eye_(self.in2weight_mat)
                            # self.in2weight_mat = self.in2weight_mat.to_sparse()

                        target_weight = self.reduced_weight
                    else:
                        # Optimize the output dimension
                        if self.weight2out_mat is None:
                            self.weight2out_mat = self.reduced_weight.new_empty([size, size])
                            nn.init.eye_(self.weight2out_mat)
                            # self.weight2out_mat = self.weight2out_mat.to_sparse()

                        target_weight = self.reduced_weight.t()

                    vec1_idx = row
                    vec2_idx = col

                    vec1 = target_weight[vec1_idx]
                    vec2 = target_weight[vec2_idx]

                    vec1_mag = vec1.pow(2).sum().sqrt()
                    vec2_mag = vec2.pow(2).sum().sqrt()

                    if remove_longer and (vec1_mag > vec2_mag):
                        # Make vec1 the shorter one.
                        vec1_idx, vec2_idx = vec2_idx, vec1_idx
                    elif (not remove_longer) and (vec1_mag < vec2_mag):
                        # Make vec1 the longer one.
                        vec1_idx, vec2_idx = vec2_idx, vec1_idx

                    # Calculate and log the length-normalized Euclidean distance.
                    cos_val = 1 if cos_sim[row][col] > 0 else -1
                    if remove_longer:
                        euc_dist = (vec1 * vec2_mag / vec1_mag) - cos_val * vec2
                    else:
                        euc_dist = vec1 - (cos_val * vec2 * vec1_mag / vec2_mag)

                    euc_dist = euc_dist.pow(2).sum().sqrt()

                    if euc_dist > euc_th:
                        self.summary.add_scalar("euc_dist_reject", euc_dist)
                        return changed

                    # print(euc_dist)
                    # print(torch.acos(torch.dot(vec1, vec2) / vec1_mag / vec2_mag))
                    euc_dist_list.append(euc_dist)

                    if axis == 0:
                        new_weight, new_trans_mat = merge_rows(self.reduced_weight, self.in2weight_mat, vec1_idx, vec2_idx)
                    else:
                        new_weight, new_trans_mat, new_bias = merge_cols(self.reduced_weight, self.weight2out_mat, vec1_idx, vec2_idx, self.bias)

                    self.reduced_weight = new_weight
                    # self.reduced_weight.detach_()
                    # self.reduced_weight = nn.Parameter(new_weight, requires_grad=True)
                    # self.reduced_weight.is_worth_param = True
                    # self.reduced_weight.worth_name = self.name

                    assert not new_trans_mat.requires_grad

                    if axis == 0:
                        self.in2weight_mat = new_trans_mat
                    else:
                        self.weight2out_mat = new_trans_mat
                        if new_bias is not None:
                            self.bias.detach_()
                            self.bias = nn.Parameter(new_bias, requires_grad=True)

                if (self.reduced_weight.shape[0] != prev_shape[0]) or (self.reduced_weight.shape[1] != prev_shape[1]):
                    self._logger.debug("Reduced %s: %s -> %s." % (self.name, str(list(prev_shape)), str(list(self.reduced_weight.shape))))
                    changed = True
                    # timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                    # np.savetxt("debug/%s.csv" %timestamp, self.in2weight_mat.tolist(), delimiter=",")

                if changed and self.do_reduce_features:
                    # Try to reduce the feature space.
                    if axis == 0:
                        if self.adjacent_weights[0] is not None:
                            weight2out = self.adjacent_weights[0].weight2out_mat
                        else:
                            weight2out = None
                        in2weight = self.in2weight_mat
                    else:
                        weight2out = self.weight2out_mat
                        if self.adjacent_weights[1] is not None:
                            in2weight = self.adjacent_weights[1].in2weight_mat
                        else:
                            in2weight = None

                    if weight2out is not None and in2weight is not None:
                        new_params = merge_feature_space(weight2out, in2weight)
                        if new_params is not None:
                            if axis == 0:
                                self.adjacent_weights[0].weight2out_mat = new_params[0]
                                self.in2weight_mat = new_params[1]
                            else:
                                self.weight2out_mat = new_params[0]
                                self.adjacent_weights[1].in2weight_mat = new_params[1]

                if len(euc_dist_list) > 0:
                    self.summary.add_scalar("euc_dist_avg", sum(euc_dist_list) / len(euc_dist_list))
                    self.summary.add_scalar("euc_dist_sum", sum(euc_dist_list))
                else:
                    self.summary.add_scalar("euc_dist_avg", 0)
                    self.summary.add_scalar("euc_dist_sum", 0)

                if check_equality and (init_output is not None):
                    new_output = self(random_input, save_summary=False)
                    diff = init_output - new_output
                    self.summary.add_histogram("diff", diff)
                    self.summary.add_scalar("diff", diff.abs().mean())
                    # diff = torch.sum(torch.abs(init_output - new_output)) / test_size

        return changed

    def __repr__(self):
        in_size = self.reduced_weight.shape[0] if self.in2weight_mat is None else self.in2weight_mat.shape[0]
        out_size = self.reduced_weight.shape[1] if self.weight2out_mat is None else self.weight2out_mat.shape[1]
        weight_shape = list(self.reduced_weight.shape)
        repr_str = "WW(%s, %s) %s (%d-%d-%d-%d)" % (
            str(self.worth_loss_type),
            str(self.weight_norm_type),
            self.name,
            in_size,
            weight_shape[0],
            weight_shape[1],
            out_size
        )

        return repr_str


class WorthLinear(nn.Module):
    def __init__(self, worth_weight, bias=True):
        super().__init__()
        self.worth_weight = worth_weight
        self.use_bias = bias

        if bias:
            self.bias = nn.Parameter(torch.Tensor(worth_weight.output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @property
    def weight(self):
        return self.worth_weight.weight.t()

    def reset_parameters(self):
        kaiming_uniform_(self.worth_weight.reduced_weight, a=math.sqrt(5), mode="fan_out")
        if self.bias is not None:
            _, fan_in = _calculate_fan_in_and_fan_out(self.worth_weight.reduced_weight)
            bound = 1 / math.sqrt(fan_in)
            uniform_(self.bias, -bound, bound)

        self.worth_weight.update_weight_norm()

    def forward(self, input):
        return tfunc.linear(input, self.weight, self.bias)

    def __repr__(self):
        if self.use_bias:
            repr_str = "WorthLinear(W:%s, B:%s)" %(str(self.worth_weight), str(list(self.bias.shape)))
        else:
            repr_str = "WorthLinear(W:%s, B:None)" % (str(self.worth_weight))

        return repr_str


class OutOfSyncException(Exception):
    """ Raised when distributed training went out of sync."""
    pass
