import os
from collections import defaultdict

import torch


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def key_not_found(key):
    # print("[W] The requested key \"%s\" is not found in the dictionary. Returning None instead." %key)
    raise ValueError("The requested key \"%s\" is not found in the dictionary. Returning None instead." % key)
    # return None


BASE_PARAMS = keydefaultdict(
    key_not_found,  # Set default value to None.

    device="cuda:0" if torch.cuda.is_available() else "cpu",
    # dtype="float32",
    batch_size=256,
    train_epochs=500,

    num_parallel_calls=os.cpu_count(),

    ffnn_widths=[50, 10, 50],

    batch_norm=False,

    layer_norm=False,

    drop_rate=0.0,
    dropout_type="linear",
    do_after_ci=False,
    use_bias=True,

    # ort_norm=False,
    center_inputs=False,
    ort_loss_a0=False,
    ort_loss_a1=False,
    ort_loss_weight=1.0,
    worth_sample_rate=1.0,
    ort_alpha=0.5,
    ort_alpha_method="static",
    ort_norm_type="l2",
    ort_weighting_method="fan_avg", # one of fan_avg, fan_prod, uniform
    reduce_weights=False,
    reduction_cos_th=0.995,
    reduction_euc_th=1e+9,
    symmetric_reduction=True,
    remove_longer=True,
    target_comp_rate=0.0,
    steps_per_opt=5,
    cos_threshold=0.3,
    output_norm_type=None,

    weight_norm_type="l2_sum",

    # weight_norm=False,
    # weight_loss=False,

    # weight_norm_axis=None,

    center_weight_axis=None,
    cw_loss_weight=1.0,

    length_norm_axis=None,
    ln_loss_weight=1.0,

    l2_regularization_weight=1.0
)