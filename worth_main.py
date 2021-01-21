import json
import logging
import math
import os
import random
import time
# from datetime import datetime
import datetime

import argparse

from ignite import metrics

from datasets.fashion_mnist import FashionMNIST
from experiments.model_params import BASE_PARAMS
from models.ffnn import FFNN
from models.global_worth_manager import get_worth_manager, reset_global_worth_manager
from utils.estimator import Estimator, EstimatorConfig
from utils.logging_helper import init_logger

import torch
# torch.set_default_tensor_type(torch.FloatTensor)
print(torch.__version__)


def train_loop(params, estimator, dataset):
    logger = logging.getLogger(__name__)

    # Training loop.
    logger.info("Training for %d epochs." %params["train_epochs"])
    estimator.train(
        dataset.get_train_iterator(),
        params["train_epochs"]
    )


def run_ort(args, params, create_new_dir=True):
    reset_global_worth_manager()
    params = params.copy()
    params["model_dir"] = os.path.join(args["model_dir"], params["exp_name"])

    if create_new_dir:
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        params["model_dir"] = os.path.join(params["model_dir"], timestamp)

    os.makedirs(params["model_dir"], exist_ok=True)
    init_logger(params["model_dir"])

    logger = logging.getLogger(__name__)

    # Save model parameter settings to model directory.
    with open(os.path.join(params["model_dir"], "model_params.json"), "w") as out_fh:
        json.dump(params, out_fh, indent=2, skipkeys=True)

    logger.info("Model parameters:")
    logger.info(json.dumps(params, indent=2, sort_keys=True))

    # Make parameter objects that aren't json serializable.
    params["device"] = torch.device(params["device"])
    # dtype_dict = {
    #     "float32": torch.float32,
    #     "float64": torch.float64
    # }
    # params["dtype"] = dtype_dict[params["dtype"]]

    dataset = FashionMNIST(params)

    # Construct the estimator.
    # Config that prevents TF from allocating the whole GPU memory.
    # config = tf.ConfigProto(
    #     allow_soft_placement=True,
    #     log_device_placement=False
    # )
    # config.gpu_options.allow_growth = True

    # Calculate the number of steps between summaries, so that summaries per epoch stays the same.
    summaries_per_epoch = 3
    save_summary_steps = math.ceil(dataset.steps_per_epoch / summaries_per_epoch)
    params["total_steps"] = dataset.steps_per_epoch * params["train_epochs"]

    get_worth_manager().load_hparams(params)

    model = FFNN(params).to(params["device"])

    estimator_config = EstimatorConfig(
        model_dir=params["model_dir"],
        device=params["device"],
        save_summary_steps=save_summary_steps
        # evaluate_steps=1000
    )
    estimator = Estimator(
        model=model, params=params, config=estimator_config, eval_data_iter=dataset.get_eval_iterator())
    estimator.add_metric("accuracy", metrics.Accuracy())

    logger.info(model)

    train_loop(params, estimator, dataset)


def _make_test_params():
    print("!!!!! Using test parameters !!!!!")
    ort_base = BASE_PARAMS.copy()
    ort_base.update(
        num_parallel_calls=4,

        train_epochs=2000,
        batch_size=1024,
        ffnn_widths=[500] * 5,
        center_inputs=True,
        use_bias=False,
        ort_loss_weight=5.0,
        # output_norm_type="inf",

        reduce_weights=True,
        reduction_cos_th=0.9999,
        target_comp_rate=0.0,
        remove_longer=True,
        steps_per_opt=1,
        cos_threshold=0.3,

        drop_rate=0.2
    )

    ort = []

    # Full
    for ort_weight in [20.0]:
        exp = ort_base.copy()
        exp.update(
            description=
            """
            Testing worth loss weight.
            """,
            exp_name="test/%.1f" %ort_weight,
            ort_loss_a0=True,
            ort_loss_a1=True,
            output_norm_type=None,
            ort_weighting_method="fan_avg",  # one of fan_avg, fan_prod, uniform
            ort_alpha_method="static",
            ort_alpha=1.0,
            ort_loss_weight=ort_weight,
            ort_norm_type="min1",
            weight_norm_type="outer_l2_sum",
            l2_regularization_weight=0.1,
            reduce_weights=True,
            reduce_features=True,
            reduction_euc_th=1e-1,
            symmetric_reduction=True,
            dropout_type="linear",
            worth_sample_rate=1.0,
        )
        ort.append(exp)

    return ort


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Test")
    arg_parser.add_argument("-d", "--model_dir", dest="model_dir", required=True,
                            help="Path to store the trained model and info.")
    args = arg_parser.parse_args()
    args = vars(args)

    exps = _make_test_params()

    print("List of experiments:")
    for exp in exps:
        print("\t%s" %exp["exp_name"])

    random.shuffle(exps)

    for i in range(15):
        for exp in exps:
            run_ort(args, exp)