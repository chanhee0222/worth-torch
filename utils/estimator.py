import datetime
import logging
import os
import time
from abc import ABC

import numpy as np
import torch
from ignite import engine
from ignite.engine import Events
from ignite.metrics import Loss
from tensorboardX import SummaryWriter
from torch import optim
import torch.nn as nn

from utils.global_step_manager import set_global_step
from utils.summary_helper import SummaryBuffer

print(os.environ["PYTHONPATH"])

from models.global_worth_manager import get_worth_manager


class Estimator:
    def __init__(self,
                 model,
                 params,
                 config,
                 eval_data_iter=None,
                 optimizer="adam",
                 grad_clip_norm=5.0,
                 grad_noise_weight=0.01
                 ):
        self._logger = logging.getLogger(__name__)

        self.model = model
        self.params = params
        self.config = config
        self.device = config.device
        self.model_dir = config.model_dir
        self.metrics = {}

        self._optimizer_str = optimizer
        self.grad_clip_norm = grad_clip_norm
        self.grad_noise_weight = grad_noise_weight

        self.train_engine = engine.Engine(self._update_fn)
        self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, self._print_eta_handler)

        if config.save_summary_steps > 0:
            # Summary buffer and tensorboardX writer.
            self._summary_writer = SummaryWriter(config.model_dir)
            self.summary = SummaryBuffer(self._summary_writer)

            # Try to attach summary writer to the model.
            try:
                model.attach_summary_writer(self.summary)
                get_worth_manager().attach_summary_writer(self.summary)
                self.train_engine.add_event_handler(
                    Events.ITERATION_COMPLETED,
                    self.summary.writing_handler,
                    config.save_summary_steps
                )
            except Exception as e:
                print(e)
                self._logger.warning("Can't attach summary writer to this model. Subclass EstimatableModel to access the "
                                     "summary writer.")

        else:
            self._summary_writer = None

        # Set to True after writing graph information summary.
        self._graph_written = False

        if config.evaluate_steps != 0 and eval_data_iter is None:
            raise ValueError("eval_data_iter should be provided for config.evaluate_steps != 0")

        self.eval_data_iter = eval_data_iter

        if config.evaluate_steps == EstimatorConfig.AFTER_EACH_EPOCH:
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, self._eval_handler)
            self._eval_summary_writer = SummaryWriter(os.path.join(config.model_dir, "eval"))
            self.eval_summary = SummaryBuffer(self._eval_summary_writer)
        elif config.evaluate_steps > 0:
            self.train_engine.add_event_handler(
                Events.ITERATION_COMPLETED,
                self._eval_handler,
                config.evaluate_steps
            )
            self._eval_summary_writer = SummaryWriter(os.path.join(config.model_dir, "eval"))
            self.eval_summary = SummaryBuffer(self._eval_summary_writer)

        self.add_metric("loss", Loss(model.compute_loss))
        self.add_metric("xentropy_loss", Loss(model.loss_fn))

        self._reload_eval_engine()
        self._built = False

    @staticmethod
    def _prepare_batch(batch):
        inputs, targets = batch
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        return inputs, targets

    def _trigger_build(self, inputs, targets, engine):
        # Trigger the model build by calculating the loss once.
        with torch.no_grad():
            worth_manager = get_worth_manager()
            logits = self.model(inputs, engine=engine)
            loss = self.model.compute_loss(logits, targets)
            _ = loss.item()

            if self._optimizer_str == "adam":
                self.optimizer = optim.Adam(self.model.parameters())
                worth_manager.optimizer = self.optimizer
            else:
                raise ValueError("Unknown optimizer: %s" % self._optimizer_str)

            # Move the parameters to target device.
            self.model = self.model.to(self.device)

            self.optimizer.zero_grad()

        self.model.register_forward_pre_hook(worth_manager.clear_caches_hook)
        self._built = True
        self._logger.info("Model is built.")
        num_params = sum(p.numel() for p in self.model.parameters())
        num_t_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._logger.info('Num. model params: {:,} (num. trained: {:,})'.format(
            num_params, num_t_params))
        for idx, param in enumerate(self.model.parameters()):
            self._logger.debug(f"Param {idx}: shape={param.shape}, "
                               f"numel={param.numel()}, "
                               f"trainable={str(param.requires_grad).lower()}")

    def _update_fn(self, engine, batch):
        inputs, targets = self._prepare_batch(batch)

        if not self._built:
            self._trigger_build(inputs, targets, engine)

        self.model.train()

        if self.params["reduce_weights"]:
            changed = self.model.optimize_weight(engine)
            if changed:
                num_params = sum(p.numel() for p in self.model.parameters())
                num_t_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                self._logger.info('Num. model params: {:,} (num. trained: {:,})'.format(
                    num_params, num_t_params))

            # if changed:
            #     self.optimizer = optim.Adam(self.model.parameters())
            #     worth_manager = get_worth_manager()
            #     worth_manager.optimizer = self.optimizer
                # for param in self.optimizer.param_groups[0]["params"]:
                #     param.grad = param.data.new_zeros(param.data.shape)
                #     print(param.shape)
                #     print(param.grad)
                # print(self.model)

        self.optimizer.zero_grad()
        logits = self.model(inputs, engine=engine)

        loss = self.model.compute_loss(logits, targets)

        loss.backward()

        # Clip gradient by global norm.
        if self.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        # Add gradient noise.
        stddev = self.grad_noise_weight / np.float_power(1 + engine.state.iteration, 0.55)
        self.summary.add_scalar("noise_stddev", stddev)
        if self.grad_noise_weight > 0.0:
            for param in self.model.parameters():
                if param.grad is None:
                    continue

                noise = param.new_empty(param.shape).normal_(0, stddev)
                param.grad.data.add_(noise)

        self.optimizer.step()
        # worth_manager = get_worth_manager()
        # worth_manager.clear_caches()

        set_global_step(engine.state.iteration)

        self.summary.add_scalar("loss", loss.item())

        # if self._summary_writer is not None and not self._graph_written:
        #     self._summary_writer.add_graph(self.model, inputs)
        #     self._graph_written = True

    def _eval_fn(self, engine, batch):
        self.model.eval()

        with torch.no_grad():
            inputs, targets = self._prepare_batch(batch)
            pred = self.model(inputs, engine=engine)
            return pred, targets

    def _eval_handler(self, engine, n_steps=None):
        if (n_steps is None) or (engine.state.iteration % n_steps == 0):
            self._logger.info("Evaluating model: %s" %self.params["model_dir"])
            self.evaluate(self.eval_data_iter)

            # Log evaluation results to tensorboard.
            metrics = self.eval_engine.state.metrics
            for name in self.metrics.keys():
                self.eval_summary.add_scalar(name, metrics[name])

            self.eval_summary.write(engine.state.iteration)

    def _print_eta_handler(self, engine):
        epoch = engine.state.epoch
        remaining_epochs = self._train_epochs - epoch
        time_delta = time.time() - self._start_time
        eta = datetime.timedelta(seconds=round(time_delta / epoch * remaining_epochs))
        self._logger.info("%d more epochs to go, ETA %s." % (remaining_epochs, str(eta)))

    def add_metric(self, name, metric_fn):
        self.metrics[name] = metric_fn
        self._reload_eval_engine()

    def _reload_eval_engine(self):
        self.eval_engine = engine.Engine(self._eval_fn)

        if len(self.metrics) > 0:
            for name, metric in self.metrics.items():
                metric.attach(self.eval_engine, name)

    def train(self, data_iter, epochs=1):
        self.model.train()
        self._train_epochs = epochs
        self._start_time = time.time()
        self.train_engine.run(data_iter, epochs)

    def predict(self, data_iter):
        self.model.eval()

    def evaluate(self, data_iter):
        self.eval_engine.run(data_iter)
        metrics = self.eval_engine.state.metrics
        names = sorted(self.metrics.keys())
        eval_msgs = []
        for name in names:
            eval_msgs.append("%s=%s" %(name, str(metrics[name])))

        self._logger.info("[Step %d] Validation Results: " %self.train_engine.state.iteration + ", ".join(eval_msgs))


class EstimatorConfig:
    AFTER_EACH_EPOCH = object()

    def __init__(self,
                 model_dir=None,
                 save_summary_steps=100,
                 evaluate_steps=AFTER_EACH_EPOCH,
                 save_checkpoints_steps=AFTER_EACH_EPOCH,
                 keep_checkpoint_max=5,
                 keep_checkpoint_every_n_hours=10000,
                 device=None,
                 ):

        self.model_dir = model_dir
        self.save_summary_steps = save_summary_steps
        self.evaluate_steps = evaluate_steps
        self.save_checkpoints_steps = save_checkpoints_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self.device = device


class Estimatable(ABC):
    def compute_loss(self, logits, targets):
        raise NotImplementedError
