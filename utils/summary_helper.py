import time
from abc import ABC
from collections import defaultdict, Counter

import numpy as np
import torch
from torch import nn as nn

from utils.global_step_manager import get_global_step
from utils.scope import get_scope_string


class SummaryBuffer:
    def __init__(self, writer, average_scalars=True):
        self.writer = writer
        self.average_scalars = average_scalars
        self.reset()

        self._used_tags_sets = defaultdict(set)
        self._used_tags_counter = None

        # Write summaries every N steps.
        self.scalar_write_interval = 10
        self.histogram_write_interval = 100

    def reset(self):
        self.reset_scalars()
        self.reset_histograms()

        self._image = {}
        self._figure = {}
        self._audio = {}
        self._text = {}
        self._graph = {}
        self._onnx_graph = {}
        self._embedding = {}
        self._pr_curve = {}
        self._video = {}

    def reset_scalars(self):
        self._scalar = defaultdict(list)
        self._used_tags_counter = None

    def reset_histograms(self):
        self._histogram = {}
        self._used_tags_counter = None

    def _make_unique_tag(self, tag):
        if self._used_tags_counter is None:
            self._used_tags_counter = Counter()
            for val in self._used_tags_sets.values():
                self._used_tags_counter.update(val)
        return "%s_%d" %(tag, self._used_tags_counter[tag])

    def add_scalar(self, tag, val, scope=None):
        tag = get_scope_string(scope) + tag
        if isinstance(val, torch.Tensor):
            val = val.item()

        self._scalar[tag].append((val, time.time()))

    def add_histogram(self, tag, values, scope=None):
        if get_global_step() % self.histogram_write_interval == 0:
            tag = get_scope_string(scope) + tag
            self._histogram[tag] = (values, time.time())

    def add_audio(self, tag, snd_tensor, sample_rate=44100):
        raise NotImplementedError

    def write(self, global_step):
        self.write_scalars(global_step)
        self.write_histograms(global_step)

        # Empty buffer.
        self.reset()

    def write_histograms(self, global_step):
        # Write histogram summaries.
        self._used_tags_sets["histograms"].update(self._histogram.keys())
        for tag in self._histogram:
            val, walltime = self._histogram[tag]
            self.writer.add_histogram(self._make_unique_tag(tag), val, global_step, walltime=walltime)

        # Empty buffer.
        self.writer.flush()
        self.reset_histograms()

    def write_scalars(self, global_step):
        # Write scalar summaries.
        self._used_tags_sets["scalars"].update(self._scalar.keys())

        for tag in self._scalar:
            if self.average_scalars:
                val = np.mean([d[0] for d in self._scalar[tag]])
            else:
                val = self._scalar[tag][-1][0]

            # if tag.find("worth") >= 0:
            #     print(tag, val, global_step)
            walltime = self._scalar[tag][-1][1]
            self.writer.add_scalar(self._make_unique_tag(tag), val, global_step, walltime)

        # Empty buffer.
        self.writer.flush()
        self.reset_scalars()

    def maybe_write_summaries(self, global_step):
        if global_step % self.scalar_write_interval == 0:
            self.write_scalars(global_step)

        if len(self._histogram) > 0:
            self.write_histograms(global_step)

    def writing_handler(self, engine, n_steps):
        if engine.state.iteration % n_steps == 0:
            self.write(engine.state.iteration)


class _DummySummaryWriter(SummaryBuffer):
    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_audio(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_pr_curve(self, *args, **kwargs):
        pass

    def add_embedding(self, *args, **kwargs):
        pass

    def export_scalars_to_json(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def write(self, global_step):
        print("Warning: No summary writer is attached, but trying to write summary.")


class SummarizableModule(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self._summary = _DummySummaryWriter(None)

    @property
    def summary(self):
        return self._summary

    def attach_summary_writer(self, writer):
        # Recursively look for SummarizableModule instances, and attach summary writer to them.

        if isinstance(self._summary, _DummySummaryWriter):
            # Prevent loop in recursion.
            self._summary = writer

            for module in self.modules():
                if issubclass(module.__class__, SummarizableModule):
                    module.attach_summary_writer(writer)