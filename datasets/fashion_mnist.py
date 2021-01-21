import math

import torchvision
import numpy as np
import torch as tc

from datasets.dataset_abc import DataSetABC
from experiments.model_params import BASE_PARAMS


class FashionMNIST(DataSetABC):

    def __init__(self, params):
        self.data_dir = "/tmp"

        # def _transform_fn(input):
        #     input = np.expand_dims(input, 0)
        #     # print(input.shape)
        #     return input

        self.train_dataset = CustomTVFashionMNIST(
            self.data_dir,
            train=True,
            download=True,
            # transform=_transform_fn
        )

        self.test_dataset = CustomTVFashionMNIST(
            self.data_dir,
            train=False,
            download=True,
            # transform=_transform_fn
        )

        self._num_examples = len(self.train_dataset)

        self.params = params

    def _make_dataset(self, raw_dataset, do_shuffle):

        num_parallel_calls = self.params["num_parallel_calls"]

        data_loader = tc.utils.data.DataLoader(
            raw_dataset,
            batch_size=self.params["batch_size"],
            shuffle=do_shuffle,
            num_workers=num_parallel_calls,
            pin_memory=True
        )

        return data_loader

    def get_eval_iterator(self):
        return self._make_dataset(self.test_dataset,
                                  do_shuffle=False)

    def get_train_iterator(self):
        return self._make_dataset(self.train_dataset,
                                  do_shuffle=True)

    @property
    def steps_per_epoch(self):
        return math.ceil(self.num_examples / self.params["batch_size"])

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_classes(self):
        return 10


class CustomTVFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.to(tc.float32) / 255.0
        self.data.unsqueeze_(1)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == "__main__":
    params = BASE_PARAMS
    params["batch_size"] = 128
    dataset = FashionMNIST(params)
    print(dataset.num_examples)
    print(dataset.steps_per_epoch)
