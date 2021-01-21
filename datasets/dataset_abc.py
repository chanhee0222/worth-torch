import abc


class DataSetABC(abc.ABC):
    def get_train_iterator(self):
        raise NotImplementedError

    def get_eval_iterator(self):
        raise NotImplementedError

    @property
    def steps_per_epoch(self):
        raise NotImplementedError

    @property
    def num_examples(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError