import numpy as np
from torch.utils.data.sampler import Sampler


def create_pids2idxs(data_source):
    """Creates a mapping between pids and indexes of images for that pid.
    Returns:
        2D List with pids => idx
    """
    pid2imgs = {}
    for idx, (img, target) in enumerate(data_source.imgs):
        if target not in pid2imgs:
            pid2imgs[target] = [idx]
        else:
            pid2imgs[target].append(idx)
    return list(pid2imgs.values())


class TripletSampler(Sampler):
    def __init__(self, config, data_set, drop_last=True):
        self.data_set = data_set
        self.P = config["P"]
        self.K = config["K"]
        self.batch_size = self.P * self.K
        self.pid2idx = create_pids2idxs(self.data_set)
        self.drop_last = drop_last

    def __len__(self):
        num_batches = len(self.pid2idx) / self.P
        if self.drop_last:
            return num_batches
        return num_batches + 1

    def __iter__(self):
        P_perm = np.random.permutation(len(self.pid2idx))
        batch = []
        for p in P_perm:
            imgs_from_pid = self.pid2idx[p]
            imgs_from_pid = np.random.choice(
                imgs_from_pid,
                size=self.K,
                replace=(len(imgs_from_pid) < self.K))

            batch.extend(imgs_from_pid)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 1 and not self.drop_last:
            yield batch