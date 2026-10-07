"""ACDC dataset loader.

Ported from ALHVR's dataloaders/dataset.py (BaseDataSets, RandomGenerator)
so training code doesn't depend on the reference clone. `random_rot_flip`,
`random_rotate` and `RandomGenerator` are line-for-line identical to the
original.

Deliberate deviations, both forced:

1. `patients_to_slices` drops the original's `dataset` argument. The
   original (in train_ALHVR_acdc.py) dispatches on the root path:

       if "ACDC" in dataset:      ref_dict = {...ACDC...}
       elif "Prostate":           ref_dict = {...Prostate...}

   `elif "Prostate":` is a truthy string constant, not a comparison, so
   any root path *without* "ACDC" in it silently selects the Prostate
   table. Our root path is `data/acdc_processed/data`, so a verbatim port
   would return the Prostate dict and raise KeyError on "7". The ACDC
   table itself is copied verbatim and independently verified against
   our own train_slices.list.

   The check is case-sensitive: `acdc_processed` does not match "ACDC",
   though `ACDC_processed` would. We drop the dispatch entirely, so our
   behaviour never depends on the path spelling.

2. The split -> list-file mapping is generalised to `f"{split}.list"`.
   The original hardcodes only 'train' and 'val' (test.list is read
   separately by test_acdc.py); ours also serves the 'test' split so
   evaluate_acdc.py can reuse this class. Behaviour for 'train'/'val'
   is unchanged.
"""
import itertools
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

# Hardcoded in the official repo, not computed - verified against
# train_slices.list.
PATIENTS_TO_SLICES = {
    "3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312,
}


def patients_to_slices(num_patients):
    return PATIENTS_TO_SLICES[str(num_patients)]


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class Compose:
    """Chain sample transforms, replacing torchvision.transforms.Compose.

    Avoids a torchvision dependency: some cluster PyTorch installations
    ship torch without torchvision, and installing it separately can pull
    in a different torch build. Every call site wraps exactly one
    transform, so this is a behaviourally identical drop-in.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomGenerator:
    """Augment then resize to output_size via nearest-neighbor zoom.

    Raw ACDC slices come in 29 distinct shapes - this zoom is what
    normalizes that away, not a crop.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        return {"image": image, "label": label}


class ACDCDataset(Dataset):
    """Port of ALHVR's BaseDataSets.

    Train split reads per-slice .h5 from slices/ (via train_slices.list);
    val/test read per-volume .h5 from the root directly.
    """

    def __init__(self, base_dir, split="train", num=None, transform=None):
        self._base_dir = str(base_dir)
        self.split = split
        self.transform = transform

        list_name = "train_slices.list" if split == "train" else f"{split}.list"
        with open(f"{self._base_dir}/data_list/{list_name}") as f:
            self.sample_list = [line.strip() for line in f if line.strip()]

        if num is not None and split == "train":
            self.sample_list = self.sample_list[:num]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(f"{self._base_dir}/slices/{case}.h5", "r")
        else:
            h5f = h5py.File(f"{self._base_dir}/{case}.h5", "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices, ported from ALHVR's dataloaders/dataset.py.

    Each yielded batch is `primary_batch + secondary_batch`, i.e. labeled
    indices FIRST then unlabeled - which is what makes `batch[:labeled_bs]`
    the labeled portion and `batch[labeled_bs:]` the unlabeled portion in
    the training loop. An 'epoch' is one pass over the primary (labeled)
    indices; the secondary (unlabeled) indices cycle forever.

    Note train_ALHVR_acdc.py constructs it as
        TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                              batch_size, batch_size - labeled_bs)
    so with batch_size=16, labeled_bs=8 the split is 8 labeled + 8 unlabeled.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
