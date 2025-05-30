import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import (
    BertTokenizer,
    AutoTokenizer,
    DistilBertTokenizer,
    GPT2Tokenizer,
)
from torchvision import datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # vision
    "CMNIST",
    "CelebA",
    "MetaShift",
    "OfficeHome",
    # language
    "CivilComments",
    "MultiNLI",
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    # Normalize to ensure case-insensitive matching
    normalized_name = dataset_name.lower()
    # Map dataset names to their corresponding classes
    dataset_classes = {name.lower(): globals().get(name) for name in DATASETS}
    # Check if the normalized name exists in the dataset map
    if (
        normalized_name not in dataset_classes
        or dataset_classes[normalized_name] is None
    ):
        raise NotImplementedError(
            f"Dataset not found or not implemented: {dataset_name}"
        )
    # Return the corresponding dataset class
    return dataset_classes[normalized_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class SubpopDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    INPUT_SHAPE = None  # Subclasses should override
    SPLITS = {"tr": 0, "va": 1, "te": 2}  # Default, subclasses may override
    EVAL_SPLITS = ["te"]  # Default, subclasses may override

    def __init__(
        self,
        root,
        split,
        metadata,
        transform,
        train_attr="yes",
        subsample_type=None,
        duplicates=None,
    ):
        df = pd.read_csv(metadata)
        df = df[df["split"] == (self.SPLITS[split])]

        self.idx = list(range(len(df)))
        self.x = (
            df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        )
        self.y = df["y"].tolist()
        self.a = (
            df["a"].tolist() if train_attr == "yes" else [0] * len(df["a"].tolist())
        )
        self.transform_ = transform
        self._count_groups()

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.num_attributes = len(set(self.a))
        self.num_labels = len(set(self.y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels

        for i in self.idx:
            self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.idx:
            self.weights_g.append(
                len(self)
                / self.group_sizes[self.num_attributes * self.y[i] + self.a[i]]
            )
            self.weights_y.append(len(self) / self.class_sizes[self.y[i]])

    def subsample(self, subsample_type, max_size=None):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = (
            min(list(self.group_sizes))
            if subsample_type == "group"
            else min(list(self.class_sizes))
        )

        counts_g = [0] * self.num_attributes * self.num_labels
        counts_y = [0] * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if (
                subsample_type == "group"
                and counts_g[self.num_attributes * int(y) + int(a)] < min_size
            ) or (subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.num_attributes * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        if max_size is not None:
            # random uniform subsample
            self.idx = np.random.choice(self.idx, max_size, replace=False).tolist()
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        return i, x, y, a

    def __len__(self):
        return len(self.idx)


class CMNIST(SubpopDataset):
    data_type = "images"

    def __init__(
        self,
        data_path,
        split,
        hparams,
        train_attr="yes",
        subsample_type=None,
        duplicates=None,
        downsample_pixel=True,
    ):
        root = Path(data_path) / "cmnist"
        mnist = datasets.MNIST(root, train=True)
        X, y = mnist.data, mnist.targets

        if split == "tr":
            X, y = X[:30000], y[:30000]
        elif split == "va":
            X, y = X[30000:40000], y[30000:40000]
        elif split == "te":
            X, y = X[40000:], y[40000:]
        else:
            raise NotImplementedError

        rng = np.random.default_rng(hparams["data_seed"])

        self.binary_label = np.bitwise_xor(
            y >= 5, (rng.random(len(y)) < hparams["cmnist_flip_prob"])
        ).numpy()
        self.color = np.bitwise_xor(self.binary_label, (rng.random(len(y)) < 0.5))
        self.imgs = torch.stack([X, X, torch.zeros_like(X)], dim=1).numpy()
        self.imgs[list(range(len(self.imgs))), (1 - self.color), :, :] *= 0

        if split == "tr":
            group_matrix = np.array(
                [
                    [1, 0, 0, 1],
                    [1, 0, 1, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 1],
                ]
            )
            b = [
                hparams["cmnist_spur_prob"] * hparams["data_size"],
                hparams["cmnist_attr_prob"] * hparams["data_size"],
                hparams["cmnist_label_prob"] * hparams["data_size"],
                hparams["data_size"],
            ]
            num_per_group = np.linalg.solve(group_matrix, b).astype(int)
            print("number per group to be sampled:", num_per_group)

            # count group
            group_idx = [[], [], [], []]
            for i in range(len(self.color)):
                group_idx[self.binary_label[i] * 2 + self.color[i]].append(i)
            # subsample to the given number
            for i, n in enumerate(num_per_group):
                group_idx[i] = np.random.choice(group_idx[i], n, replace=False)
            group_idx = np.concatenate(group_idx)
            self.color = self.color[group_idx]
            self.binary_label = self.binary_label[group_idx]
            self.imgs = self.imgs[group_idx]

        self.idx = list(range(len(self.color)))
        self.x = torch.from_numpy(self.imgs).float() / 255.0
        if downsample_pixel:
            self.x = self.x[:, :, ::2, ::2]
        self.y = self.binary_label
        self.a = self.color

        self.transform_ = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize((0.1307, 0.1307, 0.0), (0.3081, 0.3081, 0.3081)),
            ]
        )
        self._count_groups()

        print("group size: ", self.group_sizes)
        print("class size: ", self.class_sizes)
        print("Total size:", sum(self.group_sizes))

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

    def _subsample(self, mask, n_samples, rng):
        assert n_samples <= mask.sum()
        idxs = np.concatenate(
            (
                np.nonzero(~mask)[0],
                rng.choice(np.nonzero(mask)[0], size=n_samples, replace=False),
            )
        )
        rng.shuffle(idxs)
        self.imgs = self.imgs[idxs]
        self.color = self.color[idxs]
        self.binary_label = self.binary_label[idxs]

    def transform(self, x):
        return self.transform_(x)


class MetaShift(SubpopDataset):

    def __init__(
        self,
        data_path,
        split,
        hparams,
        train_attr="yes",
        subsample_type=None,
        duplicates=None,
    ):
        root = os.path.join(data_path, "metashift", "COCO-Cat-Dog-indoor-outdoor")

        df = pd.read_csv(hparams["metadata"])
        df = df[df["split"] == (self.SPLITS[split])]
        print("metadata example: ", df.head())

        self.idx = list(range(len(df)))
        self.x = (
            df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        )
        self.y = df["y"].to_numpy()
        self.a = (
            df["a"].to_numpy() if train_attr == "yes" else [0] * len(df["a"].to_numpy())
        )

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.transform_ = transform

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

        self._count_groups()
        print("group size: ", self.group_sizes)
        print("class size: ", self.class_sizes)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class OfficeHome(SubpopDataset):

    def __init__(
        self,
        data_path,
        split,
        hparams,
        train_attr="yes",
        subsample_type=None,
        duplicates=None,
    ):
        root = os.path.join(data_path, "officehome")
        transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.data_type = "images"

        ######################
        df = pd.read_csv(hparams["metadata"])
        df = df[df["split"] == (self.SPLITS[split])]
        print("metadata example: ", df.head())

        self.idx = list(range(len(df)))

        self.x = (
            df["filename"]
            .astype(str)
            .map(lambda x: os.path.join(root, x))
            .to_numpy()
        )
        self.y = df["y"].to_numpy()
        self.a = (
            df["a"].to_numpy() if train_attr == "yes" else [0] * len(df["a"].to_numpy())
        )
        self.transform_ = transform

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

        ######################
        self._count_groups()
        print("group size: ", self.group_sizes)
        print("class size: ", self.class_sizes)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class CelebA(SubpopDataset):

    def __init__(
        self,
        data_path,
        split,
        hparams,
        train_attr="yes",
        subsample_type=None,
        duplicates=None,
    ):
        root = os.path.join(data_path, "CelebFaces", "img_align_celeba")
        transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.data_type = "images"

        ######################
        df = pd.read_csv(hparams["metadata"])
        df = df[df["split"] == (self.SPLITS[split])]
        print("metadata example: ", df.head())

        self.idx = list(range(len(df)))
        # Need to adapt to specific file structure to improve I/O performance
        self.x = (
            df["filename"]
            .astype(str)
            .map(lambda x: os.path.join(root, x[:3], x))
            .to_numpy()
        )
        self.y = df["y"].to_numpy()
        self.a = (
            df["a"].to_numpy() if train_attr == "yes" else [0] * len(df["a"].to_numpy())
        )
        self.transform_ = transform

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

        ######################
        self._count_groups()
        print("group size: ", self.group_sizes)
        print("class size: ", self.class_sizes)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class MultiNLI(SubpopDataset):

    def __init__(
        self,
        data_path,
        split,
        hparams,
        train_attr="yes",
        subsample_type=None,
        duplicates=None,
    ):
        root = os.path.join(data_path, "multinli")
        metadata = hparams["metadata"]
        text = pd.read_csv(os.path.join(root, "multinli_sentence2.csv"))

        # self.features_array = []
        self.arch = hparams["arch"]
        self.x_array = list(text["sentence2"])
        if self.arch == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.arch == "bert-ft":
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
        elif self.arch == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise NotImplementedError
        self.data_type = "text"
        super().__init__(
            "", split, metadata, self.transform, train_attr, subsample_type, duplicates
        )

    def transform(self, i):
        text = self.x_array[int(i)]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,  # TODO: potentially can be smaller
            return_tensors="pt",
        )

        if len(tokens) == 3:
            return torch.squeeze(
                torch.stack(
                    (
                        tokens["input_ids"],
                        tokens["attention_mask"],
                        tokens["token_type_ids"],
                    ),
                    dim=2,
                ),
                dim=0,
            )
        else:
            return torch.squeeze(
                torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2),
                dim=0,
            )


class CivilComments(SubpopDataset):

    def __init__(
        self,
        data_path,
        split,
        hparams,
        train_attr="yes",
        subsample_type=None,
        duplicates=None,
    ):
        text = pd.read_csv(
            os.path.join(data_path, "civilcomments/civilcomments_text.csv")
        )
        metadata = hparams["metadata"]

        self.text_array = list(text["comment_text"])
        if hparams["arch"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif hparams["arch"] == "bert-ft":
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
        elif hparams["arch"] == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise NotImplementedError
        self.data_type = "text"
        super().__init__(
            "", split, metadata, self.transform, train_attr, subsample_type, duplicates
        )

    def transform(self, i):
        text = self.text_array[int(i)]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        if len(tokens) == 3:
            return torch.squeeze(
                torch.stack(
                    (
                        tokens["input_ids"],
                        tokens["attention_mask"],
                        tokens["token_type_ids"],
                    ),
                    dim=2,
                ),
                dim=0,
            )
        else:
            return torch.squeeze(
                torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2),
                dim=0,
            )

