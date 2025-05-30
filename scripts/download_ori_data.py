# Adapted from https://github.com/YyzHarry/SubpopBench/blob/main/subpopbench/scripts/download.py

import argparse
import os
import json
import string
import tarfile
import logging
import gdown
import pandas as pd
import numpy as np
from pathlib import Path
from zipfile import ZipFile


logging.basicConfig(level=logging.INFO)


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_datasets(data_path, datasets):
    os.makedirs(data_path, exist_ok=True)
    dataset_downloaders = {
        "celeba": download_celeba,
        "metashift": download_metashift,
        "officehome": download_officehome,
        "cmnist": download_cmnist,
        "civilcomments": download_civilcomments,
        "multinli": download_multinli,
    }
    for dataset in datasets:
        if dataset in dataset_downloaders:
            dataset_downloaders[dataset](data_path)
        else:
            no_downloader(dataset)


def no_downloader(dataset):
    print(
        f"Dataset {dataset} cannot be automatically downloaded. Please check the repo for download instructions."
    )


def download_celeba(data_path):
    logging.info("Downloading CelebA...")
    celeba_dir = os.path.join(data_path, "celeba")
    os.makedirs(celeba_dir, exist_ok=True)
    download_and_extract(
        "https://drive.google.com/uc?id=1mb1R6dXfWbvk3DnlWOBO8pDeoBKOcLE6",
        os.path.join(celeba_dir, "img_align_celeba.zip"),
    )
    download_and_extract(
        "https://drive.google.com/uc?id=1acn0-nE4W7Wa17sIkKB0GtfW4Z41CMFB",
        os.path.join(celeba_dir, "list_eval_partition.txt"),
        remove=False,
    )
    download_and_extract(
        "https://drive.google.com/uc?id=11um21kRUuaUNoMl59TCe2fb01FNjqNms",
        os.path.join(celeba_dir, "list_attr_celeba.txt"),
        remove=False,
    )

def download_metashift(data_path):
    logging.info("Downloading MetaShift Cats vs. Dogs...")
    ms_dir = os.path.join(data_path, "metashift")
    os.makedirs(ms_dir, exist_ok=True)
    download_and_extract(
        "https://www.dropbox.com/s/a7k65rlj4ownyr2/metashift.tar.gz?dl=1",
        os.path.join(ms_dir, "metashift.tar.gz"),
        remove=True,
    )


def download_officehome(data_path):
    logging.info("Please download OfficeHome manually at https://www.hemanthdv.org/officeHomeDataset.html")


def download_cmnist(data_path):
    from torchvision import datasets

    sub_dir = Path(data_path) / "cmnist"
    train_mnist = datasets.mnist.MNIST(sub_dir, train=True, download=True)
    test_mnist = datasets.mnist.MNIST(sub_dir, train=False, download=True)


def download_civilcomments(data_path):
    logging.info("Downloading CivilComments...")
    civilcomments_dir = os.path.join(data_path, "civilcomments")
    os.makedirs(civilcomments_dir, exist_ok=True)
    download_and_extract(
        "https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/",
        os.path.join(civilcomments_dir, "civilcomments.tar.gz"),
    )
    download_and_extract(
        "https://worksheets.codalab.org/rest/bundles/0x17807ae09e364ec3b2680d71ca3d9623/contents/blob/",
        os.path.join(civilcomments_dir, "bert_finetuned.tar.gz"),
    )


def download_multinli(data_path):
    logging.info("Downloading MultiNLI...")
    multinli_dir = os.path.join(data_path, "multinli")
    os.makedirs(multinli_dir, exist_ok=True)
    # download original data to extract the text
    download_and_extract(
        "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip",
        os.path.join(multinli_dir, "multinli_1.0.zip"),
        remove=True,
    )


def generate_metadata(data_path, datasets):
    dataset_metadata_generators = {
        "celeba": generate_metadata_celeba,
        "civilcomments": generate_metadata_civilcomments,
        "multinli": generate_metadata_multinli,
        "metashift": generate_metadata_metashift,
        "cmnist": generate_metadata_cmnist,
        "officehome": generate_metadata_officehome,
    }
    for dataset in datasets:
        dataset_metadata_generators[dataset](data_path)

def generate_metadata_officehome(data_path):
    logging.info("Generating metadata for OfficeHome...")
    raise NotImplementedError("Download manually is easier.")

def generate_metadata_celeba(data_path):
    logging.info("Generating metadata for CelebA...")
    with open(os.path.join(data_path, "celeba/list_eval_partition.txt"), "r") as f:
        splits = f.readlines()

    with open(os.path.join(data_path, "celeba/list_attr_celeba.txt"), "r") as f:
        attrs = f.readlines()[2:]

    f = open(os.path.join(data_path, "celeba", "metadata_celeba.csv"), "w")
    f.write("id,filename,split,y,a\n")

    for i, (split, attr) in enumerate(zip(splits, attrs)):
        fi, si = split.strip().split()
        ai = attr.strip().split()[1:]
        yi = 1 if ai[9] == "1" else 0
        gi = 1 if ai[20] == "1" else 0
        f.write("{},{},{},{},{}\n".format(i + 1, fi, si, yi, gi))

    f.close()


def generate_metadata_civilcomments(data_path):
    logging.info("Generating metadata for CivilComments...")
    df = pd.read_csv(
        os.path.join(data_path, "civilcomments", "all_data_with_identities.csv"),
        index_col=0,
    )
    text = df["comment_text"]
    # save text
    text.to_csv(
        os.path.join(data_path, "civilcomments", f"civilcomments_text.csv"),
        index=False,
    )


def generate_metadata_multinli(data_path):
    # https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_multinli.py#L91
    logging.info("Generating metadata for MultiNLI...")
    def tokenize(s):
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = s.lower()
        s = s.split(" ")
        return s

    train_df = pd.read_json(
        os.path.join(data_path, "multinli/multinli_1.0", "multinli_1.0_train.jsonl"),
        lines=True,
    )
    val_df = pd.read_json(
        os.path.join(
            data_path, "multinli/multinli_1.0", "multinli_1.0_dev_matched.jsonl"
        ),
        lines=True,
    )
    test_df = pd.read_json(
        os.path.join(
            data_path, "multinli/multinli_1.0", "multinli_1.0_dev_mismatched.jsonl"
        ),
        lines=True,
    )
    split_dict = {"train": 0, "val": 1, "test": 2}
    train_df["split"] = split_dict["train"]
    val_df["split"] = split_dict["val"]
    test_df["split"] = split_dict["test"]
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    ### Assign labels
    df = df.loc[df["gold_label"] != "-", :]
    print(f"Total number of examples: {len(df)}")
    for k, v in split_dict.items():
        print(k, np.mean(df["split"] == v))
    label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    for k, v in label_dict.items():
        idx = df.loc[:, "gold_label"] == k
        df.loc[idx, "gold_label"] = v

    ### Assign spurious attribute (negation words)
    negation_words = [
        "nobody",
        "no",
        "never",
        "nothing",
    ]  # Taken from https://arxiv.org/pdf/1803.02324.pdf

    df["sentence2_has_negation"] = [False] * len(df)

    for negation_word in negation_words:
        df["sentence2_has_negation"] |= [
            negation_word in tokenize(sentence) for sentence in df["sentence2"]
        ]

    df["sentence2_has_negation"] = df["sentence2_has_negation"].astype(int)
    # filter out neutral
    df = df.loc[df["gold_label"] != 2, :]
    # filter out nan samples
    df = df.loc[df["sentence2"] != "n/a", :].reset_index(drop=True)

    text = df["sentence2"]

    # save text
    text.to_csv(
        os.path.join(data_path, "multinli", f"multinli_sentence2.csv"),
    )

    ## Write to disk
    df = df[["gold_label", "sentence2_has_negation", "split"]]
    df.to_csv(os.path.join(data_path, "multinli", "metadata_2classes.csv"))


def generate_metadata_metashift(data_path, test_pct=0.25, val_pct=0.1):
    logging.info("Generating metadata for MetaShift...")
    dirs = {
        "train/cat/cat(indoor)": [1, 1],
        "train/dog/dog(outdoor)": [0, 0],
        "test/cat/cat(outdoor)": [1, 0],
        "test/dog/dog(indoor)": [0, 1],
    }
    ms_dir = os.path.join(data_path, "metashift")

    all_data = []
    for dir in dirs:
        folder_path = os.path.join(ms_dir, "MetaShift-Cat-Dog-indoor-outdoor", dir)
        y = dirs[dir][0]
        g = dirs[dir][1]
        for img_path in Path(folder_path).glob("*.jpg"):
            all_data.append({"filename": img_path, "y": y, "a": g})
    df = pd.DataFrame(all_data)

    rng = np.random.RandomState(42)

    test_idxs = rng.choice(
        np.arange(len(df)), size=int(len(df) * test_pct), replace=False
    )
    val_idxs = rng.choice(
        np.setdiff1d(np.arange(len(df)), test_idxs),
        size=int(len(df) * val_pct),
        replace=False,
    )

    split_array = np.zeros((len(df), 1))
    split_array[val_idxs] = 1
    split_array[test_idxs] = 2

    df["split"] = split_array.astype(int)
    df.to_csv(os.path.join(ms_dir, "metadata_metashift.csv"), index=False)


def generate_task_officehome(data_path, output_path, task_y, task_a):
    logging.info("No need to generate metadata for Office Home...it is handled on-the-fly.")

def generate_metadata_cmnist(data_path):
    logging.info("No need to generate metadata for cmnist...it is handled on-the-fly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=[
            "celeba",
            "civilcomments",
            "multinli",
            "metashift",
            "cmnist",
            "officehome",
        ],
    )
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--download", action="store_true", default=False)
    args = parser.parse_args()

    if args.download:
        download_datasets(args.data_path, args.datasets)
    generate_metadata(args.data_path, args.datasets)
