import os
import umap
import torch
import argparse
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import datasets
import utils
from datasets import get_dataset_class
from models import get_pretrained_model


def get_embeddings(
    data_name, data_dir, output_dir, arch, y, a, n, sc, ci, ai, split, seed
):
    model = get_pretrained_model(arch, data_name, data_dir)

    hparams = {}
    hparams["metadata"] = os.path.join(output_dir, f"metadata.csv")
    hparams["arch"] = arch

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if data_name == "cmnist":
        hparams["data_seed"] = seed
        hparams["cmnist_flip_prob"] = 0
        hparams["cmnist_spur_prob"] = sc
        hparams["cmnist_attr_prob"] = ai
        hparams["cmnist_label_prob"] = ci
        hparams["data_size"] = n
        dataset = datasets.CMNIST(data_dir, split, hparams, train_attr="yes")
    else:
        dataset = get_dataset_class(data_name)(
            data_dir, split, hparams, train_attr="yes"
        )

    finite_loader = torch.utils.data.DataLoader(
        dataset,
        drop_last=False,
        batch_size=64,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    activ, ys, attrs = [], [], []
    model.eval()
    with torch.no_grad():
        for _, x, y, a in finite_loader:
            feat = model(x.to(device))
            activ.append(feat.detach().cpu().numpy())
            ys.append(y)
            attrs.append(a)
    activ = np.concatenate(activ, axis=0)
    ys = np.concatenate(ys, axis=0)
    attrs = np.concatenate(attrs, axis=0)

    return activ, ys, attrs


def extract_data_desc(activ, ys, attrs):
    # dimension reduction
    reducer = umap.UMAP(n_components=2)
    activ_ = reducer.fit_transform(activ)
    normed_activ_ = StandardScaler().fit_transform(activ)
    normed_activ_ = reducer.fit_transform(normed_activ_)

    # compute distances wrt gt labels
    c_intra, c_inter = utils.infer_dist(activ_, ys)
    a_intra, a_inter = utils.infer_dist(activ_, attrs)
    c_intra_n, c_inter_n = utils.infer_dist(normed_activ_, ys)
    a_intra_n, a_inter_n = utils.infer_dist(normed_activ_, attrs)

    # infer group labels with reduced activ
    activ0, activ1 = activ_[ys == 0], activ_[ys == 1]
    attrs0, attrs1 = attrs[ys == 0], attrs[ys == 1]
    sub_class0 = utils.infer_labels(activ0, attrs0)
    sub_class1 = utils.infer_labels(activ1, attrs1)
    group00 = np.sum(sub_class0 == 0)
    group01 = np.sum(sub_class0 == 1)
    group10 = np.sum(sub_class1 == 0)
    group11 = np.sum(sub_class1 == 1)
    total = group00 + group01 + group10 + group11
    sc_ = (group00 + group11) / total
    ci_ = (group00 + group01) / total
    ai_ = (group00 + group10) / total

    # compute distances wrt inferred labels
    inferred_attrs = np.concatenate([sub_class0, sub_class1])
    activ_ = np.concatenate([activ0, activ1])
    a_intra_i, a_inter_i = utils.infer_dist(activ_, inferred_attrs)

    # infer group labels with normed reduced activ
    activ0, activ1 = normed_activ_[ys == 0], normed_activ_[ys == 1]
    attrs0, attrs1 = attrs[ys == 0], attrs[ys == 1]
    sub_class0 = utils.infer_labels(activ0, attrs0)
    sub_class1 = utils.infer_labels(activ1, attrs1)
    group00 = np.sum(sub_class0 == 0)
    group01 = np.sum(sub_class0 == 1)
    group10 = np.sum(sub_class1 == 0)
    group11 = np.sum(sub_class1 == 1)
    total = group00 + group01 + group10 + group11
    normed_sc_ = (group00 + group11) / total
    normed_ci_ = (group00 + group01) / total
    normed_ai_ = (group00 + group10) / total

    # compute distances wrt inferred labels
    inferred_attrs = np.concatenate([sub_class0, sub_class1])
    normed_activ_ = np.concatenate([activ0, activ1])
    a_intra_in, a_inter_in = utils.infer_dist(normed_activ_, inferred_attrs)

    return {
        "sc_": sc_,
        "ci_": ci_,
        "ai_": ai_,
        "normed_sc_": normed_sc_,
        "normed_ci_": normed_ci_,
        "normed_ai_": normed_ai_,
        "c_intra": c_intra,
        "a_intra": a_intra,
        "c_inter": c_inter,
        "a_inter": a_inter,
        "c_intra_n": c_intra_n,
        "a_intra_n": a_intra_n,
        "c_inter_n": c_inter_n,
        "a_inter_n": a_inter_n,
        "a_intra_i": a_intra_i,
        "a_inter_i": a_inter_i,
        "a_intra_in": a_intra_in,
        "a_inter_in": a_inter_in,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The following ones specify the task, data size, distribution shifts, and the model to get embeddings from."
    )
    parser.add_argument("--data_name", type=str, default="celeba")
    parser.add_argument("--data_dir", type=str, help="directory to store the dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="directory to store the outputs",
        default="../data_src",
    )
    parser.add_argument("--model", type=str, default="resnet", help="Model to use")
    parser.add_argument("--split", type=str, choices=["tr", "te"])
    parser.add_argument("--data_size", type=int, nargs="+")
    parser.add_argument("--sc", type=float, nargs="+")
    parser.add_argument("--ci", type=float, nargs="+")
    parser.add_argument("--ai", type=float, nargs="+")
    parser.add_argument("--task_y", type=int, nargs="+")
    parser.add_argument("--task_a", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(args)

    seed, split, model, data_dir, data_name = (
        args.seed,
        args.split,
        args.model,
        args.data_dir,
        args.data_name.lower(),
    )

    # zip shifts and tasks
    shifts = list(zip(args.sc, args.ci, args.ai))
    tasks = list(zip(args.task_y, args.task_a))
    if data_name == "officehome":
        tasks = [(args.task_y[0], args.task_a[0])] * len(shifts)
        combinations = zip(args.data_size, tasks, shifts)
    else:
        combinations = itertools.product(args.data_size, tasks, shifts)

    for data_size, (task_y, task_a), (sc, ci, ai) in tqdm(list(combinations)):
        task_dir = os.path.join(
            args.output_dir,
            data_name,
            f"tasks_y{task_y}_a{task_a}",
            "task_{}_sc{:.2f}_ci{:.2f}_ai{:.2f}".format(data_size, sc, ci, ai),
        )
        # check if task_invalid file exists in task_dir
        if os.path.exists(os.path.join(task_dir, "task_invalid")):
            print("Task is invalid. Exiting...")
            continue

        emb_file = os.path.join(
            task_dir,
            model,
            f"emb_{split}.npz",
        )
        desc_file = os.path.join(task_dir, model, f"desc_{split}.csv")
        if not os.path.exists(os.path.join(task_dir, model)):
            os.makedirs(os.path.join(task_dir, model))
        else:
            if os.path.exists(emb_file) and os.path.exists(desc_file):
                print("Embeddings and descriptors already exist. Exiting...")
                continue

        activ, ys, attrs = get_embeddings(
            data_name,
            data_dir,
            task_dir,
            model,
            task_y,
            task_a,
            data_size,
            sc,
            ci,
            ai,
            split,
            seed,
        )

        np.savez(emb_file, activ=activ, ys=ys, attrs=attrs)

        # extract data descriptors
        desc = extract_data_desc(activ, ys, attrs)
        desc["sc"] = sc
        desc["ci"] = ci
        desc["ai"] = ai
        desc["n"] = data_size
        desc["y_task"] = task_y
        desc["a_task"] = task_a

        # save desc to csv
        desc_df = pd.DataFrame([desc])
        desc_df.to_csv(desc_file, index=False)
