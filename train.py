import numpy as np
import torch
from torch import nn
import copy
import pandas as pd
import os
import argparse
import itertools
from tqdm import tqdm
from algs import groupdro_loss, oversample, undersample, mixupsample, irm_loss


def load_emb(data_dir, model, split):
    emb_file = os.path.join(data_dir, model, f"emb_{split}.npz")
    # load embeddings
    emb = np.load(emb_file)
    activ = emb["activ"]
    ys = emb["ys"]
    attrs = emb["attrs"]
    n_groups = 4
    # generate group labels by ys and attrs
    g = 2 * ys + attrs
    return (activ, ys, g), n_groups, attrs


def run(
    data_name,
    data_dir,
    algorithm,
    n,
    sc,
    ci,
    ai,
    y_task,
    a_task,
    model,
    classifier,
    num_epochs,
    tol=1e-3,
    verbose=False,
):

    data_args = {
        "data_name": data_name,
        "n": n,
        "sc": sc,
        "ci": ci,
        "ai": ai,
        "y_task": y_task,
        "a_task": a_task,
        "model": model,
        "classifier": classifier,
        "num_epochs": num_epochs,
    }

    (tr_x, tr_y, tr_g), n_groups, tr_a = load_emb(data_dir, model, split="tr")
    (te_x, te_y, te_g), n_groups, te_a = load_emb(data_dir, model, split="te")

    if classifier == "linear":
        net = nn.Linear(tr_x.shape[1], 1, bias=False)
    elif classifier == "mlp":
        net = nn.Sequential(
            nn.Linear(tr_x.shape[1], 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
    else:
        raise ValueError(f"unknown classifier {classifier}")

    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    if algorithm == "ERM":
        loss_fn = nn.BCEWithLogitsLoss()
    elif algorithm == "GroupDRO":
        loss_fn = groupdro_loss
        q = torch.ones(n_groups, dtype=torch.float32)
        tr_g = torch.tensor(tr_g, dtype=torch.int64)
    elif algorithm == "remax-margin":
        loss_fn = nn.BCEWithLogitsLoss()
        cnts = np.unique(tr_g, return_counts=True)[1]
        c = cnts / np.sum(cnts)
        c = c / c.max()
    elif algorithm == "oversample":
        loss_fn = nn.BCEWithLogitsLoss()
        over_resample_idx = oversample(tr_g, n_groups)
        tr_x = tr_x[over_resample_idx, :]
        tr_y = tr_y[over_resample_idx]
    elif algorithm == "undersample":
        loss_fn = nn.BCEWithLogitsLoss()
        under_resample_idx = undersample(tr_g, n_groups)
        tr_x = tr_x[under_resample_idx, :]
        tr_y = tr_y[under_resample_idx]
    elif algorithm == "LISA":
        loss_fn = nn.BCEWithLogitsLoss()
        tr_x, tr_y = mixupsample(tr_x, tr_y, tr_a)
    elif algorithm == "IRM":
        loss_fn = irm_loss
    else:
        raise ValueError(f"unknown algorithm")

    train_iter = num_epochs
    log_every = 100

    # convert data
    tr_x = torch.tensor(tr_x, dtype=torch.float32)
    tr_g = torch.tensor(tr_g, dtype=torch.int64)
    tr_y = torch.tensor(tr_y, dtype=torch.float32).reshape(-1, 1)

    te_x = torch.tensor(te_x, dtype=torch.float32)
    te_y = torch.tensor(te_y, dtype=torch.float32).reshape(-1, 1)

    last_loss = torch.tensor(0.0)
    for t in range(train_iter + 1):
        logits = net(tr_x)
        if algorithm == "GroupDRO":
            loss, q = loss_fn(logits, tr_y, tr_g, q)
        elif algorithm in ["ERM", "LISA", "oversample", "undersample"]:
            loss = loss_fn(logits, tr_y)
        elif algorithm == "remax-margin":
            loss = 0.0
            for i in range(n_groups):
                idx = tr_g == i
                loss += loss_fn(c[i] * logits[idx], tr_y[idx]) / n_groups
        elif algorithm == "IRM":
            loss = loss_fn(logits, tr_y, tr_g, t)
        else:
            raise ValueError(f"unknown algorithm {algorithm}")

        # add l2 regularization
        l2_reg = torch.tensor(0.0)
        for param in net.parameters():
            l2_reg += torch.norm(param)
        loss += 1e-6 * l2_reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        if t % log_every == 0:
            if (last_loss - loss.item()) < tol and t > 0:
                break
            else:
                last_loss = loss.item()
            if verbose:
                print(f"{t=} xent {loss.item():.5f}")

    # get training accuracy
    pred = torch.sigmoid(net(tr_x))
    pred = (pred > 0.5).float()
    correct = (pred == tr_y).float().sum()
    tr_acc = correct / len(tr_y)

    # get test accuracy
    te_logits = net(te_x)
    pred = torch.sigmoid(te_logits)
    pred = (pred > 0.5).float()
    correct = (pred == te_y).float().sum()
    te_acc = correct / len(te_y)

    # get worst-case accuracy
    worst_case_acc = []
    for g in range(n_groups):
        idx = te_g == g
        g_pred = pred[idx]
        correct = (g_pred == te_y[idx]).float().sum()
        acc = correct / len(te_y[idx])
        worst_case_acc.append(acc.item())
        # print(f"test acc group {g} {acc.item():.5f}")

    print(f"train acc {tr_acc.item():.5f}")
    print(f"avg test acc {te_acc.item():.5f}")
    print(f"avg test 0-1 error {(1 - te_acc).item():.5f}")
    print(f"worst-case test error {1 - min(worst_case_acc):.5f}")

    # collect results
    res = copy.deepcopy(data_args)
    res["algorithm"] = algorithm
    res["avg_tr_err"] = 1 - tr_acc.item()
    res["avg_te_err"] = 1 - te_acc.item()
    res["wga_te_err"] = 1 - min(worst_case_acc)

    # collect predictions and labels
    pred = pd.DataFrame(
        {
            "label": te_y.flatten().detach().numpy(),
            "group": te_g,
            algorithm: te_logits.flatten().detach().numpy(),
        }
    )

    return res, pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments with different algorithms and configurations."
    )
    parser.add_argument("--model", type=str, default="resnet", help="Model to use")
    parser.add_argument("--classifier", type=str, default="linear")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="directory to store the outputs",
        default="../data_src",
    )
    parser.add_argument("--data_name", type=str, default="celeba")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--algorithm", type=str, nargs="+")
    parser.add_argument("--data_size", type=int, nargs="+")
    parser.add_argument("--sc", type=float, nargs="+")
    parser.add_argument("--ci", type=float, nargs="+")
    parser.add_argument("--ai", type=float, nargs="+")
    parser.add_argument("--task_y", type=int, nargs="+")
    parser.add_argument("--task_a", type=int, nargs="+")
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(args)

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    seed, model, classifier, data_name, num_epochs = (
        args.seed,
        args.model,
        args.classifier,
        args.data_name.lower(),
        args.num_epochs,
    )

    # zip shifts and tasks and algorithms
    shifts = list(zip(args.sc, args.ci, args.ai))
    tasks = list(zip(args.task_y, args.task_a))
    # combinations = itertools.product(args.data_size, tasks, shifts, args.algorithm)
    if data_name == "officehome":
        tasks = [(args.task_y[0], args.task_a[0])] * len(shifts) * len(args.algorithm)
        ds = args.data_size * len(args.algorithm)
        shifts = shifts * len(args.algorithm)
        algs = [a for a in args.algorithm for _ in range(len(args.data_size))]
        combinations = zip(ds, tasks, shifts, algs)
    else:
        combinations = itertools.product(args.data_size, tasks, shifts, args.algorithm)

    for data_size, (task_y, task_a), (sc, ci, ai), algorithm in tqdm(
        list(combinations)
    ):
        # append to a potentially existing csv file
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

        res, pred = run(
            data_name.lower(),
            task_dir,
            algorithm,
            data_size,
            sc,
            ci,
            ai,
            task_y,
            task_a,
            model,
            classifier,
            num_epochs,
            verbose=args.verbose,
        )

        res_file = os.path.join(task_dir, model, f"{classifier}1_res_seed{seed}.csv")
        prediction_file = os.path.join(task_dir, model, f"{classifier}1_prediction_seed{seed}.csv")

        if os.path.exists(res_file):
            res_df = pd.read_csv(res_file)
            # check if the current configuration has been run
            if (
                res_df[
                    (res_df["data_name"] == data_name)
                    & (res_df["n"] == data_size)
                    & (res_df["algorithm"] == algorithm)
                    & (res_df["sc"] == sc)
                    & (res_df["ci"] == ci)
                    & (res_df["ai"] == ai)
                    & (res_df["y_task"] == task_y)
                    & (res_df["a_task"] == task_a)
                    & (res_df["model"] == model)
                    & (res_df["num_epochs"] == num_epochs)
                ].shape[0]
                > 0
            ):
                print("Configuration already run")
                continue
        else:
            res_df = pd.DataFrame()

        res = pd.DataFrame(res, index=[0])
        res_df = pd.concat([res_df, res], ignore_index=True)
        res_df.to_csv(res_file, index=False)

        if os.path.exists(prediction_file):
            pred_df = pd.read_csv(prediction_file)
            # check if the current configuration has been run
            if algorithm in pred_df.columns:
                print("Configuration already run")
                continue
        else:
            pred_df = pred[["label", "group"]]

        # add a new column to the prediction dataframe
        pred_df[algorithm] = pred[algorithm]
        pred_df.to_csv(prediction_file, index=False)
