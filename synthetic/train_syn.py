import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import copy
import pandas as pd
import datetime
import argparse


def generate_toy_data(
    n,
    sc,
    ci,
    ai,
    mean_causal,
    var_causal,
    mean_spurious,
    var_spurious,
    d_feat=None,
    d_noise=None,
    train=True,
    verbose=False,
):
    groups = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    n_groups = len(groups)

    group_matrix = np.array(
        [
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ]
    )
    b = [sc * n, ai * n, ci * n, n]
    num_per_group = np.linalg.solve(group_matrix, b)
    if verbose:
        if train:
            print("num_per_group", num_per_group)
        else:
            print("num_per_group", n / 4)

    y_list, a_list, x_causal_list, x_spurious_list, g_list = [], [], [], [], []
    for group_idx, (y_value, a_value) in enumerate(groups):
        if train:
            n_group = int(num_per_group[group_idx])
        else:
            n_group = int(n / n_groups)

        y_list.append(np.ones(n_group) * y_value)
        a_list.append(np.ones(n_group) * a_value)
        g_list.append(np.ones(n_group) * group_idx)
        if d_feat is None:
            x_causal_list.append(
                np.random.normal(
                    y_value * mean_causal, var_causal**0.5, n_group
                ).reshape(n_group, 1)
            )
            x_spurious_list.append(
                np.random.normal(
                    a_value * mean_spurious, var_spurious**0.5, n_group
                ).reshape(n_group, 1)
            )
        else:
            assert d_feat > 1
            x_causal_list.append(
                np.random.multivariate_normal(
                    mean=y_value * np.ones(d_feat) * mean_causal,
                    cov=np.eye(d_feat) * var_causal,
                    size=n_group,
                )
            )
            x_spurious_list.append(
                np.random.multivariate_normal(
                    mean=a_value * np.ones(d_feat) * mean_spurious,
                    cov=np.eye(d_feat) * var_spurious,
                    size=n_group,
                )
            )

    y = np.concatenate(y_list)
    a = np.concatenate(a_list)
    g = np.concatenate(g_list)
    x_causal = np.vstack(x_causal_list)
    x_spurious = np.vstack(x_spurious_list)

    if d_noise is not None:
        assert d_noise > 0
        x_noise = np.random.multivariate_normal(
            mean=0 * np.ones(d_noise), cov=np.eye(d_noise) / d_noise, size=n
        )
        x = np.hstack([x_causal, x_spurious, x_noise])
    else:
        x = np.hstack([x_causal, x_spurious])
    return (x, y, g), n_groups, a


def generate_valid_dist_shifts(num_exps, min_n=200):
    import random

    group_matrix = np.array(
        [
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ]
    )

    i = 0
    dist_shift_lists = []
    while i < num_exps:
        # sample sc, ci, ai from (0,1) randomly and independently
        sc, ci, ai = [round(random.uniform(0, 1), 2) for _ in range(3)]
        b = [sc * min_n, ai * min_n, ci * min_n, min_n]
        num_per_group = np.linalg.solve(group_matrix, b)
        # check constraints
        c1 = (sc + ci + ai) > 1
        c2 = sc > (ci + ai - 1)
        c3 = ci > (sc + ai - 1)
        c4 = ai > (sc + ci - 1)
        # c5 = sc > 0.01
        # c6 = ci > 0.01
        # c7 = ai > 0.01
        if c1 and c2 and c3 and c4:  # and c5 and c6 and c7:
            if (num_per_group >= 2).all():
                dist_shift_lists.append([sc, ci, ai])
                i += 1
    return dist_shift_lists


def oversample(g, n_groups):
    group_counts = []
    for group_idx in range(n_groups):
        group_counts.append((g == group_idx).sum())
    resampled_idx = []
    for group_idx in range(n_groups):
        (idx,) = np.where(g == group_idx)
        if group_counts[group_idx] < max(group_counts):
            for _ in range(max(group_counts) // group_counts[group_idx]):
                resampled_idx.append(idx)
            resampled_idx.append(
                np.random.choice(
                    idx, max(group_counts) % group_counts[group_idx], replace=False
                )
            )
        else:
            resampled_idx.append(idx)
    resampled_idx = np.concatenate(resampled_idx)
    return resampled_idx


def undersample(g, n_groups):
    group_counts = []
    for group_idx in range(n_groups):
        group_counts.append((g == group_idx).sum())
    resampled_idx = []
    for group_idx in range(n_groups):
        (idx,) = np.where(g == group_idx)
        resampled_idx.append(np.random.choice(idx, min(group_counts), replace=False))
    resampled_idx = np.concatenate(resampled_idx)
    return resampled_idx


def groupdro_loss(yhat, y, gs, q):
    losses = F.binary_cross_entropy_with_logits(yhat, y, reduction="none")

    for g in np.unique(gs):
        idx_g = g == gs
        q[g] *= (1e-3 * losses[idx_g].mean()).exp().item()

    q /= q.sum()
    loss = 0
    for g in np.unique(gs):
        idx_g = g == gs
        loss += q[g] * losses[idx_g].mean()

    return loss, q


def run(method, n, sc, ci, ai, var_causal, d_feat, seed=0, tol=1e-3, verbose=False):
    np.random.seed(seed)
    tr_data_args = {
        "n": n,
        "sc": sc,
        "ci": ci,
        "ai": ai,
        "mean_causal": 1,
        "var_causal": var_causal,
        "mean_spurious": 1,
        "var_spurious": 1,
        "d_feat": d_feat,
    }

    te_data_args = {
        "n": n,
        "sc": 0.5,
        "ci": 0.5,
        "ai": 0.5,
        "mean_causal": 1,
        "var_causal": var_causal,
        "mean_spurious": 1,
        "var_spurious": 1,
        "d_feat": d_feat,
    }

    (tr_x, tr_y, tr_g), n_groups, tr_a = generate_toy_data(**tr_data_args, train=True)
    (te_x, te_y, te_g), n_groups, te_a = generate_toy_data(**te_data_args, train=False)

    tr_y = (tr_y + 1) / 2
    te_y = (te_y + 1) / 2

    net = nn.Linear(tr_x.shape[1], 1, bias=False)
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    if method == "ERM":
        loss_fn = nn.BCEWithLogitsLoss()
    elif method == "GroupDRO":
        loss_fn = groupdro_loss
        q = torch.ones(n_groups, dtype=torch.float32)
        tr_g = torch.tensor(tr_g, dtype=torch.int64)
    elif method == "remax-margin":
        loss_fn = nn.BCEWithLogitsLoss()
        cnts = np.unique(tr_g, return_counts=True)[1]
        c = cnts / np.sum(cnts)
        c = c / c.max()
    elif method == "oversample":
        loss_fn = nn.BCEWithLogitsLoss()
        over_resample_idx = oversample(tr_g, n_groups)
        tr_x = tr_x[over_resample_idx, :]
        tr_y = tr_y[over_resample_idx]
    elif method == "undersample":
        loss_fn = nn.BCEWithLogitsLoss()
        under_resample_idx = undersample(tr_g, n_groups)
        tr_x = tr_x[under_resample_idx, :]
        tr_y = tr_y[under_resample_idx]

    train_iter = 10000
    log_every = 1000

    # convert data
    tr_x = torch.tensor(tr_x, dtype=torch.float32)
    tr_y = torch.tensor(tr_y, dtype=torch.float32).reshape(-1, 1)

    te_x = torch.tensor(te_x, dtype=torch.float32)
    te_y = torch.tensor(te_y, dtype=torch.float32).reshape(-1, 1)

    last_loss = torch.tensor(0.0)
    for t in range(train_iter + 1):
        logits = net(tr_x)
        if method == "GroupDRO":
            loss, q = loss_fn(logits, tr_y, tr_g, q)
        elif method in ["ERM", "oversample", "undersample"]:
            loss = loss_fn(logits, tr_y)
        elif method == "remax-margin":
            loss = 0.0
            for i in range(n_groups):
                idx = tr_g == i
                loss += loss_fn(c[i] * logits[idx], tr_y[idx]) / n_groups
        else:
            raise ValueError(f"unknown method {method}")

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
    pred = torch.sigmoid(net(te_x))
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

    if verbose:
        print(f"train acc {tr_acc.item():.5f}")
        print(f"avg test acc {te_acc.item():.5f}")
        print(f"avg test 0-1 error {(1 - te_acc).item():.5f}")
        print(f"worst-case test acc {min(worst_case_acc):.5f}")

    # collect results
    res = copy.deepcopy(tr_data_args)
    res["method"] = method
    res["avg_tr_err"] = 1 - tr_acc.item()
    res["avg_te_err"] = 1 - te_acc.item()
    res["wga_te_err"] = 1 - min(worst_case_acc)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments with different algorithms and configurations."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="directory to store the outputs",
        default="../data_src",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--sc", type=float, nargs="+")
    parser.add_argument("--ci", type=float, nargs="+")
    parser.add_argument("--ai", type=float, nargs="+")

    args = parser.parse_args()
    print(args)

    methods = ["ERM", "GroupDRO", "remax-margin", "oversample", "undersample"]
    n_list = [200, 500, 1000, 2000, 3000, 5000, 10000]

    dist_shift_lists = list(zip(args.sc, args.ci, args.ai))
    var_list = [1, 5, 10, 20, 50, 100]
    d_feat = [2, 10, 50, 100]
    verbose = False

    # print hyperparameters
    print(f"methods: {methods}")
    print(f"n_list: {n_list}")
    print(f"dist_shift_lists: {dist_shift_lists}")
    print(f"var_list: {var_list}")
    print(f"d_feat: {d_feat}")
    print(f"verbose: {verbose}")

    total_num_loop = (
        len(n_list) * len(dist_shift_lists) * len(var_list) * len(d_feat) * len(methods)
    )
    i = 0
    all_res = []
    for n in n_list:
        for dist_s in dist_shift_lists:
            for var in var_list:
                for d in d_feat:
                    for method in methods:
                        curr_res = run(
                            method,
                            n,
                            sc=dist_s[0],
                            ci=dist_s[1],
                            ai=dist_s[2],
                            var_causal=var,
                            d_feat=d,
                            seed=args.seed,
                            verbose=verbose,
                        )
                        all_res.append(curr_res)
                        i += 1
                        if i % 100 == 0:
                            print(
                                f"Done {i}/{total_num_loop}",
                                datetime.datetime.now(),
                            )
    pd.DataFrame(all_res).to_csv(
        os.path.join(argparse.output_dir, f"synthetic_results_seed{args.seed}.csv")
    )
