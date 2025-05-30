import os
import json
import pickle
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


def generate_task_metadata(data_name, data_dir, output_dir, task_y, task_a):
    if data_name == "celeba":
        return generate_task_celeba(data_dir, output_dir, task_y, task_a)
    elif data_name == "metashift":
        return generate_task_metashift(data_dir, output_dir)
    elif data_name == "multinli":
        return generate_task_multinli(data_dir, output_dir)
    elif data_name == "civilcomments":
        return generate_task_civilcomments(data_dir, output_dir, task_a)
    elif data_name == "officehome":
        return generate_task_officehome(data_dir, output_dir, task_y, task_a)
    elif data_name == "cmnist":
        return generate_task_cmnist(data_dir, output_dir, task_y, task_a)
    else:
        raise ValueError("Invalid data name.")


def generate_task_celeba(data_path, output_path, y_col=9, a_col=20):
    logging.info("Generating metadata for CelebA...")
    with open(os.path.join(data_path, "CelebFaces/list_eval_partition.txt"), "r") as f:
        splits = f.readlines()

    with open(os.path.join(data_path, "CelebFaces/list_attr_celeba.txt"), "r") as f:
        attrs = f.readlines()[2:]

    p = os.path.join(output_path, "metadata_all.csv")
    f = open(p, "w")
    f.write("id,filename,split,y,a\n")

    for i, (split, attr) in enumerate(zip(splits, attrs)):
        fi, si = split.strip().split()
        ai = attr.strip().split()[1:]
        yi = 1 if ai[y_col] == "1" else 0
        gi = 1 if ai[a_col] == "1" else 0
        f.write("{},{},{},{},{}\n".format(i + 1, fi, si, yi, gi))

    f.close()

    return p


def generate_task_metashift(data_path, output_path):
    logging.info("Generating metadata for MetaShift...")
    data_path = os.path.join(data_path, "metashift/COCO-Cat-Dog-indoor-outdoor")
    with open(os.path.join(data_path, "imageID_to_group.pkl"), "rb") as pkl_f:
        gt = pickle.load(pkl_f)

    metadata_all = {"filename": [], "split": [], "y": [], "a": []}

    folder_list = ["train", "val_out_of_domain"]
    cls_list = ["cat", "dog"]
    for f in folder_list:
        for c in cls_list:
            curr_path = os.path.join(data_path, f, c)
            imgs = os.listdir(curr_path)
            metadata_all["filename"].extend([os.path.join(f, c, img) for img in imgs])
            if f == "train":
                metadata_all["split"].extend([0] * len(imgs))
            else:
                metadata_all["split"].extend([2] * len(imgs))
            if c == "dog":
                metadata_all["y"].extend([0] * len(imgs))
            else:
                metadata_all["y"].extend([1] * len(imgs))
            metadata_all["a"].extend([gt[img[:-4]][0][4:-1] for img in imgs])
    df_metadata = pd.DataFrame(metadata_all)

    df_metadata["a"] = df_metadata["a"].apply(lambda x: 1 if x == "indoor" else 0)
    df_metadata["id"] = df_metadata.index
    p = os.path.join(output_path, "metadata_all.csv")
    df_metadata.to_csv(p, index=False)

    return p

def generate_task_officehome(data_path, output_path, task_y, task_a):
    logging.info("No need to generate metadata for OfficeHome...it is handled on-the-fly.")

def generate_task_cmnist(data_path, output_path, task_y, task_a):
    logging.info("No need to generate metadata for cmnist...it is handled on-the-fly.")


def generate_task_multinli(data_path, output_path):
    logging.info("Generating metadata for MultiNLI...")
    df = pd.read_csv(
        os.path.join(data_path, "multinli", "metadata_2classes.csv"), index_col=0
    )
    df = df.rename(columns={"gold_label": "y", "sentence2_has_negation": "a"})
    df = df.reset_index(drop=True)
    df.index.name = "id"
    df = df.reset_index()
    df["filename"] = df["id"]
    df = df.reset_index()[["id", "filename", "split", "y", "a"]]

    p = os.path.join(output_path, "metadata_all.csv")
    df.to_csv(p, index=False)

    return p


def generate_task_civilcomments(data_path, output_path, a_col=None):
    logging.info("Generating metadata for CivilComments...")
    df = pd.read_csv(
        os.path.join(data_path, "civilcomments", "all_data_with_identities.csv"),
        index_col=0,
    )
    group_attrs = [
        "male",
        "female",
        "LGBTQ",
        "christian",
        "muslim",
        "other_religions",
        "black",
        "white",
    ]
    cols_to_keep = ["comment_text", "split", "toxicity"]
    df = df[cols_to_keep + group_attrs]
    df = df.rename(columns={"toxicity": "y"})
    df["y"] = (df["y"] >= 0.5).astype(int)
    df[group_attrs] = (df[group_attrs] >= 0.5).astype(int)

    df["id"] = df.index
    df["filename"] = df["id"]
    a = group_attrs[a_col]
    df = df.reset_index()[["id", "filename", "split", "y", a]]
    # map split to 0, 1, 2
    df["split"] = df["split"].map({"train": 0, "val": 1, "test": 2})
    # rename columns
    df.columns = ["id", "filename", "split", "y", "a"]
    p = os.path.join(output_path, "metadata_all.csv")
    df.to_csv(p, index=False)

    return p


def infer_labels(activ, attrs, verbose=False):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(activ)
    if verbose:
        print("silhouette_score", silhouette_score(activ, kmeans.labels_))
        print(
            max(
                np.array(kmeans.labels_ == attrs, dtype=int).mean(),
                np.array(kmeans.labels_ == (1 - attrs), dtype=int).mean(),
            )
        )
    if (
        np.array(kmeans.labels_ == attrs, dtype=int).mean()
        > np.array(kmeans.labels_ == (1 - attrs), dtype=int).mean()
    ):
        sub_class = kmeans.labels_
    else:
        sub_class = 1 - kmeans.labels_

    return sub_class


def infer_dist(activ, ys, verbose=False):
    c0 = activ[ys == 0].mean(axis=0)
    c1 = activ[ys == 1].mean(axis=0)
    c0_ave_dist = np.mean(np.linalg.norm(activ[ys == 0] - c0, axis=1))
    c1_ave_dist = np.mean(np.linalg.norm(activ[ys == 1] - c1, axis=1))
    c_ave_dist = (c0_ave_dist + c1_ave_dist) / 2
    c_inter_dist = np.linalg.norm(c0 - c1)

    return c_ave_dist, c_inter_dist


# Load YAML file
def load_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)  # Use safe_load for security
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


# Define the MLP model using nn.Sequential
class MLPTorch(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(MLPTorch, self).__init__()
        layers = []
        layer_sizes = [input_size] + list(hidden_layer_sizes)
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Define the linear model
class LinearTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# Define the KNN model using the sklearn implementation
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, n_neighbors):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets, weights):
        loss = self.bce(outputs, targets)
        # print(weights)
        weighted_loss = loss * weights.unsqueeze(1)
        return weighted_loss.mean()


def sigmoid(x, k):
    return 1 / (1 + np.exp(-k * x))


def load_data(model, data_path, classifier, algorithms, identifier, metric, margin=0.05):
    tasks = os.listdir(data_path)
    all_res = []
    for t in tasks:
        datasets = os.listdir(os.path.join(data_path, t))
        for d in datasets:
            if "csv" in d: continue

            data = os.path.join(data_path, t, d, model)
            # check if data exists
            if not os.path.exists(data):
                print(f"{data} does not exist")
                continue
            else:
                print("processing...", data)
            # load the results with different seeds
            res_files = os.listdir(data)
            res = []
            for r in res_files:
                if f"{classifier}_res" in r:
                    res_s = pd.read_csv(os.path.join(data, r))
                    # parse the seed from the file name
                    res_s["seed"] = int(r.split('seed')[-1][0])
                    res.append(res_s)
            res = pd.concat(res, ignore_index=True)
            # groupby the identifier and take the mean of the results
            res = res.groupby(identifier+["algorithm"]).agg({"avg_tr_err": "mean", "avg_te_err": "mean", "wga_te_err": "mean", "num_epochs": "mean", "seed": "mean", "data_name": "first", "model": "first", "classifier": "first"}).reset_index()
            # load the data descriptors
            desc = pd.read_csv(os.path.join(data, "desc_tr.csv"))
            # concatenate the results with the data descriptors
            # assert that the identifier is the same for the data descriptor and the results
            assert np.all(desc[identifier].values[0] == res[identifier].values[0])
            # join the results with the data descriptor
            res = res.merge(desc, on=identifier)
            all_res.append(res)
    all_res = pd.concat(all_res, ignore_index=True)

    # get the best performing algorithm for each dataset, used for deduplicate
    all_res['rank'] = all_res.groupby(identifier)[metric].rank("first")

    # get the winners for each dataset
    def get_gt_rank(x, filter_thre=margin):
        min_err = x[metric].min()
        winners = x[x[metric] <= min_err + filter_thre]["algorithm"].to_list()
        return '|'.join(winners)
    # apply the function to each group and reset the index to merge back with the original dataframe
    winners_series = all_res.groupby(identifier)[['algorithm', metric]].apply(lambda x: get_gt_rank(x)).reset_index(name='winners')
    all_res = all_res.merge(winners_series, on=identifier)
    # get the multi-hot encoding of the winners
    all_res["multi_hot"] = all_res["winners"].map(lambda x: [1 if m in x.split("|") else 0 for m in algorithms])

    return all_res

# convert the loaded data to the format required by the model
def prepare_data(mode, train_df, eval_df, tr_idx, val_idx, input_feats, algorithms, metric, margin=0.05, normalize=True):
    if mode == "mlc" or mode == "baseline":
        # deduplicate
        train_df = train_df[train_df["rank"]==1.0]
        eval_df = eval_df[eval_df["rank"]==1.0]

        # process train
        X = train_df[input_feats].to_numpy().astype('float')
        y = np.array(list(train_df["multi_hot"].to_numpy()))

        # use tr_idx to split the data
        X_train, X_val, y_train, y_val = X[tr_idx], X[val_idx], y[tr_idx], y[val_idx]

        if normalize:
            # standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # process eval
        X_test = eval_df[input_feats].to_numpy().astype('float')
        y_test = np.array(list(eval_df["multi_hot"].to_numpy()))
        if normalize:
            X_test = scaler.transform(X_test)
    elif mode == "regression":
        num_alg = len(algorithms)
        # expand the indices
        tr_idx = np.concatenate([np.arange(num_alg*i, num_alg*i+num_alg) for i in tr_idx])
        val_idx = np.concatenate([np.arange(num_alg*i, num_alg*i+num_alg) for i in val_idx])

        # process train
        X = train_df[input_feats+["algorithm"]]
        X['algorithm'] = X['algorithm'].map(lambda x: algorithms.index(x))
        X = X.to_numpy().astype('float')
        y = train_df[metric].to_numpy()

        # use tr_idx to split the data
        X_train, X_val, y_train, y_val = X[tr_idx], X[val_idx], y[tr_idx], y[val_idx]

        # standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # process eval
        X_test = eval_df[input_feats+["algorithm"]]
        X_test['algorithm'] = X_test['algorithm'].map(lambda x: algorithms.index(x))
        y_test = eval_df[metric].to_numpy()
        X_test = scaler.transform(X_test)

        # convert y_val and y_test to multi-hot
        def get_rank(x):
            min_err = x.min()
            return (x <= min_err + margin).astype(int)

        y_val_mh = []
        for i in range(len(y_val)//len(algorithms)):
            curr_pred = y_val[len(algorithms)*i:len(algorithms)*i+len(algorithms)]
            y_val_mh.append(get_rank(curr_pred))
        y_val = np.array(y_val_mh)

        y_test_mh = []
        for i in range(len(y_test)//len(algorithms)):
            curr_pred = y_test[len(algorithms)*i:len(algorithms)*i+len(algorithms)]
            y_test_mh.append(get_rank(curr_pred))
        y_test = np.array(y_test_mh)

    # return a dictionary with the data
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test}


# simple training function
def train_model(
    model, dataloader, criterion, optimizer, num_epochs, patience, tol, verbose=True
):
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss - tol:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return model


# evaluate the 0-1 accuracy of algorithm selection task itself
def eval_acc(y_test, y_pred, mode, verbose=True):
    # assuming numpy arrays
    if mode == "0-1":
        acc = (y_pred == y_test).all(axis=1).mean()
        if verbose:
            print(f"Eval {mode} accuracy: {acc}")
    elif mode == "soft 0-1":
        correct = 0
        for i, curr_y in enumerate(y_test):
            curr_pred = y_pred[i]
            # check if the positions of 1s in curr_pred are also 1s in curr_y
            pos = np.where(curr_pred==1)[0]
            if np.all(curr_y[pos] == 1):
                correct += 1
        acc = correct / y_test.shape[0]
        if verbose:
            print(f"Eval {mode} accuracy: {acc}")


# evaluate the actual accuracy of the selected algorithms
def eval_wga_err(test_df, y_pred, identifier, algorithms, metric, y_prob=None):
    # check if y_pred is already a list
    if not isinstance(y_pred, list):
        y_pred = y_pred.tolist()
    # convert the multi-hot vector to the algorithm name
    y_to_alg, prob_to_alg = [], []
    for i, y_ in enumerate(y_pred):
        y_ = np.array(y_)
        y_to_alg.append(np.array(algorithms)[np.where(y_==1)[0]])

        if y_prob is not None:
            y_l = y_prob[i]
            prob_to_alg.append(y_l[np.where(y_==1)[0]])
        else:
            prob_to_alg.append(np.ones(len(np.where(y_==1)[0])))

    y_to_alg = [y for y in y_to_alg for _ in range(len(algorithms))]
    prob_to_alg = [p for p in prob_to_alg for _ in range(len(algorithms))]
    test_df['pred_alg'] = y_to_alg
    test_df['prob'] = prob_to_alg

    def get_pred_metric(x):
        pred_winners = x['pred_alg'].iloc[0]
        if len(pred_winners) == 0:
            pred_winner = np.random.choice(algorithms)
        else:
            if y_prob is not None:
                # print(pred_winners)
                if isinstance(pred_winners, list) or isinstance(pred_winners, np.ndarray):
                    pred_winner = pred_winners[np.argmax(x['prob'].iloc[0])]
                else:
                    pred_winner = pred_winners
            else:
                # if not a list
                if isinstance(pred_winners, list) or isinstance(pred_winners, np.ndarray):
                    pred_winner = np.random.choice(pred_winners)
                else:
                    pred_winner = pred_winners
        return x[x['algorithm']==pred_winner][metric].iloc[0]
    pred_err = test_df.groupby(identifier)[[metric, 'algorithm' ,'pred_alg', 'prob']].apply(lambda x: get_pred_metric(x))
    print(f"Eval wg err: {pred_err.mean()}")
    return pred_err.mean()