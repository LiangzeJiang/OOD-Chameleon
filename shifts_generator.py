import os

import argparse
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import generate_task_metadata


class TaskGenerator:
    """
    Given an original dataset's metadata represented by a csv file, generate a set of tasks.
    The entry in metadata file is normally in the form of (id, filename, label, attribute).
    Each resulting task is essentially a constraint-subsetted version of the original dataset, .

    Function pipeline() is the main function to generate a task.
    Function generate_valid_dist_shifts() is used to generate valid degrees of distribution shifts (we used it to generate the degrees of shifts in `configs/`).
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.num_tasks = 0
        np.random.seed(self.seed)

        # define the group matrix for solving group size in different conditions.
        # {(0, 0), (0, 1), (1, 0), (1, 1)}
        self.group_matrix = np.array(
            [
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ]
        )

    def pipeline(
        self,
        meta_info,
        sc,
        ci,
        ai,
        datasize,
        add_val=False,
        output_path=None,
        note=None,
    ):
        """Traverse the task grid and generate all tasks."""
        all_metadata = pd.read_csv(meta_info)
        # ignore the original split, if exists
        if "split" in all_metadata.columns:
            all_metadata.drop(columns=["split"], inplace=True)
        # get train/val/test size
        group_dict = all_metadata.groupby(["y", "a"]).groups

        task = self.generate(sc, ci, ai, group_dict, all_metadata, datasize, add_val)

        if output_path is not None:
            if task is None:
                # meaning the task is invalid, i.e., number of samples for some group is not enough
                with open(os.path.join(output_path, "task_invalid"), "w") as _:
                    pass
            else:
                self.num_tasks += 1
                self.save(os.path.join(output_path, "metadata.csv"), task)
            if note is not None:
                with open(os.path.join(output_path, note), "w") as _:
                    pass

    def generate(self, sc, ci, ai, group_dict, metadata, datasize, add_val):
        """Main function to generate a single task given sc/ci/ai statistics."""
        print(
            "Generating task with n={}, sc={}, ci={}, ai={}".format(
                datasize, sc, ci, ai
            )
        )
        tr_size, te_size = datasize, int(datasize / 4)
        te_num_per_group = int(te_size / len(group_dict))

        b = [sc * tr_size, ai * tr_size, ci * tr_size, tr_size]
        num_per_group = np.linalg.solve(self.group_matrix, b)
        num_per_group = [int(n) for n in num_per_group]
        print("num of sample per group in training set: ", num_per_group)

        # assert all numbers are positive
        assert all([n >= 0 for n in num_per_group])

        tr, val, te = [], [], []
        for i, (_, v) in enumerate(group_dict.items()):
            np.random.shuffle(np.array(v))

            required_num = num_per_group[i] + te_num_per_group
            if required_num - len(v) > 0:
                return None
            else:
                tr.extend(v[: num_per_group[i]])
                te.extend(v[num_per_group[i] : required_num])

            if add_val:
                val.extend(
                    v[
                        num_per_group[i]
                        + te_num_per_group : num_per_group[i]
                        + te_num_per_group
                        + int(num_per_group) / 4
                    ]
                )

        # assert the size of data
        assert tr_size - len(tr) <= len(group_dict)
        assert te_size - len(te) <= len(group_dict)

        # extract from metadata with group_dict
        tr_data = metadata.iloc[tr]
        te_data = metadata.iloc[te]
        # formulate the task
        tr_data["split"], te_data["split"] = 0, 2

        if add_val:
            val_data = metadata.iloc[val]
            val_data["split"] = 1
            task = pd.concat([tr_data, val_data, te_data])
        else:
            task = pd.concat([tr_data, te_data])
        return task

    def save(self, output_file, task):
        """Save the tasks to a csv file."""
        task.to_csv(output_file, index=False)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="parameters specifying different types and magnitudes of distribution shifts we want to create."
    )
    parser.add_argument("--data_name", type=str, default="celeba")
    parser.add_argument(
        "--data_dir", type=str, help="directory that stores the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="directory to store the outputs",
        default="../data_src",
    )
    parser.add_argument("--seed", type=int, default=0)
    ##### ONLY VALID FOR OFFICEHOME DATASET #####
    parser.add_argument("--num_tasks", type=int, default=100)
    ##### ONLY VALID FOR OTHER DATASETS #####
    parser.add_argument("--data_size", type=int, nargs="+")
    parser.add_argument("--sc", type=float, nargs="+")
    parser.add_argument("--ci", type=float, nargs="+")
    parser.add_argument("--ai", type=float, nargs="+")
    parser.add_argument("--task_y", type=int, nargs="+")
    parser.add_argument("--task_a", type=int, nargs="+")
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    ######
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    if args.data_name.lower() != "officehome":
        seed, data_dir, data_name, note = (
            args.seed,
            args.data_dir,
            args.data_name.lower(),
            args.note,
        )
        task_generator = TaskGenerator(seed)

        # zip shifts and tasks
        shifts = list(zip(args.sc, args.ci, args.ai))
        tasks = list(zip(args.task_y, args.task_a))
        combinations = itertools.product(args.data_size, tasks, shifts)

        all_data_info, all_output_dir = {}, {}
        # generate metadata for tasks
        for task_y, task_a in tasks:
            current_output_dir = os.path.join(
                args.output_dir, data_name, f"tasks_y{task_y}_a{task_a}"
            )
            if not os.path.exists(current_output_dir):
                os.makedirs(current_output_dir)
            all_output_dir[f"y_{task_y}_a_{task_a}"] = current_output_dir
            # form the metadata, in the form of [id, filename, split, label, attribute]
            all_data_info[f"y_{task_y}_a_{task_a}"] = generate_task_metadata(
                data_name, data_dir, current_output_dir, task_y, task_a
            )

        valid_tasks = 0
        # for loop with tqdm
        for data_size, (task_y, task_a), (sc, ci, ai) in tqdm(list(combinations)):
            data_info = all_data_info[f"y_{task_y}_a_{task_a}"]
            output_dir = all_output_dir[f"y_{task_y}_a_{task_a}"]
            task_dir = os.path.join(
                output_dir,
                "task_{}_sc{:.2f}_ci{:.2f}_ai{:.2f}".format(data_size, sc, ci, ai),
            )

            # check if the path exists
            if not os.path.exists(task_dir):
                os.makedirs(task_dir)
            else:
                # check if 'metadata.csv' exists
                if os.path.exists(os.path.join(task_dir, "metadata.csv")):
                    print("Task already exists. Exiting...")
                    valid_tasks += 1
                    continue
            task_generator.pipeline(
                meta_info=data_info,
                sc=sc,
                ci=ci,
                ai=ai,
                datasize=data_size,
                output_path=task_dir,
                note=note,
            )
            print("Current processed valid tasks: ", task_generator.num_tasks)
        print("Saved valid tasks: ", valid_tasks + task_generator.num_tasks)
    else:
        # officehome does not use TaskGenerator
        domains = ['Art', 'Clipart', 'Product', 'Real World']
        class_list = os.listdir(os.path.join(args.data_dir, args.data_name.lower(), domains[0]))

        curr_n = 0
        while curr_n < args.num_tasks:
            print("processing task: ", curr_n)
            curr_n += 1
            # sample two domains
            ds = np.random.choice(domains, 2, replace=False)
            # sample two classes
            cs = np.random.choice(class_list, 2, replace=False)
            # collect the samples
            df_metadata = pd.DataFrame()
            for d in ds:
                for c in cs:
                    img_files = os.listdir(os.path.join(args.data_dir, args.data_name.lower(), d, c))
                    img_files = [os.path.join(d, c, img) for img in img_files]
                    curr_metadata = {
                        "filename": img_files,
                        "y": [0] * len(img_files) if c == cs[0] else [1] * len(img_files),
                        "a": [0] * len(img_files) if d == ds[0] else [1] * len(img_files),
                    }
                    df_metadata = pd.concat(
                        [df_metadata, pd.DataFrame(curr_metadata)], ignore_index=True
                    )
            # check if the samples are enough
            if min(df_metadata.groupby(["y", "a"]).size()) < 30:
                curr_n -= 1
                continue
            # prepare the split, randomly take 15 samples from each group as test set, and the remaining as train, set split column accordingly.
            df_metadata["split"] = 0
            df_metadata.loc[df_metadata.groupby(['y', 'a']).sample(n=15).index, 'split'] = 2
            # infer the distribution shift statistics and data size from df_metadata (split = 0)
            tr_df = df_metadata[df_metadata["split"] == 0]
            data_size = len(tr_df)
            ci = len(tr_df[tr_df["y"] == 0]) / data_size
            ai = len(tr_df[tr_df["a"] == 0]) / data_size
            sc = (len(tr_df[(tr_df["y"] == 0) & (tr_df["a"] == 0)]) + len(tr_df[(tr_df["y"] == 1) & (tr_df["a"] == 1)])) / data_size
            # save the metadata
            task_dir = os.path.join(
                args.output_dir, args.data_name.lower(), f"tasks_y-1_a-1",
                "task_{}_sc{:.2f}_ci{:.2f}_ai{:.2f}".format(data_size, sc, ci, ai),
            )
            # check if the path exists
            if not os.path.exists(task_dir):
                os.makedirs(task_dir)
            else:
                # check if 'metadata.csv' exists
                if os.path.exists(os.path.join(task_dir, "metadata.csv")):
                    print("Task already exists...")
                    continue
            # save the df_metadata
            df_metadata["id"] = df_metadata.index
            p = os.path.join(task_dir, "metadata.csv")
            df_metadata.to_csv(p, index=False)
