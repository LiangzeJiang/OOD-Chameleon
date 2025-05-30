import torch
import numpy as np
import torch.nn.functional as F


def mixupsample(x, y, a, alpha=2.0):

    def _mix_up(alpha, x1, x2, y1, y2):
        length = min(len(x1), len(x2))
        x1 = x1[:length]
        x2 = x2[:length]
        y1 = y1[:length]
        y2 = y2[:length]

        # n_classes = y1.shape[1]
        bsz = len(x1)
        l = np.random.beta(alpha, alpha, [bsz, 1])
        if len(x1.shape) == 4:
            l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
        else:
            l_x = np.tile(l, (1, *x1.shape[1:]))
        # l_y = np.tile(l, [1, n_classes])
        l_y = l.squeeze()

        # mixed_input = l * x + (1 - l) * x2
        mixed_x = l_x * x1 + (1 - l_x) * x2
        mixed_y = l_y * y1 + (1 - l_y) * y2

        return mixed_x, mixed_y

    fn = _mix_up

    all_mix_x, all_mix_y = [], []
    bs = len(x)
    # repeat until enough samples
    while sum(list(map(len, all_mix_x))) < bs:
        start_len = sum(list(map(len, all_mix_x)))
        s = np.random.random() <= 0.5  # self.hparams["LISA_p_sel"]
        # same label, mixup between attributes
        if s:
            for y_i in np.unique(y):
                mask = y == y_i
                x_i, y_i, a_i = x[mask], y[mask], a[mask]
                unique_a_is = np.unique(a_i)
                # # if there are multiple attributes, choose a random pair
                # a_i1, a_i2 = unique_a_is[np.randperm(len(unique_a_is))][:2]
                a_i1, a_i2 = unique_a_is[0], unique_a_is[1]
                mask2_1 = a_i == a_i1
                mask2_2 = a_i == a_i2
                all_mix_x_i, all_mix_y_i = fn(
                    alpha, x_i[mask2_1], x_i[mask2_2], y_i[mask2_1], y_i[mask2_2]
                )
                all_mix_x.append(all_mix_x_i)
                all_mix_y.append(all_mix_y_i)
        # same attribute, mixup between labels
        else:
            for a_i in np.unique(a):
                mask = a == a_i
                x_i, y_i = x[mask], y[mask]
                unique_y_is = np.unique(y)
                # # if there are multiple labels, choose a random pair
                # y_i1, y_i2 = unique_y_is[np.randperm(len(unique_y_is))][:2]
                y_i1, y_i2 = unique_y_is[0], unique_y_is[1]
                # mask2_1 = y_i[:, y_i1].squeeze().bool()
                # mask2_2 = y_i[:, y_i2].squeeze().bool()
                mask2_1 = y_i == y_i1
                mask2_2 = y_i == y_i2
                all_mix_x_i, all_mix_y_i = fn(
                    alpha, x_i[mask2_1], x_i[mask2_2], y_i[mask2_1], y_i[mask2_2]
                )
                all_mix_x.append(all_mix_x_i)
                all_mix_y.append(all_mix_y_i)

        end_len = sum(list(map(len, all_mix_x)))
        # each attribute only has one unique label
        if end_len == start_len:
            return x, y

    all_mix_x = np.concatenate(all_mix_x, axis=0)
    all_mix_y = np.concatenate(all_mix_y, axis=0)

    # shuffle the mixed samples
    all_mix_x = all_mix_x[np.random.permutation(len(all_mix_x))]
    all_mix_y = all_mix_y[np.random.permutation(len(all_mix_y))]

    return all_mix_x[:bs], all_mix_y[:bs]


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


def irm_penalty(logits, y):
    device = "cuda" if logits[0][0].is_cuda else "cpu"
    scale = torch.tensor(1.0).to(device).requires_grad_()
    loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
    loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    grad_1 = torch.autograd.grad(loss_1, [scale], create_graph=True)[0]
    grad_2 = torch.autograd.grad(loss_2, [scale], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)
    return result


def irm_loss(logits, y, a, step, penalty_weight=1e2, penalty_anneal_iters=500):
    nll, penalty = 0.0, 0.0
    penalty_weight = 1.0 if step < penalty_anneal_iters else penalty_weight
    print(torch.unique(a))
    for a_val in torch.unique(a):
        # find corresponding indices
        idx_samples = a == a_val
        nll += F.cross_entropy(logits[idx_samples], y[idx_samples])
        penalty += irm_penalty(logits[idx_samples], y[idx_samples])
    nll /= len(a.unique())
    penalty /= len(a.unique())
    loss_value = nll + (penalty_weight * penalty)

    # update_count += 1
    return loss_value
