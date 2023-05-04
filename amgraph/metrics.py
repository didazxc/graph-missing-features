from .data import data as d
import numpy as np
import torch
from sklearn.metrics import ndcg_score


@torch.no_grad()
def to_acc(y_hat: torch.Tensor, y: torch.Tensor):
    return torch.sum(y_hat == y) / y.size(0)


@torch.no_grad()
def to_recall(input, target, k=10):
    pred = input.topk(k, dim=1, sorted=False)[1]
    row_index = torch.arange(target.size(0))
    target_list = []
    for i in range(k):
        target_list.append(target[row_index, pred[:, i]])
    num_pred = torch.stack(target_list, dim=1).sum(dim=1)
    num_true = target.sum(dim=1)
    return (num_pred[num_true > 0] / num_true[num_true > 0]).mean().item()


@torch.no_grad()
def to_ndcg(input, target, k=10, version='sat'):
    if version == 'base':
        return ndcg_score(target, input, k=k)
    elif version == 'sat':
        device = target.device
        target_sorted = torch.sort(target, dim=1, descending=True)[0]
        pred_index = torch.topk(input, k, sorted=True)[1]
        row_index = torch.arange(target.size(0))
        dcg = torch.zeros(target.size(0), device=device)
        for i in range(k):
            dcg += target[row_index, pred_index[:, i]] / np.log2(i + 2)
        idcg_divider = torch.log2(torch.arange(target.size(1), dtype=float, device=device) + 2)
        idcg = (target_sorted / idcg_divider).sum(dim=1)
        return (dcg[idcg > 0] / idcg[idcg > 0]).mean().item()
    else:
        raise ValueError(version)


@torch.no_grad()
def to_rmse(input, target):
    return ((input - target) ** 2).mean(dim=1).sqrt().mean().item()


@torch.no_grad()
def to_r2(input, target):
    a = ((input - target) ** 2).sum()
    b = ((target - target.mean(dim=0)) ** 2).sum()
    return (1 - a / b).item()


@torch.no_grad()
def calc_single_score(dataset_name, x_hat, x_all, nodes, metric, k):
    if d.is_continuous(dataset_name):
        if metric == "CORR":
            return to_r2(x_hat[nodes], x_all[nodes])
        elif metric == "RMSE":
            return to_rmse(x_hat[nodes], x_all[nodes])
        else:
            raise Exception(f"no this metric {metric}")
    else:
        if metric == "nDCG":
            return to_ndcg(x_hat[nodes], x_all[nodes], k=k)
        elif metric == "Recall":
            return to_recall(x_hat[nodes], x_all[nodes], k=k)
        elif metric == "CORR":
            return to_r2(x_hat[nodes], x_all[nodes])
        else:
            raise Exception(f"no this metric {metric}")


def greater_is_better(metric):
    return metric != "RMSE"
