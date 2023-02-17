import data as d
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from torch import optim
from sklearn.metrics import ndcg_score
from torch_scatter import scatter_add
from torch_geometric.typing import Adj, OptTensor
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, remove_self_loops
"""
https://github.com/twitter-research/feature-propagation
"""


@torch.no_grad()
def to_recall(input, target, k=10):
    """
    Compute the recall score from a prediction.
    """
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
    """
    Compute the NDCG score from a prediction.
    """
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


class Scores:

    def __init__(self, x_all, trn_nodes, val_nodes, test_nodes) -> None:
        self.trn_nodes = trn_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.datas = {"trn": trn_nodes, "val": val_nodes, "tst": test_nodes}
        self.x_all = x_all
        self.dict = {}

    def _calc_scores(self, x_hat, dataset_name, nodes, k):
        if d.is_continuous(dataset_name):
            return f"{dataset_name}@CORR", to_r2(x_hat[nodes], self.x_all[nodes])
        else:
            return f"{dataset_name}@{k}", to_recall(x_hat[nodes], self.x_all[nodes], k=k)

    def _validate(self, x_hat, dataset_name, algo_name, k, datas):
        for data_name in datas:
            col, score = self._calc_scores(x_hat, dataset_name, self.datas[data_name], k)
            col = col if data_name=='tst' else f"{col}_{data_name}"
            scores._add_score(algo_name, col, score)

    def validate(self, x_hat, dataset_name, algo_name, ks=[10,20,50], datas=['tst']):
        if d.is_continuous(dataset_name):
            ks = [10]
        for k in ks:
            self._validate(x_hat, dataset_name, algo_name, k, datas)
    
    def validate_best(self, apa_fn, edge_index, alphas, dataset_name, algo_name, ks=[10,20,50], datas=['tst']):
        if d.is_continuous(dataset_name):
            ks = [10]
        for k in ks:
            score = None
            best_x_hat = None
            for alpha in alphas:
                x_hat = apa_fn(self.x_all, edge_index, self.trn_nodes, alpha)
                _, curr_score = self._calc_scores(x_hat, dataset_name, self.val_nodes, k)
                if score is None or curr_score>score:
                    score = curr_score
                    best_x_hat = x_hat
            self._validate(best_x_hat, dataset_name, algo_name, k, datas)
    
    def _add_score(self, row, col, value):
        if col not in self.dict:
            self.dict[col] = {}
        if row not in self.dict[col]:
            self.dict[col][row] = [value]
        else:
            self.dict[col][row].append(value)

    def print(self, file_name=None):
        df_dict = {col:{row: np.mean(row_v) for row, row_v in col_v.items()} for col, col_v in self.dict.items()}
        df = pd.DataFrame(df_dict)
        if file_name is not None:
            df.to_csv("{file_name}.csv")
        print(df)


def get_normalized_adjacency(edge_index, n_nodes, mode=None):
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    if mode == "left":
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight
    elif mode == "right":
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = edge_weight * deg_inv_sqrt[col]
    elif mode == "article_rank":
        d = deg.mean()
        deg_inv_sqrt = (deg+d).pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight
    else:
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD


def get_propagation_matrix(edge_index, n_nodes, mode="adj"):
    edge_index, edge_weight = get_laplacian(edge_index, num_nodes=n_nodes, normalization="sym")
    if mode == "adj":
        edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
    # edge_index, edge_weight = get_normalized_adjacency(edge_index, n_nodes)
    adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj


class APA(nn.Module):

    def __init__(self, num_iterations: int):
        super().__init__()
        self.num_iterations = num_iterations

    def fp(self, x: torch.Tensor, edge_index: Adj, known_feature_mask: torch.Tensor) -> torch.Tensor:
        if known_feature_mask is not None:
            out = torch.zeros_like(x)
            out[known_feature_mask] = x[known_feature_mask]
        else:
            out = x.clone()

        n_nodes = x.shape[0]
        adj = get_propagation_matrix(edge_index, n_nodes)
        for _ in range(self.num_iterations):
            out = torch.spmm(adj, out)
            out[known_feature_mask] = x[known_feature_mask]

        return out

    def fp_analytical_solution(self, x: torch.Tensor, edge_index: Adj, known_feature_mask: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)

        adj = get_propagation_matrix(edge_index, n_nodes).to_dense()

        assert known_feature_mask.dtype == torch.int64
        known_mask = torch.zeros(n_nodes, dtype=torch.bool)
        known_mask[known_feature_mask] = True
        unknow_mask = torch.ones(n_nodes, dtype=torch.bool)
        unknow_mask[known_feature_mask] = False

        A_uu = adj[unknow_mask][:,unknow_mask]
        A_uk = adj[unknow_mask][:,known_mask]

        L_uu = torch.eye(unknow_mask.sum())-A_uu
        L_inv = torch.linalg.inv(L_uu)

        out = x.clone()
        out[unknow_mask] = torch.mm(torch.mm(L_inv, A_uk), out[known_mask])

        return out

    def ppr(self, x: torch.Tensor, edge_index: Adj, known_feature_mask: torch.Tensor, alpha: float = 0.85, weight = None) -> torch.Tensor:
        n_nodes = x.size(0)
        adj = get_propagation_matrix(edge_index, n_nodes)
        if weight is None:
            weight = x[known_feature_mask].mean(dim=0)

        out = torch.zeros_like(x)
        out[known_feature_mask] = x[known_feature_mask]

        for _ in range(self.num_iterations):
            out = alpha * torch.spmm(adj, out) + (1-alpha)*weight  
            out[known_feature_mask] = x[known_feature_mask]

        return out

    def pr(self, x: torch.Tensor, edge_index: Adj, known_feature_mask: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
        n_nodes = x.size(0)
        adj = get_propagation_matrix(edge_index, n_nodes)

        out = torch.zeros_like(x)
        out[known_feature_mask] = x[known_feature_mask]

        for _ in range(self.num_iterations):
            #seed=0| out.mean(dim=0):0.1732 x[known_feature_mask].mean(dim=0):0.1738;
            out = alpha * torch.spmm(adj, out) + (1-alpha)*out.mean(dim=0)  
            out[known_feature_mask] = x[known_feature_mask]

        return out

    def mtp(self, x: torch.Tensor, edge_index: Adj, known_feature_mask: torch.Tensor, alpha: float = 0.83) -> torch.Tensor:
        n_nodes = x.size(0)
        adj = get_propagation_matrix(edge_index, n_nodes)

        out = torch.zeros_like(x)
        out_init = x[known_feature_mask]
        out[known_feature_mask] = out_init

        for _ in range(self.num_iterations):
            # seed=0| 0.1843; 0.1773
            out = torch.spmm(adj, out)
            out[known_feature_mask] = alpha*out[known_feature_mask] + (1-alpha)*out_init
        
        return out



if __name__=="__main__":

    run_baseline = False
    run_pr = False
    run_mtp = True
    alphas=np.arange(0, 1, step = 0.02)
    dataset_names = ['cora', 'citeseer', 'computers', 'photo', 'pubmed', 'cs', 'steam', 'arxiv']

    for seed in range(10):
        print(seed)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for dataset_name in dataset_names:
            edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes = d.load_data(dataset_name, split=(0.4, 0.1, 0.5), seed=seed)
            scores = Scores(x_all, trn_nodes, val_nodes, test_nodes)
            apa = APA(50)

            ks=[3, 5, 10] if dataset_name=="steam" else [10, 20, 50]
            datas=['trn', 'val', 'tst']

            if run_baseline:
                x_hat = apa.fp(x_all, edge_index, trn_nodes)
                scores.validate(x_hat, dataset_name, "fp", ks=ks, datas=datas)
            
            if run_pr:
                scores.validate_best(apa.pr, edge_index, alphas, dataset_name, "pr", ks=ks, datas=datas)

            if run_mtp:
                scores.validate_best(apa.mtp, edge_index, alphas, dataset_name, "mtp", ks=ks, datas=datas)

    scores.print("mtp")
