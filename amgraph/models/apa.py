from collections import defaultdict
from itertools import permutations
import random
import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian, remove_self_loops


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


def get_propagation_matrix(edge_index: Adj, n_nodes: int, mode: str = "adj") -> torch.sparse.FloatTensor:
    edge_index, edge_weight = get_laplacian(edge_index, num_nodes=n_nodes, normalization="sym")
    if mode == "adj":
        edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
    adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj


def get_edge_index_from_y(y: torch.Tensor, know_mask: torch.Tensor = None) -> Adj:
    nodes = defaultdict(list)
    label_idx_iter = enumerate(y.numpy()) if know_mask is None else zip(know_mask.numpy(),y[know_mask].numpy())
    for idx, label in label_idx_iter:
        nodes[label].append(idx)
    arr = []
    for v in nodes.values():
        arr += list(permutations(v, 2))
    return torch.tensor(arr, dtype=torch.long).T


def get_edge_index_from_y_ratio(y: torch.Tensor, ratio: float = 1.0) -> (Adj, torch.Tensor):
    n = y.size(0)
    mask = []
    nodes = defaultdict(list)
    for idx, label in random.sample(list(enumerate(y.numpy())), int(ratio*n)):
        mask.append(idx)
        nodes[label].append(idx)
    arr = []
    for v in nodes.values():
        arr += list(permutations(v, 2))
    return torch.tensor(arr, dtype=torch.long).T, torch.tensor(mask, dtype=torch.long)


def to_dirichlet_loss(attrs, laplacian):
    return torch.bmm(attrs.t().unsqueeze(1), laplacian.matmul(attrs).t().unsqueeze(2)).view(-1).sum()


class APA:
    def __init__(self, edge_index: Adj, x: torch.Tensor, know_mask: torch.Tensor, is_binary: bool, is_connect: bool=False):
        self.edge_index = edge_index
        self.x = x
        self.n_nodes = x.size(0)
        self.know_mask = know_mask
        self.is_connect = is_connect
        self.mean = 0 if is_binary else x[know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (self.x[self.know_mask] - self.mean) / self.std
        self.out = torch.zeros_like(self.x)
        self.out[self.know_mask] = self.x[self.know_mask]
        self._adj = None
        self._unlearn_mask = self._unlearn_mask()

    def _unlearn_mask(self):
        num_iter = 30
        out = torch.zeros(self.n_nodes, 1)
        out[self.know_mask] = 1
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
        return torch.nonzero(out == 0, as_tuple=True)[0]

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def fp(self, out: torch.Tensor = None, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out.clone().detach()
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def fp_analytical_solution(self, **kw) -> torch.Tensor:
        adj = self.adj.to_dense()

        assert self.know_mask.dtype == torch.int64
        know_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        know_mask[self.know_mask] = True
        unknow_mask = torch.ones(self.n_nodes, dtype=torch.bool)
        unknow_mask[self.know_mask] = False

        A_uu = adj[unknow_mask][:, unknow_mask]
        A_uk = adj[unknow_mask][:, know_mask]

        L_uu = torch.eye(unknow_mask.sum()) - A_uu
        L_inv = torch.linalg.inv(L_uu)

        out = self.out.clone()
        out[unknow_mask] = torch.mm(torch.mm(L_inv, A_uk), self.out_k_init)

        return out * self.std + self.mean

    def pr(self, out: torch.Tensor = None, alpha: float = 0.999, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out.clone().detach()
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def ppr(self, out: torch.Tensor = None, alpha: float = 0.999, weight: torch.Tensor = None, num_iter: int = 1,
            **kw) -> torch.Tensor:
        if out is None:
            out = self.out.clone().detach()
        out = (out - self.mean) / self.std
        if weight is None:
            weight = self.mean
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * weight
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def mtp_partial(self, out: torch.Tensor = None, beta: float = 0.999, num_iter: int = 1, **kw) -> torch.Tensor:
        # 相比mtp，将未学到信息的点属性直接填充为有信息点属性的均值
        if out is None:
            out = self.out.clone().detach()
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self._unlearn_mask] = out[~self._unlearn_mask].mean(dim=0)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        out = out * self.std + self.mean
        return out

    def mtp(self, out: torch.Tensor = None, beta: float = 0.999, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out.clone().detach()
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def mtp_analytical_solution(self, beta: float = 0.999, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        eta = (1 / beta - 1)
        edge_index, edge_weight = get_laplacian(self.edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.know_mask] = 1
        Ik = torch.diag(Ik_diag)
        out = (self.out - self.mean) / self.std
        out = torch.mm(torch.inverse(L + eta * Ik), eta * torch.mm(Ik, out))
        return out * self.std + self.mean

    def umtp(self, out: torch.Tensor = None, alpha: float = 0.999, beta: float = 0.70, num_iter: int = 1,
             **kw) -> torch.Tensor:
        if out is None:
            out = self.out.clone().detach()
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def umtp_beta(self, out: torch.Tensor = None, beta: float = 0.70, num_iter: int = 1, **kw) -> torch.Tensor:
        alpha: float = 1.0 if self.is_connect else 0.999
        if out is None:
            out = self.out.clone().detach()
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def umtp2(self, out: torch.Tensor = None, alpha: float = 0.999, beta: float = 0.70, gamma: float = 0.75,
              num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out.clone().detach()
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma * (alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)) + (1 - gamma) * out
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def umtp_analytical_solution(self, alpha: float = 0.999, beta: float = 0.70, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        theta = (1 - 1 / self.n_nodes) * (1 / alpha - 1)
        eta = (1 / beta - 1) / alpha
        edge_index, edge_weight = get_laplacian(self.edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.know_mask] = 1
        Ik = torch.diag(Ik_diag)
        L1 = torch.eye(n_nodes) * (n_nodes / (n_nodes - 1)) - torch.ones(n_nodes, n_nodes) / (n_nodes - 1)
        out = (self.out - self.mean) / self.std
        out = torch.mm(torch.inverse(L + eta * Ik + theta * L1), eta * torch.mm(Ik, out))
        return out * self.std + self.mean


class UMTPLabel:

    def __init__(self, edge_index: Adj, x: torch.Tensor, y: torch.Tensor, know_mask: torch.Tensor, is_binary: bool, is_connect: bool=False):
        self.x = x
        self.y = y
        self.n_nodes = x.size(0)
        self.edge_index = edge_index
        self.is_connect = is_connect
        self._adj = None

        self._label_adj = None
        self._label_adj_25 = None
        self._label_adj_50 = None
        self._label_adj_75 = None
        self._label_adj_all = None
        self._label_mask = know_mask
        self._label_mask_25 = None
        self._label_mask_50 = None
        self._label_mask_75 = None

        self.know_mask = know_mask
        self.mean = 0 if is_binary else self.x[self.know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (self.x[self.know_mask]-self.mean) / self.std
        # init self.out without normalized
        self.out = torch.zeros_like(self.x)
        self.out[self.know_mask] = self.x[self.know_mask]

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def label_adj(self):
        if self._label_adj is None:
            edge_index = get_edge_index_from_y(self.y, self.know_mask)
            self._label_adj = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj, self._label_mask
    
    def label_adj_25(self):
        if self._label_adj_25 is None:
            _, label_mask_50 = self.label_adj_50()
            self._label_mask_25 = torch.tensor(random.sample(label_mask_50.tolist(), int(0.5*label_mask_50.size(0))),dtype=torch.long)
            edge_index = get_edge_index_from_y(self.y, self._label_mask_25)
            self._label_adj_25 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_25, self._label_mask_25

    def label_adj_50(self):
        if self._label_adj_50 is None:
            _, label_mask_75 = self.label_adj_75()
            self._label_mask_50 = torch.tensor(random.sample(label_mask_75.tolist(), int(0.75*label_mask_75.size(0))),dtype=torch.long)
            edge_index = get_edge_index_from_y(self.y, self._label_mask_50)
            self._label_adj_50 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_50, self._label_mask_50

    def label_adj_75(self):
        if self._label_adj_75 is None:
            edge_index, self._label_mask_75 = get_edge_index_from_y_ratio(self.y, 0.75)
            self._label_adj_75 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_75, self._label_mask_75

    @property
    def label_adj_all(self):
        if self._label_adj_all is None:
            edge_index = get_edge_index_from_y(self.y)
            self._label_adj_all = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_all

    def umtp(self, out: torch.Tensor = None, beta: float = 0.70, num_iter: int = 1, **kw) -> torch.Tensor:
        alpha: float = 1.0 if self.is_connect else 0.999
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean

    def _umtp_label(self, adj: Adj, mask:torch.Tensor, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1):
        alpha: float = 1.0 if self.is_connect else 0.999
        G = torch.ones(self.n_nodes)
        G[mask] = gamma
        G = G.unsqueeze(1)
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = G*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)) + (1-G)*torch.spmm(adj, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean

    def umtp_label_25(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_25()
        return self._umtp_label(adj,mask,out,beta,gamma,num_iter)

    def umtp_label(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj()
        return self._umtp_label(adj,mask,out,beta,gamma,num_iter)

    def umtp_label_50(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_50()
        return self._umtp_label(adj,mask,out,beta,gamma,num_iter)

    def umtp_label_75(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_75()
        return self._umtp_label(adj,mask,out,beta,gamma,num_iter)

    def umtp_label_100(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        alpha: float = 1.0 if self.is_connect else 0.999
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)) + (1-gamma)*torch.spmm(self.label_adj_all, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean
    
    def umtp_label_all(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*torch.spmm(self.adj, out) + (1-gamma)*torch.spmm(self.label_adj_all, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean


class UMTPLoss(nn.Module):
    def __init__(self, edge_index:Adj, raw_x:torch.Tensor, know_mask:torch.Tensor, alpha, beta, is_binary:bool, **kw):
        super().__init__()
        num_nodes = raw_x.size(0)
        self.n_nodes = num_nodes
        num_attrs = raw_x.size(1)
        self.know_mask = know_mask

        self.mean = 0 if is_binary else raw_x[know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (raw_x[know_mask]-self.mean) / self.std

        edge_index, edge_weight = get_laplacian(edge_index, num_nodes=num_nodes, normalization="sym")
        self.L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(num_nodes, num_nodes)).to_dense().to(edge_index.device)
        self.avg_L = num_nodes/(num_nodes-1)*torch.eye(num_nodes) - 1/(num_nodes-1)*torch.ones(num_nodes, num_nodes)
        self.x = nn.Parameter(torch.zeros(num_nodes, num_attrs))
        self.x.data[know_mask] = raw_x[know_mask].clone().detach().data
        if alpha == 0:
            alpha = 0.00001
        if beta == 0:
            beta = 0.00001
        self.theta = (1 - 1/num_nodes) * (1/alpha - 1)
        self.eta = (1/beta - 1)/alpha
        print(alpha, beta, self.theta, self.eta)

    def get_loss(self, x):
        x = (x - self.mean)/self.std
        dirichlet_loss = to_dirichlet_loss(x, self.L)
        avg_loss = to_dirichlet_loss(x, self.avg_L)
        recon_loss = nn.functional.mse_loss(x[self.know_mask], self.out_k_init, reduction="sum")
        return dirichlet_loss + self.eta * recon_loss + self.theta * avg_loss

    def forward(self):
        return self.get_loss(self.x)
    
    def get_out(self):
        return self.x


class UMTPwithParams(nn.Module):

    def __init__(self, x: torch.Tensor, y: torch.Tensor, edge_index: Adj, know_mask: torch.Tensor, is_binary: bool):
        super().__init__()
        self.x = x
        self.y = y
        self.n_nodes = x.size(0)
        self.n_attrs = x.size(1)
        self.edge_index = edge_index
        self._adj = None
        self.know_mask = know_mask
        self.is_binary = is_binary
        self.mean = 0 if is_binary else x[know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (x[know_mask]-self.mean) / self.std
        # init self.out without ormalized
        self.out = torch.zeros_like(x)
        self.out[know_mask] = x[know_mask]
        # parameters
        self.eta, self.theta = nn.Parameter(torch.zeros(self.n_attrs)), nn.Parameter(torch.zeros(self.n_attrs))

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def forward(self, num_iter: int = 30) -> torch.Tensor:
        alpha = (self.n_nodes-1)/(self.theta*self.n_nodes+self.n_nodes-1)
        beta = 1/alpha / (1/alpha+self.eta)
        out = (self.out.clone().detach() - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean
    


