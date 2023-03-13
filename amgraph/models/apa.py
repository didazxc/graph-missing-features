import torch
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


class APA:

    def __init__(self, x: torch.Tensor, edge_index: Adj, known_mask: torch.Tensor, is_binary: bool, out_init=None):
        self.x = x
        self.n_nodes = x.size(0)
        self.edge_index = edge_index
        self._adj = None
        self.known_mask = known_mask
        self.mean = 0 if is_binary else x[known_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[known_mask].std(dim=0)
        self.out_k_init = (x[known_mask]-self.mean) / self.std
        # init
        if out_init is None:
            self.out = torch.zeros_like(x)
            self.out[known_mask] = x[known_mask]
        else:
            self.out = out_init
    
    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def fp(self, out: torch.Tensor = None, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            return self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.known_mask] = self.out_k_init
        return out * self.std + self.mean

    def fp_analytical_solution(self, **kw) -> torch.Tensor:
        adj = self.adj.to_dense()

        assert self.known_mask.dtype == torch.int64
        known_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        known_mask[self.known_mask] = True
        unknow_mask = torch.ones(self.n_nodes, dtype=torch.bool)
        unknow_mask[self.known_mask] = False

        A_uu = adj[unknow_mask][:,unknow_mask]
        A_uk = adj[unknow_mask][:,known_mask]

        L_uu = torch.eye(unknow_mask.sum())-A_uu
        L_inv = torch.linalg.inv(L_uu)

        out = self.out.clone()
        out[unknow_mask] = torch.mm(torch.mm(L_inv, A_uk), self.out_k_init)

        return out * self.std + self.mean

    def pr(self, out: torch.Tensor = None, alpha: float = 0.85, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            return self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1-alpha)*out.mean(dim=0)
            out[self.known_mask] = self.out_k_init
        return out * self.std + self.mean

    def ppr(self, out: torch.Tensor = None, alpha: float = 0.85, weight: torch.Tensor = None, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            return self.out
        out = (out - self.mean) / self.std
        if weight is None:
            weight = self.mean
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1-alpha)*weight
            out[self.known_mask] = self.out_k_init
        return out * self.std + self.mean

    def mtp(self, out: torch.Tensor = None, alpha: float = 0.85, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            return self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.known_mask] = alpha*out[self.known_mask] + (1-alpha)*self.out_k_init
        return out * self.std + self.mean

    def umtp(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            return self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.known_mask] = beta*out[self.known_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean
    
    def umtp2(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            return self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0))+(1-gamma)*out
            out[self.known_mask] = beta*out[self.known_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean

    def umtp_analytical_solution(self, alpha: float = 0.85, beta: float = 0.70, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        theta = (n_nodes-1)*(1-alpha)/alpha/n_nodes
        lamda = 1/(1-beta)-1/theta
        edge_index, edge_weight = get_laplacian(self.edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.known_mask] = 1
        Ik = torch.diag(Ik_diag)
        L1 = torch.eye(n_nodes)*(n_nodes/(n_nodes-1)) - torch.ones(n_nodes, n_nodes)/(n_nodes-1)
        out = torch.mm(torch.inverse(L+lamda*Ik+theta*L1), lamda*torch.mm(Ik, self.out))
        return out
