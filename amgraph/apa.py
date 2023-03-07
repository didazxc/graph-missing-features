from .data import data as d
from .metric import calc_single_score, greater_is_better
from .utils import SearchPoints, Scores, print_time_cost
import itertools
import logging
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian, remove_self_loops


logger = logging.getLogger('g.apa')


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


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, num_layers=2, dropout=0.5) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = output_size if i == num_layers - 1 else hidden_size
            if i > 0:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(in_size, out_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class APA:

    def __init__(self, x: torch.Tensor, edge_index: Adj, known_mask: torch.Tensor, is_binary: bool):
        self.x = x
        self.n_nodes = x.size(0)
        self.edge_index = edge_index
        self.adj = get_propagation_matrix(edge_index, self.n_nodes)
        self.known_mask = known_mask
        self.mean = 0 if is_binary else x[known_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[known_mask].std(dim=0)
        self.out_k_init = (x[known_mask]-self.mean) / self.std
        # init
        self.out = torch.zeros_like(x)
        self.out[known_mask] = self.out_k_init

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


class APA2:

    def __init__(self, x: torch.Tensor, edge_index: Adj, known_mask: torch.Tensor, val_mask:torch.Tensor):
        self.x = x
        self.n_nodes = x.size(0)
        self.edge_index = edge_index
        self.adj = get_propagation_matrix(edge_index, self.n_nodes)
        self.known_mask = known_mask
        self.val_mask = val_mask
        self.out_k_init = x[known_mask]
        self.mean = x[known_mask].mean(dim=0)
        # for Ps
        self.z_ones = torch.zeros(self.n_nodes, 1)
        self.z_ones[self.known_mask] = 1
        self.z_ones_k_init = self.z_ones[self.known_mask].clone()
        # init
        self.out = torch.zeros_like(x)
        self.out[known_mask] = self.out_k_init
        
    def umtp(self, out: torch.Tensor, out_k_init: torch.Tensor, alpha: float = 0.85, beta: float = 0.70, num_iter: int = 1, **kw) -> torch.Tensor:
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.known_mask] = beta*out[self.known_mask] + (1-beta)*out_k_init
        return out

    def run(self, Ps=None, x_hat=None, new_x_hat=None, alpha: float = 0.85, beta: float = 0.70, num_iter: int = 1, **kw):
        if Ps is None:
            Ps = self.z_ones
        if x_hat is None:
            x_hat = self.out
        if new_x_hat is None:
            new_x_hat = self.out
        Ps = self.umtp(Ps, self.z_ones_k_init, alpha, beta, num_iter, **kw)
        x_Ps = Ps[self.known_mask] - 1
        x_hat = self.umtp(x_hat, self.out_k_init, alpha, beta, num_iter, **kw)
        M = torch.mm(x_Ps.T,(x_hat[self.known_mask] - self.x[self.known_mask])) / torch.mm(x_Ps.T, x_Ps)
        # M = self.mean
        new_x_hat = self.umtp(new_x_hat - M, self.out_k_init - M, alpha, beta, num_iter, **kw) + M
        return Ps, x_hat, new_x_hat
 


class UMTPDecoder:
    def __init__(self, x: torch.Tensor, edge_index: Adj, known_mask: torch.Tensor):
        super().__init__()
        self.n_nodes = x.size(0)
        self.n_attrs = x.size(1)
        self.adj = get_propagation_matrix(edge_index, self.n_nodes)
        self.known_mask = known_mask
        self.out_k_init = x[known_mask]
        self.out = torch.zeros_like(x)
        self.out[known_mask] = self.out_k_init
        self.known_mean = self.out_k_init.mean(dim=0)

        self.decoder = Decoder(self.n_attrs, self.n_attrs)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.05)

    def umtp_with_decoder(self, out:torch.Tensor=None, alpha: float=0.85, beta: float=0.70, num_iter: int=1, **kw) -> torch.Tensor:
        if out is None:
            return self.out
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.known_mask] = beta*out[self.known_mask] + (1-beta)*self.out_k_init

        best_loss = None
        no_better_i = 0
        for i_iter in range(50):
            x_hat=self.decoder(out)
            loss = self.loss_fn(x_hat[self.known_mask], self.out_k_init)
            loss_value = loss.item()
            if best_loss is None or loss_value<best_loss:
                no_better_i=0
                best_loss = loss_value
            else:
                no_better_i+=1
            print(f"{i_iter}: loss= {loss_value}, best_loss={best_loss}, no_better_i={no_better_i}")
            if no_better_i>4:
                break
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return x_hat.detach()


def sgd(apa_fn, x, edge_index, known_feature_mask, val_nodes_mask, epoches=50, init_alpha=0.85, init_beta=0.7):
    alpha, beta = torch.tensor([init_alpha], requires_grad=True), torch.tensor([init_beta], requires_grad=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam([alpha, beta], lr=0.05)
    best_loss = None
    best_alpha = None
    best_beta = None
    best_epoch = 0
    x_hat = None
    for epoch in range(epoches):
        a = torch.clamp(alpha, 0.0, 1.0)
        b = torch.clamp(beta, 0.0, 1.0)
        out = apa_fn(x, edge_index, known_feature_mask, alpha=a, beta=b)
        loss=loss_fn(out[val_nodes_mask], x[val_nodes_mask])
        logger.info('epoch:', epoch, 'loss:', loss.item(), 'alpha', a.item(), 'beta', b.item())
        if best_loss is None or loss.item()<best_loss:
            best_epoch = epoch
            best_loss = loss.item()
            best_alpha = alpha.item()
            best_beta = beta.item()
            logger.info('best_epoch', best_epoch, 'loss', loss.item(), 'alpha', best_alpha, 'beta', best_beta)
            x_hat = out

        if epoch - best_epoch > 1:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if best_alpha == alpha and best_beta == beta:
            break

    return x_hat, best_alpha, best_beta


class Validator:
    def __init__(self, scores: Scores, dataset_name:str, edge_index, x_all, trn_nodes, val_nodes, test_nodes) -> None:
        self.scores = scores
        self.dataset_name = dataset_name
        self.edge_index, self.x_all, self.trn_nodes, self.val_nodes, self.test_nodes = edge_index, x_all, trn_nodes, val_nodes, test_nodes
        self.datas = {"trn": self.trn_nodes, "val": self.val_nodes, "tst": self.test_nodes}
        self.record_datas = ['tst']
        if d.is_continuous(dataset_name):
            self.val_top_ks = [-1]
            self.metrics = ['CORR', 'RMSE']
        else:
            self.val_top_ks = [3, 5, 10] if dataset_name=="steam" else [10, 20, 50]
            self.metrics = ["nDCG", "Recall"]

    def _calc_single_score(self, x_hat, nodes, k, metric):
        return calc_single_score(self.dataset_name, x_hat, self.x_all, nodes, k, metric)

    def _calc_and_record_scores(self, x_hat, algo_name, k, metric, params_kw):
        row = f"{metric}_{algo_name}"
        for data_name in self.record_datas:
            col, score = self._calc_single_score(x_hat, self.datas[data_name], k, metric)
            col = col if data_name=='tst' else f"{col}_{data_name}"
            self.scores.add_score(row, col, score)
        if params_kw is not None:
            self.scores.add_score(row, f"{self.dataset_name}@{k}_params", str(params_kw))

    def validate(self, apa_fn, params_kw:dict={}):
        algo_name = apa_fn.__name__
        logger.info(f"validate [{algo_name}] in [{self.dataset_name}]")
        x_hat = apa_fn(**params_kw)
        for metric in self.metrics:
            for k in self.val_top_ks:
                self._calc_and_record_scores(x_hat, algo_name, k, metric, params_kw)
        self.scores.print(None)

    def _product_search_best(self, apa_fn, params_kw:dict = {'alpha':[0.0,0.5,1.0], 'beta':[0.0,0.5,1.0], 'num_iter':[1]}, k:int=10, metric:str="CORR"):
        best_params = None
        best_x_hat = None
        best_score = None
        params_product = [{p_k:p_v for p_k,p_v in zip(params_kw.keys(),p_vs)}  for p_vs in itertools.product(*params_kw.values()) ]
        for params in params_product:
            x_hat = apa_fn(**params)
            _, curr_score = self._calc_single_score(x_hat, self.val_nodes, k, metric)
            if best_score is None or greater_is_better(metric) == (curr_score>best_score):
                best_score = curr_score
                best_x_hat = x_hat
                best_params = params
        return best_params, best_x_hat

    def _search_best(self, apa_fn, max_num_iter:int, min_num_iter:int=0, params_range_kw:dict = {'alpha':(0.0,1.0), 'beta':(0.0,1.0)}, k:int=10, metric:str="CORR"):
        s = SearchPoints(**params_range_kw)

        def score_fn(point):
            params = {tag:p for tag, p in zip(s.tags, point)}
            params['num_iter'] = max_num_iter
            x_hat = apa_fn(**params)
            _, score = self._calc_single_score(x_hat, self.val_nodes, k, metric)
            return score, (x_hat, max_num_iter), f"{self.dataset_name}@{k} {metric} "

        def multi_score_fn(point):
            params = {tag:p for tag, p in zip(s.tags, point)}
            best_score = None
            best_x_hat = None
            best_num_iter = 0
            for iter_i in range(1, max_num_iter+1):
                params['num_iter'] = iter_i
                x_hat = apa_fn(**params)
                _, curr_score = self._calc_single_score(x_hat, self.val_nodes, k, metric)
                if iter_i>=min_num_iter and (best_score is None or greater_is_better(metric) == (curr_score>best_score)):
                    best_score = curr_score
                    best_x_hat = x_hat
                    best_num_iter=iter_i
            debug_msg = f"{self.dataset_name}@{k} {metric}, num_iter={best_num_iter:3d}"
            return best_score, (best_x_hat, best_num_iter), debug_msg

        def apa2_score_fn(point):
            params = {tag:p for tag, p in zip(s.tags, point)}
            best_score = None
            best_x_hat = None
            best_num_iter = 0
            Ps, x_hat, new_x_hat = None, None, None
            for iter_i in range(1, max_num_iter+1):
                Ps, x_hat, new_x_hat = apa_fn(Ps, x_hat, new_x_hat, **params)
                _, curr_score = self._calc_single_score(new_x_hat, self.val_nodes, k, metric)
                if iter_i>=min_num_iter and (best_score is None or greater_is_better(metric) == (curr_score>best_score)):
                    best_score = curr_score
                    best_x_hat = new_x_hat
                    best_num_iter=iter_i
            debug_msg = f"{self.dataset_name}@{k} {metric}, num_iter={best_num_iter:3d}"
            return best_score, (best_x_hat, best_num_iter), debug_msg

        def iter_score_fn(point):
            best_score = None
            best_x_hat = None
            best_num_iter = 0
            x_hat = None
            for iter_i in range(0, max_num_iter+1):
                x_hat = apa_fn(x_hat,**{tag:p for tag, p in zip(s.tags, point)})
                _, curr_score = self._calc_single_score(x_hat, self.val_nodes, k, metric)
                if iter_i>=min_num_iter and (best_score is None or greater_is_better(metric) == (curr_score>best_score)):
                    best_score = curr_score
                    best_x_hat = x_hat
                    best_num_iter=iter_i
            debug_msg = f"{self.dataset_name}@{k} {metric}, num_iter={best_num_iter:3d}"
            return best_score, (best_x_hat, best_num_iter), debug_msg

        s.search(apa2_score_fn, greater_is_better(metric))

        best_params = {tag:p for tag, p in zip(s.tags, s.best_points[0])}
        best_params["num_iter"] = s.best_out[0][1]
        return  best_params, s.best_out[0][0]

    @print_time_cost
    def validate_best(self, apa_fn, max_num_iter:int, params_range_kw:dict = {'alpha':(0.0,1.0), 'beta':(0.0,1.0)}):
        algo_name = apa_fn.__name__
        logger.info(f"validate_best [{algo_name}] in [{self.dataset_name}] with params_range_kw={params_range_kw}")
        for metric in self.metrics:
            for k in self.val_top_ks:
                best_params, best_x_hat = self._search_best(apa_fn, max_num_iter, params_range_kw=params_range_kw, k=k, metric=metric)
                self._calc_and_record_scores(best_x_hat, algo_name, k, metric, best_params)
                self.scores.print(None)


def main():
    run_baseline = False
    run_pr = False
    run_ppr = False
    run_mtp = False
    run_umtp = True
    dataset_names = ['pubmed']  # [ 'cora', 'citeseer', 'computers', 'photo', 'steam', 'pubmed', 'cs', 'arxiv']
    file_name = "pubmed_umtp30"
    scores = Scores()
    max_num_iter = 50

    for seed in range(1):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for dataset_name in dataset_names:

            logger.info(f"dataset_name={dataset_name}, seed={seed}")

            edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes = d.load_data(dataset_name, split=(0.4, 0.1, 0.5), seed=seed)
            apa = APA(x_all, edge_index, trn_nodes, d.is_binary(dataset_name))
            v = Validator(scores, dataset_name, edge_index, x_all, trn_nodes, val_nodes, test_nodes)

            apa2 = APA2(x_all, edge_index, trn_nodes, val_nodes)

            if run_baseline:
                v.validate(apa.fp)

            if run_pr:
                v.validate_best(apa.pr, max_num_iter, {"alpha":(0.0, 1.0)})

            if run_ppr:
                v.validate_best(apa.ppr, max_num_iter, {"alpha":(0.0, 1.0)})

            if run_mtp:
                v.validate_best(apa.mtp, max_num_iter, {"alpha":(0.0, 1.0)})

            if run_umtp:
                # v.validate_best(apa.umtp, max_num_iter, {"alpha":(0.0, 1.0), "beta":(0.0, 1.0)})
                v.validate_best(apa2.run, max_num_iter, {"alpha":(0.0, 1.0), "beta":(0.0, 1.0)})

    scores.print(file_name)
    print("end")
