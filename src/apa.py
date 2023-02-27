import data as d
from utils import SearchPoints, Scores, print_time_cost
import itertools
import logging
import numpy as np
import torch
from sklearn.metrics import ndcg_score
from torch_scatter import scatter_add
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian, remove_self_loops


logger = logging.getLogger('g.apa')


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


def greater_is_better(metric):
        return metric!="RMSE"


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


def get_propagation_matrix(edge_index:Adj, n_nodes:int, mode:str="adj") -> torch.sparse.FloatTensor:
    edge_index, edge_weight = get_laplacian(edge_index, num_nodes=n_nodes, normalization="sym")
    if mode == "adj":
        edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
    # edge_index, edge_weight = get_normalized_adjacency(edge_index, n_nodes)
    adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj


class APA(torch.nn.Module):

    def __init__(self, x: torch.Tensor, edge_index: Adj, known_mask: torch.Tensor):
        super().__init__()
        self.n_nodes = x.size(0)
        self.adj = get_propagation_matrix(edge_index, self.n_nodes)
        self.known_mask = known_mask
        self.out_k_init = x[known_mask]
        self.out = torch.zeros_like(x)
        self.out[known_mask] = self.out_k_init
        self.known_mean = self.out_k_init.mean(dim=0)

    def fp(self, out:torch.Tensor=None, num_iter: int=1, **kw) -> torch.Tensor:
        if out is None:
            out = torch.nn.functional.normalize(self.out, p=2, dim=1)
        out_k_init = torch.nn.functional.normalize(self.out_k_init, p=2, dim=1)
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.known_mask] = out_k_init
        return out

    def fp_analytical_solution(self) -> torch.Tensor:
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

        out = self.x.clone()
        out[unknow_mask] = torch.mm(torch.mm(L_inv, A_uk), out[known_mask])

        return out
    
    def pr(self, out:torch.Tensor=None, alpha: float=0.85, num_iter: int=1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1-alpha)*out.mean(dim=0)  
            out[self.known_mask] = self.out_k_init
        return out

    def ppr(self, out:torch.Tensor=None, alpha: float=0.85, weight: torch.Tensor = None, num_iter: int=1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        if weight is None:
            weight = self.known_mean
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1-alpha)*weight  
            out[self.known_mask] = self.out_k_init
        return out

    def mtp(self, out:torch.Tensor=None, alpha: float=0.85, num_iter: int=1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.known_mask] = alpha*out[self.known_mask] + (1-alpha)*self.out_k_init
        return out

    def umtp(self, out:torch.Tensor=None, alpha: float=0.85, beta: float=0.70, num_iter: int=1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.known_mask] = beta*out[self.known_mask] + (1-beta)*self.out_k_init
        return out

    def umtp_analytical_solution(self, alpha: float=0.85, beta: float=0.70, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        theta = (n_nodes-1)*(1-alpha)/alpha/n_nodes
        lamda = 1/(1-beta)-1/theta
        edge_index, edge_weight = get_laplacian(edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.known_mask] = 1
        Ik = torch.diag(Ik_diag)
        L1 = torch.eye(n_nodes)*(n_nodes/(n_nodes-1)) - torch.ones(n_nodes,n_nodes)/(n_nodes-1)
        out = torch.mm(torch.inverse(L+lamda*Ik+theta*L1), lamda*torch.mm(Ik,self.x))
        return out

    
def sgd(apa_fn, x, edge_index, known_feature_mask, val_nodes_mask, epoches=50, init_alpha=0.85, init_beta=0.7):
    alpha, beta = torch.tensor([init_alpha], requires_grad=True), torch.tensor([init_beta], requires_grad=True)
    loss_fn = torch.nn.MSELoss()
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

    def _calc_single_score(self, x_hat, nodes, k, metric):
        if d.is_continuous(self.dataset_name):
            if metric=="CORR":
                return self.dataset_name, to_r2(x_hat[nodes], self.x_all[nodes])
            elif metric=="RMSE":
                return self.dataset_name, to_rmse(x_hat[nodes], self.x_all[nodes])
            else:
                raise Exception(f"no this metric {metric}")
        else:
            if metric=="nDCG":
                return f"{self.dataset_name}@{k}", to_ndcg(x_hat[nodes], self.x_all[nodes], k=k)
            elif metric=="Recall":
                return f"{self.dataset_name}@{k}", to_recall(x_hat[nodes], self.x_all[nodes], k=k)
            else:
                raise Exception(f"no this metric {metric}")

    def _calc_and_record_scores(self, x_hat, algo_name, k, metric, datas, params_kw):
        row = f"{metric}_{algo_name}"
        for data_name in datas:
            col, score = self._calc_single_score(x_hat, self.datas[data_name], k, metric)
            col = col if data_name=='tst' else f"{col}_{data_name}"
            self.scores.add_score(row, col, score)
        if params_kw is not None:
            self.scores.add_score(row, f"{self.dataset_name}@{k}_params", str(params_kw))

    def validate(self, apa_fn, params_kw=None, ks=[10,20,50], datas=['tst']):
        algo_name = apa_fn.__name__
        logger.info(f"validate [{algo_name}] in [{self.dataset_name}]")
        x_hat = apa_fn(**params_kw)
        if d.is_continuous(self.dataset_name):
            ks = [10]
            metrics = ['CORR', 'RMSE']
        else:
            metrics = ["nDCG", "Recall"]
        for metric in metrics:
            for k in ks:
                self._calc_and_record_scores(x_hat, algo_name, k, metric, datas, params_kw)
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
    
    def _search_best(self, apa_fn, max_num_iter:int, params_range_kw:dict = {'alpha':(0.0,1.0), 'beta':(0.0,1.0)}, k:int=10, metric:str="CORR"):
        s = SearchPoints(**params_range_kw)
        s.best_num_iter = 0
        def score_fn(point):
            logger.debug(f"{dataset_name}@{k}, metric={metric}, point={point}")
            best_score = None
            best_x_hat = None
            x_hat = None
            for iter_i in range(1, max_num_iter+1):
                x_hat = apa_fn(x_hat,**{tag:p for tag, p in zip(s.tags, point)})
                _, curr_score = self._calc_single_score(x_hat, self.val_nodes, k, metric)
                if best_score is None or greater_is_better(metric) == (curr_score>best_score):
                    best_score = curr_score
                    best_x_hat = x_hat
                    s.best_num_iter=iter_i
            return best_score, best_x_hat
        s.search(score_fn, greater_is_better(metric))

        best_params = {tag:p for tag, p in zip(s.tags, s.best_points[0])}
        best_params["num_iter"] = s.best_num_iter
        return  best_params, s.best_out[0]

    @print_time_cost
    def validate_best(self, apa_fn, max_num_iter:int, params_range_kw:dict = {'alpha':(0.0,1.0), 'beta':(0.0,1.0)}, ks=[10,20,50], datas=['tst']):
        algo_name = apa_fn.__name__
        logger.info(f"validate_best [{algo_name}] in [{self.dataset_name}] with params_range_kw={params_range_kw}")
        if d.is_continuous(self.dataset_name):
            ks = [10]
            metrics = ['CORR', 'RMSE']
        else:
            metrics = ["nDCG", "Recall"]
        for metric in metrics:
            for k in ks:
                best_params, best_x_hat = self._search_best(apa_fn, max_num_iter, params_range_kw, k, metric)
                self._calc_and_record_scores(best_x_hat, algo_name, k, metric, datas, best_params)
                self.scores.print(None)


if __name__=="__main__":
    run_baseline = False
    run_pr = False
    run_ppr = False
    run_mtp = False
    run_umtp = True
    dataset_names = [ 'cora', 'citeseer', 'computers', 'photo', 'steam', 'pubmed', 'cs', 'arxiv']  # 'steam' why only related to alpha?
    datas=['tst']
    file_name = "umtp30"
    scores = Scores()
    max_num_iter = 30

    for dataset_name in dataset_names:
        ks= [3, 5, 10] if dataset_name=="steam" else [10, 20, 50]
        for seed in range(10):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            logger.info(f"dataset_name={dataset_name}, seed={seed}")
            
            edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes = d.load_data(dataset_name, split=(0.4, 0.1, 0.5), seed=seed)
            apa = APA(x_all, edge_index, trn_nodes)
            v = Validator(scores, dataset_name, edge_index, x_all, trn_nodes, val_nodes, test_nodes)
            
            if run_baseline:
                v.validate(apa.fp, ks=ks, datas=datas)
            
            if run_pr:
                v.validate_best(apa.pr, max_num_iter, {"alpha":(0.0, 1.0)}, ks=ks, datas=datas)
            
            if run_ppr:
                v.validate_best(apa.ppr, max_num_iter, {"alpha":(0.0, 1.0)}, ks=ks, datas=datas)

            if run_mtp:
                v.validate_best(apa.mtp, max_num_iter, {"alpha":(0.0, 1.0)}, ks=ks, datas=datas)
            
            if run_umtp:
                v.validate_best(apa.umtp, max_num_iter, {"alpha":(0.0, 1.0), "beta":(0.0, 1.0)}, ks=ks, datas=datas)
            
    scores.print(file_name)
    print("end")
