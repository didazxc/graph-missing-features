from .data import data as d
from .metrics import calc_single_score, greater_is_better, to_acc
from .utils import SearchPoints, Scores, print_time_cost
from .models.apa import APA
from .models.gnn import GNN
from .models.mlp import MLP
import itertools
import logging
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from multiprocessing import Pool, Queue, Process, Manager


logger = logging.getLogger('amgraph.apa')


class EstimationValidator:
    def __init__(self, dataset_name:str, x_all, trn_nodes, val_nodes, test_nodes, max_num_iter:int, seed_idx, min_num_iter: int = 0) -> None:
        self.dataset_name = dataset_name
        self.x_all = x_all
        self.seed_idx = seed_idx
        self.datas = {"trn": trn_nodes, "val": val_nodes, "tst": test_nodes}
        self.record_datas = ['tst']
        self.max_num_iter = max_num_iter
        self.min_num_iter = min_num_iter

    @staticmethod
    def is_executed(scores: Scores, seed_idx, dataset_name, algo_name, metric, k):
        row = f"{metric}_{algo_name}"
        col = f"{dataset_name}@{k}_params"
        return scores.has_value(row, col, idx=seed_idx)

    def _search_best(self, apa_fn, params_range_kw:dict = {'alpha':(0.0,1.0), 'beta':(0.0,1.0)}, metric:str="CORR", k:int=10):
        s = SearchPoints(**params_range_kw)

        def iter_score_fn(point):
            best_score = None
            best_x_hat = None
            best_num_iter = 0
            x_hat = None
            for iter_i in range(0, self.max_num_iter+1):
                x_hat = apa_fn(x_hat,**{tag:p for tag, p in zip(s.tags, point)})
                curr_score = calc_single_score(self.dataset_name, x_hat, self.x_all, self.datas['val'], metric, k)
                if iter_i>=self.min_num_iter and (best_score is None or greater_is_better(metric) == (curr_score>best_score)):
                    best_score = curr_score
                    best_x_hat = x_hat
                    best_num_iter=iter_i
            debug_msg = f"{apa_fn.__name__:5s} {self.dataset_name}@{k} {metric}, seed_idx={self.seed_idx}, num_iter={best_num_iter:3d}"
            return best_score, (best_x_hat, best_num_iter), debug_msg

        s.search(iter_score_fn, greater_is_better(metric))

        best_params = {tag:p for tag, p in zip(s.tags, s.best_points[0])}
        best_params["num_iter"] = s.best_out[0][1]
        return  best_params, s.best_out[0][0]

    @print_time_cost
    def search_best_scores(self, apa_fn, params_range_kw:dict, metric:str, k:int, metrics=None, ks=None) -> list:
        algo_name = apa_fn.__name__
        best_params, best_x_hat = self._search_best(apa_fn, params_range_kw=params_range_kw, metric=metric, k=k)
        scores = []
        if metrics is None:
            metrics = [metric]
        if ks is None:
            ks = [k]
        for metric in metrics:
            row = f"{metric}_{algo_name}"
            for k in ks:
                col = f"{self.dataset_name}@{k}"
                for data_name in self.record_datas:
                    score = calc_single_score(self.dataset_name, best_x_hat, self.x_all, self.datas[data_name], metric, k)
                    col = col if data_name=='tst' else f"{col}_{data_name}"
                    scores.append((row, col, score))
                scores.append((row, f"{self.dataset_name}@{k}_params", best_params))
        return scores


class ClassificationValidator:

    def __init__(self, scores: Scores, dataset_name:str, edges, x_all, y_all, test_nodes, num_classes, seed, epoches, seed_idx) -> None:
        self.scores = scores
        self.dataset_name = dataset_name
        self.edges = edges
        self.x_all = x_all
        self.y_all = y_all
        self.test_nodes = test_nodes
        self.num_nodes = x_all.size(0)
        self.num_classes = num_classes
        self.num_attrs = x_all.size(1)
        self.seed = seed
        self.seed_idx = seed_idx
        self.epoches = epoches
        if d.is_continuous(dataset_name):
            self.val_top_ks = [-1]
            self.metrics = ['CORR', 'RMSE']
        else:
            self.val_top_ks = [3, 5, 10] if dataset_name.startswith("steam") else [10, 20, 50]
            self.metrics = ["nDCG", "Recall"]

    @staticmethod
    def train(model, loss_fn, optimizer, epoches, x, edges, y, trn_nodes) -> torch.Tensor:
        model.train()
        best_loss = None
        best_y_hat = None
        for _ in range(epoches):
            y_hat = model(x, edges)
            optimizer.zero_grad()
            loss = loss_fn(y_hat[trn_nodes], y[trn_nodes])
            loss.backward()
            optimizer.step()
            if best_loss is None or loss.item() <= best_loss:
                best_loss = loss.item()
                best_y_hat = y_hat
        return best_y_hat

    def calc_acc(self, x_hat, conv="GCN"):
        x = self.x_all.copy()
        x[self.test_nodes] = x_hat[self.test_nodes]
        if conv=="GCN":
            model = GNN(x_hat.size(1), self.num_classes)
        else:
            model = MLP(x_hat.size(1), self.num_classes)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        acc_list = []
        splits=KFold(n_splits=5,shuffle=True,random_state=self.seed)
        for trn_nodes, val_nodes in splits.split(range(self.num_nodes)):
            y_hat = self.train(model, loss_fn, optimizer, self.epoches, x, self.edges, self.y_all, trn_nodes)
            acc = to_acc(y_hat[val_nodes], self.y_all[val_nodes])
            acc_list.append(acc)
        acc = np.mean(acc_list)
        return acc

    def _is_executed(self, row, col, conv):
        new_row = f"{conv}_{row}"
        new_col = col.rstrip("_params")
        return self.scores.get_cnt(new_row, new_col) > self.seed_idx
    
    def _add_score(self, row, col, conv, value):
        new_row = f"{conv}_{row}"
        new_col = col.rstrip("_params")
        self.scores.add_score(new_row, new_col, value)
        self.scores.print()

    def validate_from_scores(self, apa_fn, est_scores: Scores):
        algo_name = apa_fn.__name__
        for metric in self.metrics:
            for k in self.val_top_ks:
                row = f"{metric}_{algo_name}"
                col = f"{self.dataset_name}@{k}_params"

                if not self._is_executed(row, col, "GCN") or not self._is_executed(row, col, "MLP"):
                    params = est_scores.get_value(row, col)[self.seed_idx]
                    x_hat=apa_fn(**params)
                    for conv in ["GCN", "MLP"]:
                        if not self._is_executed(row, col, conv):
                            acc = self.calc_acc(x_hat, conv)
                            self._add_score(row, col, conv, acc)


def q2scores(q:Queue, scores:Scores):
    while True:
        row, col, value, seed = q.get(True)
        scores.add_score(row, col, value, seed)
        if col.endswith("_params"):
            scores.print()


def qval(q:Queue, v:EstimationValidator, algo, kw, metric, k, seed):
    for row,col,value in v.search_best_scores(algo, kw, metric, k):
        q.put((row, col, value, seed))


def estimate():
    dataset_names=['cora', 'citeseer']  # ['cora', 'citeseer', 'computers', 'photo', 'steam', 'steam1', 'pubmed', 'cs', 'arxiv']
    run_algos=['fp', 'pr', 'ppr', 'mtp', 'umtp', 'umtp2']
    scores = Scores(file_name="all_umtp30")
    scores.load()
    max_num_iter = 30
    
    pool = Pool(processes=2)
    q = Manager().Queue()

    read2scores = Process(target=q2scores, args=(q,scores))
    read2scores.start()

    for seed in range(10):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for dataset_name in dataset_names:
            if d.is_continuous(dataset_name):
                val_top_ks = [-1]
                metrics = ['RMSE', 'CORR']
            else:
                val_top_ks = [3, 5, 10] if dataset_name.startswith("steam") else [10, 20, 50]
                metrics = ["Recall", "nDCG"]

            if dataset_name == 'steam' and max_num_iter>5:
                max_num_iter = 5
            elif dataset_name == 'steam1':
                max_num_iter = 1

            edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes, num_classes = d.load_data(dataset_name, split=(0.4, 0.1, 0.5), seed=seed)
          
            apa = APA(x_all, edge_index, trn_nodes, d.is_binary(dataset_name))
            v = EstimationValidator(dataset_name, x_all, trn_nodes, val_nodes, test_nodes, max_num_iter, seed_idx=seed)

            algos = {
                'fp':(apa.fp, {"alpha":(0.0, 0.0)}),
                'pr':(apa.pr, {"alpha":(0.0, 1.0)}),
                'ppr':(apa.ppr, {"alpha":(0.0, 1.0)}),
                'mtp':(apa.mtp, {"alpha":(0.0, 1.0)}),
                'umtp':(apa.umtp, {"alpha":(0.0, 1.0), "beta":(0.0, 1.0)}),
                'umtp2':(apa.umtp2, {"alpha":(0.0, 1.0), "beta":(0.0, 1.0), "gamma":(0.0, 1.0)})
            }

            for algo_name in run_algos:
                algo = algos[algo_name][0]
                kw = algos[algo_name][1]
                for metric in metrics:
                    for k in val_top_ks:
                        # if not v.is_executed(scores, seed_idx=seed, dataset_name=dataset_name, algo_name=algo.__name__, metric=metric, k=k):
                        #     for row, col, value in v.search_best_scores(algo, kw, metric, k):
                        #         scores.add_score(row, col, value, idx=seed)
                        #         scores.print()
                        if not v.is_executed(scores, seed_idx=seed, dataset_name=dataset_name, algo_name=algo.__name__, metric=metric, k=k):
                            pool.apply_async(qval, args=(q, v, algo, kw, metric, k, seed), error_callback=lambda x:print(x))
    
    pool.close()
    pool.join()
    scores.print()
    read2scores.terminate()
    print("end")


def classify(run_baseline = True, run_pr = True, run_ppr = True, run_mtp = True, run_umtp = True, run_umtp2 = True):

    dataset_names = ['cora']  # [ 'cora', 'citeseer', 'computers', 'photo', 'steam', 'steam1', 'pubmed', 'cs', 'arxiv']
    est_scores = Scores(file_name="all_umtp30")
    est_scores.load()
    scores = Scores(file_name="class_all_umtp30")
    scores.load()
    max_num_iter = 30
    epoches = 100

    for seed in range(10):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for dataset_name in dataset_names:
            if dataset_name == 'steam' and max_num_iter>5:
                max_num_iter = 5
            elif dataset_name == 'steam1':
                max_num_iter = 1

            logger.info(f"dataset_name={dataset_name}, seed={seed}")

            edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes, num_classes = d.load_data(dataset_name, split=(0.4, 0.1, 0.5), seed=seed)
          
            apa = APA(x_all, edge_index, trn_nodes, d.is_binary(dataset_name))
            c = ClassificationValidator(scores, dataset_name, x_all, y_all, test_nodes, num_classes, seed=seed, epoches=epoches, seed_idx=seed)

            if run_baseline:
                c.validate_from_scores(apa.fp, est_scores)

            if run_pr:
                c.validate_from_scores(apa.pr, est_scores)

            if run_ppr:
                c.validate_from_scores(apa.ppr, est_scores)

            if run_mtp:
                c.validate_from_scores(apa.mtp, est_scores)

            if run_umtp:
                c.validate_from_scores(apa.umtp, est_scores)
            
            if run_umtp2:
                c.validate_from_scores(apa.umtp2, est_scores)

    scores.print()
    print("end")


def main():
    estimate()
