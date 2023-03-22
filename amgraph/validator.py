from .data import data as d
from .metrics import calc_single_score, greater_is_better, to_acc
from .utils import SearchPoints, Scores
from .models.apa import APA
from .models.gnn import GNN
from .models.mlp import MLP
import logging
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
import multiprocessing
from torch_geometric.utils import subgraph


logger = logging.getLogger('amgraph.apa')


class EstimationValidator:
    def __init__(self, dataset_name:str, x_all, trn_nodes, val_nodes, test_nodes, max_num_iter:int, seed_idx, min_num_iter: int = 0, early_stop: bool = True, k_index=0) -> None:
        self.dataset_name = dataset_name
        self.x_all = x_all
        self.seed_idx = seed_idx
        self.datas = {"trn": trn_nodes, "val": val_nodes, "tst": test_nodes}
        self.record_datas = ['tst']
        if early_stop:
            if dataset_name == 'steam' and max_num_iter>5:
                max_num_iter = 5
            elif dataset_name == 'steam1':
                max_num_iter = 1
        self.max_num_iter = max_num_iter
        self.min_num_iter = min_num_iter
        self.all_metrics, self.all_ks = self.metric_and_ks(dataset_name)
        self.metric, self.k = self.all_metrics[k_index], self.all_ks[k_index]
        self.early_stop = early_stop

    @staticmethod
    def metric_and_ks(dataset_name):
        if d.is_continuous(dataset_name):
            all_ks = [-1]
            all_metrics = ['RMSE', 'CORR']
        else:
            all_ks = [3, 5, 10] if dataset_name.startswith("steam") else [10, 20, 50]
            all_metrics = ["Recall", "nDCG"]
        return all_metrics, all_ks

    @staticmethod
    def is_executed(scores: Scores, seed_idx, dataset_name, algo_name, metric, k):
        row = f"{metric}_{algo_name}"
        col = f"{dataset_name}@{k}_params"
        return scores.has_value(row, col, idx=seed_idx)

    def _search_best(self, apa_fn, params_range_kw:dict = {'alpha':(0.0,1.0), 'beta':(0.0,1.0)}, metric:str="CORR", k:int=10):
        s = SearchPoints(**params_range_kw)

        def score_fn(point):
            x_hat = apa_fn(num_iter=self.max_num_iter, **{tag:p for tag, p in zip(s.tags, point)})
            score = calc_single_score(self.dataset_name, x_hat, self.x_all, self.datas['val'], metric, k)
            debug_msg = f"{apa_fn.__name__:5s} {self.dataset_name}@{k} {metric}, seed_idx={self.seed_idx}"
            return score, (x_hat, self.max_num_iter), debug_msg

        def iter_score_fn(point):
            best_score = None
            best_x_hat = None
            best_num_iter = 0
            x_hat = None
            for iter_i in range(1, self.max_num_iter+1):
                x_hat = apa_fn(x_hat,**{tag:p for tag, p in zip(s.tags, point)})
                curr_score = calc_single_score(self.dataset_name, x_hat, self.x_all, self.datas['val'], metric, k)
                if iter_i>=self.min_num_iter and (best_score is None or greater_is_better(metric) == (curr_score>best_score)):
                    best_score = curr_score
                    best_x_hat = x_hat
                    best_num_iter=iter_i
            debug_msg = f"{apa_fn.__name__:5s} {self.dataset_name}@{k} {metric}, seed_idx={self.seed_idx}, num_iter={best_num_iter:3d}"
            return best_score, (best_x_hat, best_num_iter), debug_msg

        fn = iter_score_fn if self.early_stop else score_fn
        s.search(fn, greater_is_better(metric))

        best_params = {tag:p for tag, p in zip(s.tags, s.best_points[0])}
        best_params["num_iter"] = s.best_out[0][1]
        return  best_params, s.best_out[0][0]

    def search_best_scores_once(self, apa_fn, params_range_kw:dict) -> list:
        scores = []
        algo_name = apa_fn.__name__
        best_params, best_x_hat = self._search_best(apa_fn, params_range_kw=params_range_kw, metric=self.metric, k=self.k)
        for metric in self.all_metrics:
            row = f"{metric}_{algo_name}"
            for k in self.all_ks:
                col = f"{self.dataset_name}@{k}"
                for data_name in self.record_datas:
                    score = calc_single_score(self.dataset_name, best_x_hat, self.x_all, self.datas[data_name], metric, k)
                    col = col if data_name=='tst' else f"{col}_{data_name}"
                    scores.append((row, col, score))
                scores.append((row, f"{self.dataset_name}@{k}_params", best_params))
        return scores

    def search_best_scores(self, apa_fn, params_range_kw:dict, metric, k) -> list:
        scores = []
        algo_name = apa_fn.__name__
        best_params, best_x_hat = self._search_best(apa_fn, params_range_kw=params_range_kw, metric=metric, k=k)
        row = f"{metric}_{algo_name}"
        col = f"{self.dataset_name}@{k}"
        for data_name in self.record_datas:
            score = calc_single_score(self.dataset_name, best_x_hat, self.x_all, self.datas[data_name], metric, k)
            col = col if data_name=='tst' else f"{col}_{data_name}"
            scores.append((row, col, score))
        scores.append((row, f"{self.dataset_name}@{k}_params", best_params))
        return scores

    @staticmethod
    def run(file_name="iter_le_30", dataset_names = ['cora', 'citeseer', 'computers', 'photo', 'steam', 'pubmed', 'cs', 'arxiv'], run_algos=['fp', 'pr', 'ppr', 'mtp', 'umtp', 'umtp2'], max_num_iter = 30, only_val_once=True, early_stop=True, k_index=0):
        scores = Scores(file_name=file_name)
        scores.load()
        for seed in range(10):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            for dataset_name in dataset_names:
                edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes, num_classes = d.load_data(dataset_name, split=(0.4, 0.1, 0.5), seed=seed)
                apa = APA(x_all, edge_index, trn_nodes, d.is_binary(dataset_name))
                v = EstimationValidator(dataset_name, x_all, trn_nodes, val_nodes, test_nodes, max_num_iter, seed_idx=seed, early_stop=early_stop, k_index=k_index)
                algos = {
                    'fp':(apa.fp, {"alpha":(0.0, 0.0)}),
                    'pr':(apa.pr, {"alpha":(0.0, 1.0)}),
                    'ppr':(apa.ppr, {"alpha":(0.0, 1.0)}),
                    'mtp':(apa.mtp, {"alpha":(0.0, 1.0)}),
                    'umtp':(apa.umtp, {"alpha":(0.0, 1.0), "beta":(0.0, 1.0)}),
                    'umtp2':(apa.umtp2, {"alpha":(0.0, 1.0), "beta":(0.0, 1.0), "gamma":(0.0, 1.0)})
                }
                for algo_name in run_algos:
                    if only_val_once:
                        metric = v.metric
                        k = v.k
                        if not EstimationValidator.is_executed(scores, seed_idx=seed, dataset_name=dataset_name, algo_name=algo_name, metric=metric, k=k):
                            for row, col, value in v.search_best_scores_once(algos[algo_name][0], algos[algo_name][1]):
                                scores.add_score(row, col, value, seed)
                            scores.print()
                    else:
                        for metric in v.all_metrics:
                            for k in v.all_ks:
                                if not EstimationValidator.is_executed(scores, seed_idx=seed, dataset_name=dataset_name, algo_name=algo_name, metric=metric, k=k):
                                    for row, col, value in v.search_best_scores(algos[algo_name][0], algos[algo_name][1], metric, k):
                                        scores.add_score(row, col, value, seed)
                                    scores.print()

    @staticmethod
    def multi_run(file_name="combine_iter_le_30", dataset_names = ['cora', 'citeseer', 'computers', 'photo', 'steam', 'pubmed', 'cs', 'arxiv'], run_algos=['fp', 'pr', 'ppr', 'mtp', 'umtp', 'umtp2'], max_num_iter = 30, only_val_once=True, early_stop=True, k_index=0):
        file_names = []
        task_list = []
        mp = multiprocessing.get_context('spawn')
        for d in dataset_names:
            file_name_d = f"multiprocess_{file_name}_{d}"
            file_names.append(file_name_d)
            p=mp.Process(target=EstimationValidator.run, args=(file_name_d, [d], run_algos, max_num_iter, only_val_once, early_stop, k_index))
            task_list.append(p)
            p.start()
        for p in task_list:
            p.join()
        s = Scores(file_name)
        s.combine(file_names).print()


class ClassificationValidator:

    def __init__(self, scores: Scores, dataset_name:str, edges, y_all, test_nodes, num_attrs, num_classes, epoches, seed, seed_idx, k_index=0) -> None:
        self.scores = scores
        self.dataset_name = dataset_name

        self.edges = edges
        self.y_all = y_all
        self.test_nodes = test_nodes

        self.num_classes = num_classes
        self.num_attrs = num_attrs

        self.seed = seed
        self.seed_idx = seed_idx
        self.epoches = epoches
        if d.is_continuous(dataset_name):
            self.ks = [-1]
            self.metrics = ['RMSE', 'CORR']
        else:
            self.ks = [3, 5, 10] if dataset_name.startswith("steam") else [10, 20, 50]
            self.metrics = ["Recall", "nDCG"]
        self.metric, self.k = self.metrics[k_index], self.ks[k_index]

    @staticmethod
    def train(model, loss_fn, optimizer, epoches, x, edges, y, trn_nodes, val_nodes) -> torch.Tensor:
        model.train()
        best_loss = None
        train_acc = 0
        acc = 0
        for epoch in range(epoches):
            y_hat = model(x, edges)
            optimizer.zero_grad()
            loss = loss_fn(y_hat[trn_nodes], y[trn_nodes])
            loss.backward()
            optimizer.step()
            if best_loss is None or loss.item() <= best_loss:
                best_loss = loss.item()
                train_acc = to_acc(torch.argmax(y_hat[trn_nodes], dim=1), y[trn_nodes])
                acc = to_acc(torch.argmax(y_hat[val_nodes], dim=1), y[val_nodes])
            if epoch % 100 == 0:
                logger.debug(f"{epoch:4d} best_loss:{best_loss:7.5f} train_acc:{train_acc:7.5f} acc:{acc:7.5f}")
        return acc

    def calc_acc(self, x_hat, conv="GCN"):
        acc_list = []
        splits=KFold(n_splits=5,shuffle=True,random_state=self.seed)
        logger.info(f"start model {conv}")
        for trn_nodes, val_nodes in splits.split(self.test_nodes):
            if conv=="GCN":
                model = GNN(self.num_attrs, self.num_classes, num_layers=2, hidden_size=256)
            else:
                model = MLP(self.num_attrs, self.num_classes, num_layers=2, hidden_size=256)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            acc = self.train(model, loss_fn, optimizer, self.epoches, x_hat, self.edges, self.y_all, trn_nodes, val_nodes)
            logger.info(f"model {conv} acc {acc}")
            acc_list.append(acc)
        acc = np.mean(acc_list)
        return acc
    
    def _is_executed(self, row, col, conv):
        new_row = f"{conv}_{row}"
        new_col = col.rstrip("_params")
        return self.scores.has_value(new_row, new_col, idx=self.seed_idx)
    
    def _add_score(self, row, col, conv, value):
        new_row = f"{conv}_{row}"
        new_col = col.rstrip("_params")
        self.scores.add_score(new_row, new_col, value)
        self.scores.print()

    def validate_from_scores(self, apa_fn, est_scores: Scores, val_only_once):
        algo_name = apa_fn.__name__
        convs = ["GCN"]  # ["GCN", "MLP"]
        metrics, ks = ([self.metric], [self.k]) if val_only_once else (self.metrics, self.ks)
        for metric in metrics:
            for k in ks:
                row = f"{metric}_{algo_name}"
                col = f"{self.dataset_name}@{k}_params"

                if any([not self._is_executed(row, col, conv) for conv in convs]):
                    params = est_scores.get_value(row, col)[self.seed_idx]
                    logger.info(f"{algo_name}, {row}, {col}, {params}")
                    x_hat=apa_fn(**params)

                    for conv in convs:
                        if not self._is_executed(row, col, conv):
                            acc = self.calc_acc(x_hat, conv)
                            self._add_score(row, col, conv, acc)
        self.scores.print()

    @staticmethod
    def run(file_name="class_all_umtp30", est_scores_file_name="all_umtp30", dataset_names = ['cora', 'citeseer', 'computers', 'photo', 'steam', 'pubmed', 'cs', 'arxiv'], run_algos=['fp', 'pr', 'ppr', 'mtp', 'umtp', 'umtp2'], val_only_once=True, k_index=0):
        est_scores = Scores(file_name=est_scores_file_name)
        est_scores.load()
        scores = Scores(file_name=file_name)
        scores.load()
        epoches = 2000

        for seed in range(10):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            for dataset_name in dataset_names:
                edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes, num_classes = d.load_data(dataset_name, split=(0.4, 0.1, 0.5), seed=seed)
                apa = APA(x_all, edge_index, trn_nodes, d.is_binary(dataset_name))
                induced_edge_index, _ = subgraph(test_nodes, edge_index)
                c = ClassificationValidator(scores, dataset_name=dataset_name, edges=induced_edge_index, y_all=y_all, test_nodes=test_nodes, num_attrs=x_all.size(1), num_classes=num_classes, epoches=epoches, seed=seed, seed_idx=seed, k_index=k_index)
                algos = {
                        'fp':apa.fp,
                        'pr':apa.pr,
                        'ppr':apa.ppr,
                        'mtp':apa.mtp,
                        'umtp':apa.umtp,
                        'umtp2':apa.umtp2
                    }
                for algo_name in run_algos:
                    c.validate_from_scores(algos[algo_name], est_scores, val_only_once)
                
        scores.print()
        print("end")


def main():
    # EstimationValidator.multi_run("combine_all_le30", max_num_iter=30, early_stop=True, only_val_once= False)
    # EstimationValidator.multi_run("combine_all_eq30", max_num_iter=30, early_stop=False, only_val_once= False)
    # EstimationValidator.multi_run("combine_k10_le30", max_num_iter=30, early_stop=True)
    # EstimationValidator.multi_run("combine_k10_eq30", max_num_iter=30, early_stop=False)
    # EstimationValidator.multi_run("combine_k50_le30", max_num_iter=30, early_stop=True, k_index= -1)
    # EstimationValidator.multi_run("combine_k50_eq30", max_num_iter=30, early_stop=False, k_index= -1)
    # ClassificationValidator.run(file_name="class_k10_eq30_test", est_scores_file_name="combine_k10_eq30")
    ClassificationValidator.run(file_name="class_all_le30", est_scores_file_name="combine_all_le30", val_only_once=False)

