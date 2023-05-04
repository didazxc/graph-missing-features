from ..data import data as d
from ..metrics import calc_single_score, greater_is_better
from ..utils import SearchPoints, Scores
from ..models.apa import APA
import logging
import numpy as np
import torch
import multiprocessing

logger = logging.getLogger('amgraph.validators.estimation')


class EstDataset:
    def __init__(self, data_name: str, split, seed: int,
                 max_num_iter: int, min_num_iter: int, k_index: int,
                 early_stop: bool = True):
        self.data = d.load_data(data_name, split, seed)
        self.seed_idx = seed
        self.max_num_iter = max_num_iter
        self.min_num_iter = min_num_iter
        self.all_metrics, self.all_ks = self.metric_and_ks(data_name)
        self.metric, self.k = self.all_metrics[k_index], self.all_ks[k_index]
        self.early_stop = early_stop
        self.data_mask_names = {"trn": self.data.trn_mask, "val": self.data.val_mask, "tst": self.data.test_mask}

    @staticmethod
    def metric_and_ks(dataset_name):
        if d.is_continuous(dataset_name):
            all_ks = [-1]
            all_metrics = ['RMSE', 'CORR']
        else:
            all_ks = [3, 5, 10] if dataset_name.startswith("steam") else [10, 20, 50]
            all_metrics = ["Recall", "nDCG"]
        return all_metrics, all_ks

    def score(self, x_hat, mask_name='val'):
        return calc_single_score(self.data.data_name, x_hat, self.data.x, self.data_mask_names[mask_name], self.metric, self.k)


class EstimationValidator:
    def __init__(self, dataset: EstDataset) -> None:
        self.dataset = dataset
        self.dataset_name = dataset.data.data_name
        self.seed_idx = dataset.seed_idx
        self.max_num_iter = dataset.max_num_iter
        if dataset.early_stop:
            if self.dataset_name == 'steam' and dataset.max_num_iter > 5:
                self.max_num_iter = 5
            elif self.dataset_name == 'steam1':
                self.max_num_iter = 1
        self.min_num_iter = dataset.min_num_iter
        self.metric, self.k = dataset.metric, dataset.k
        self.early_stop = dataset.early_stop

    @staticmethod
    def is_executed(scores: Scores, seed_idx, dataset_name, algo_name, metric, k):
        row = f"{metric}_{algo_name}"
        col = f"{dataset_name}@{k}_params"
        return scores.has_value(row, col, idx=seed_idx)

    def _search_best(self, apa_fn, params_range_kw: dict = {'alpha': (0.0, 1.0), 'beta': (0.0, 1.0)},
                     metric: str = "CORR", k: int = 10):
        s = SearchPoints(**params_range_kw)

        def score_fn(point):
            x_hat = apa_fn(num_iter=self.max_num_iter, **{tag: p for tag, p in zip(s.tags, point)})
            score = self.dataset.score(x_hat)
            debug_msg = f"{apa_fn.__name__:5s} {self.dataset_name}@{k} {metric}, seed_idx={self.seed_idx}"
            return score, (x_hat, self.max_num_iter), debug_msg

        def iter_score_fn(point):
            best_score = None
            best_x_hat = None
            best_num_iter = 0
            x_hat = None
            for iter_i in range(1, self.max_num_iter + 1):
                x_hat = apa_fn(x_hat, **{tag: p for tag, p in zip(s.tags, point)})
                curr_score = self.dataset.score(x_hat)
                if iter_i >= self.min_num_iter and (
                        best_score is None or greater_is_better(metric) == (curr_score > best_score)):
                    best_score = curr_score
                    best_x_hat = x_hat
                    best_num_iter = iter_i
            debug_msg = f"{apa_fn.__name__:5s} {self.dataset_name}@{k} {metric}, seed_idx={self.seed_idx}, num_iter={best_num_iter:3d}"
            return best_score, (best_x_hat, best_num_iter), debug_msg

        fn = iter_score_fn if self.early_stop else score_fn
        s.search(fn, greater_is_better(metric))

        best_params = {tag: p for tag, p in zip(s.tags, s.best_points[0])}
        best_params["num_iter"] = s.best_out[0][1]
        return best_params, s.best_out[0][0]

    def search_best_scores_once(self, apa_fn, params_range_kw: dict) -> list:
        scores = []
        algo_name = apa_fn.__name__
        best_params, best_x_hat = self._search_best(apa_fn, params_range_kw=params_range_kw, metric=self.metric,
                                                    k=self.k)
        for metric in self.dataset.all_metrics:
            row = f"{metric}_{algo_name}"
            for k in self.dataset.all_ks:
                col = f"{self.dataset_name}@{k}"
                for data_mask_name in self.dataset.data_mask_names:
                    score = self.dataset.score(best_x_hat, data_mask_name)
                    col = col if data_mask_name == 'tst' else f"{col}_{data_mask_name}"
                    scores.append((row, col, score))
                scores.append((row, f"{self.dataset_name}@{k}_params", best_params))
        return scores

    def search_best_scores(self, apa_fn, params_range_kw: dict, metric, k) -> list:
        scores = []
        algo_name = apa_fn.__name__
        best_params, best_x_hat = self._search_best(apa_fn, params_range_kw=params_range_kw, metric=metric, k=k)
        row = f"{metric}_{algo_name}"
        col = f"{self.dataset_name}@{k}"
        for data_mask_name in self.dataset.data_mask_names:
            score = self.dataset.score(best_x_hat, data_mask_name)
            col = col if data_mask_name == 'tst' else f"{col}_{data_mask_name}"
            scores.append((row, col, score))
        scores.append((row, f"{self.dataset_name}@{k}_params", best_params))
        return scores

    @staticmethod
    def run(file_name="iter_le_30",
            dataset_names=['cora', 'citeseer', 'computers', 'photo', 'steam', 'pubmed', 'cs', 'arxiv'],
            run_algos=['fp', 'pr', 'ppr', 'mtp', 'umtp', 'umtp2'], max_num_iter=30, only_val_once=True,
            early_stop=True, k_index=0):
        scores = Scores(file_name=file_name)
        scores.load()
        for seed in range(10):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            for dataset_name in dataset_names:
                data = EstDataset(dataset_name, split=(0.4, 0.1, 0.5), seed=seed, max_num_iter=max_num_iter, min_num_iter=1, k_index=k_index, early_stop=early_stop)
                apa = APA(data.data.edges, data.data.x, data.data.trn_mask, data.data.is_binary)
                v = EstimationValidator(data)
                algos = {
                    'fp': (apa.fp, {"alpha": (0.0, 0.0)}),
                    'pr': (apa.pr, {"alpha": (0.0, 1.0)}),
                    'ppr': (apa.ppr, {"alpha": (0.0, 1.0)}),
                    'mtp': (apa.mtp, {"alpha": (0.0, 1.0)}),
                    'umtp': (apa.umtp, {"alpha": (0.0, 1.0), "beta": (0.0, 1.0)}),
                    'umtp2': (apa.umtp2, {"alpha": (0.0, 1.0), "beta": (0.0, 1.0), "gamma": (0.0, 1.0)})
                }
                for algo_name in run_algos:
                    if only_val_once:
                        metric = v.metric
                        k = v.k
                        if not EstimationValidator.is_executed(scores, seed_idx=seed, dataset_name=dataset_name,
                                                               algo_name=algo_name, metric=metric, k=k):
                            for row, col, value in v.search_best_scores_once(algos[algo_name][0], algos[algo_name][1]):
                                scores.add_score(row, col, value, seed)
                            scores.print()
                    else:
                        for metric in data.all_metrics:
                            for k in data.all_ks:
                                if not EstimationValidator.is_executed(scores, seed_idx=seed, dataset_name=dataset_name,
                                                                       algo_name=algo_name, metric=metric, k=k):
                                    for row, col, value in v.search_best_scores(algos[algo_name][0],
                                                                                algos[algo_name][1], metric, k):
                                        scores.add_score(row, col, value, seed)
                                    scores.print()

    @staticmethod
    def multi_run(file_name="combine_iter_le_30",
                  dataset_names=['cora', 'citeseer', 'computers', 'photo', 'steam', 'pubmed', 'cs', 'arxiv'],
                  run_algos=['fp', 'pr', 'ppr', 'mtp', 'umtp', 'umtp2'], max_num_iter=30, only_val_once=True,
                  early_stop=True, k_index=0):
        file_names = []
        task_list = []
        mp = multiprocessing.get_context('spawn')
        for d in dataset_names:
            file_name_d = f"multiprocess_{file_name}_{d}"
            file_names.append(file_name_d)
            p = mp.Process(target=EstimationValidator.run,
                           args=(file_name_d, [d], run_algos, max_num_iter, only_val_once, early_stop, k_index))
            task_list.append(p)
            p.start()
        for p in task_list:
            p.join()
        s = Scores(file_name)
        s.combine(file_names).print()


