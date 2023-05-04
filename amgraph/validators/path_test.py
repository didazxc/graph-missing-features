from ..data import data as d
from ..metrics import calc_single_score, greater_is_better
from ..utils import SearchPoints
from ..models.apa import APA
from .estimation import EstDataset
import pandas as pd
import numpy as np
import torch


class PathTest:

    @staticmethod
    def score_paths(points, num_iter, apa_fn, score_fn) -> pd.DataFrame:
        tags = ['alpha', 'beta', 'gamma', 'iter_num']
        res: dict = {}
        for point in points:
            scores = []
            x_hat = None
            for iter_i in range(0, num_iter):
                x_hat = apa_fn(x_hat, **dict(zip(tags, point)))
                curr_score = score_fn(x_hat)
                scores.append(curr_score)
            res[','.join(point)] = scores
        return pd.DataFrame(res)

    @staticmethod
    def params_robust(dataset_name='cora'):
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        est_dataset = EstDataset(dataset_name, (0.4, 0.1, 0.5), seed, max_num_iter=30, min_num_iter=1, k_index=-1,
                                 early_stop=False)
        est_dataset.metric = "CORR"
        apa = APA(est_dataset.data.edges, est_dataset.data.x, est_dataset.data.trn_mask, est_dataset.data.is_binary)
        df = PathTest.score_paths([i/100 for i in range(0, 101)], 30, apa.pr, est_dataset.score)
        print(df)
        df.to_csv('params_robust_test')

    @staticmethod
    def params_consistency(dataset_name='cora', metric='CORR', k=50,
                           params_range_kw: dict = {'alpha': (0.0, 1.0), 'beta': (0.0, 1.0)}):
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        est_dataset = EstDataset(dataset_name, (0.4, 0.1, 0.5), seed, max_num_iter=30, min_num_iter=1, k_index=-1, early_stop=False)
        est_dataset.metric = "CORR"
        apa = APA(est_dataset.data.edges, est_dataset.data.x, est_dataset.data.trn_mask, est_dataset.data.is_binary)
        apa_fn = apa.umtp
        s = SearchPoints(**params_range_kw)

        s.last_max_score = 0
        s.last_end_score = 0
        s.last_scores = []

        def fn(point):
            scores = []
            x_hat = None
            for iter_i in range(1, 31):
                x_hat = apa_fn(x_hat, **{tag: p for tag, p in zip(s.tags, point)})
                curr_score = est_dataset.score(x_hat)
                scores.append(curr_score)
            max_score = max(scores)
            end_score = scores[-1]
            if max_score > s.last_max_score and end_score > s.last_end_score:
                s.last_max_score = max_score
                s.last_end_score = end_score
                s.last_scores.append((point, max_score, scores))
                print(point, max_score, scores)
            elif max_score < s.last_max_score and end_score < s.last_end_score:
                pass
            else:
                s.last_scores.append((point, max_score, scores))
                print(point, max_score, scores)
            return max_score, 0, ''

        s.search(fn, greater_is_better(metric))
        for v in s.last_scores:
            print(v)

    @staticmethod
    def early_stop(dataset_name='pubmed', metric='CORR', k=50, params_kw={'alpha': 0.9921875, 'beta': 0.90625}):
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes, num_classes = d.load_data(dataset_name,
                                                                                              split=(0.4, 0.1, 0.5),
                                                                                              seed=seed)
        apa = APA(x_all, y_all, edge_index, trn_nodes, d.is_binary(dataset_name))

        best_score = None
        best_test_score = None
        best_num_iter = 0
        x_hat = None
        print(f"epoch curr_score test_score best_num_iter best_score best_test_score")
        for iter_i in range(1, 100):
            x_hat = apa.fp(x_hat, **params_kw)
            curr_score = calc_single_score(dataset_name, x_hat, x_all, val_nodes, metric, k)
            test_score = calc_single_score(dataset_name, x_hat, x_all, test_nodes, metric, k)
            if best_score is None or greater_is_better(metric) == (curr_score > best_score):
                best_score = curr_score
                best_test_score = test_score
                best_num_iter = iter_i
            print(
                f"{iter_i:{len('epoch')}d} {curr_score:{len('curr_score')}.4f} {test_score:{len('test_score')}.4f} {best_num_iter:{len('best_num_iter')}d} {best_score:{len('best_score')}.4f} {best_test_score:{len('best_test_score')}.4f}")




