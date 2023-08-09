from ..data import data as d
from ..metrics import to_acc
from ..utils import Scores
from ..models.apa import APA
from ..models.gnn import GNN
from ..models.mlp import MLP
import logging
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch_geometric.utils import subgraph

logger = logging.getLogger('amgraph.validators.classification')
default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ClassificationValidator:

    def __init__(self, scores: Scores, data: d.Dataset, epoches,
                 seed, seed_idx, k_index=0) -> None:
        self.scores = scores
        self.dataset_name = data.data_name
        self.is_multi_task = data.is_multi_task

        self.edges = data.edges
        self.y_all = data.y
        self.test_nodes = data.test_mask

        self.num_classes = data.num_classes
        self.num_attrs = data.num_attrs

        self.seed = seed
        self.seed_idx = seed_idx
        self.epoches = epoches
        if data.is_continuous:
            self.ks = [-1]
            self.metrics = ['RMSE', 'CORR']
        else:
            self.ks = [3, 5, 10] if self.dataset_name.startswith("steam") else [10, 20, 50]
            self.metrics = ["Recall", "nDCG"]
        self.metric, self.k = self.metrics[k_index], self.ks[k_index]

    @staticmethod
    def train(model, loss_fn, optimizer, epoches, x, edges, y, trn_nodes, val_nodes, is_multi_task: bool = False) -> torch.Tensor:
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
                train_acc = to_acc(y_hat[trn_nodes], y[trn_nodes], is_multi_task).to("cpu")
                acc = to_acc(y_hat[val_nodes], y[val_nodes], is_multi_task).to("cpu")
            if epoch % 100 == 0:
                logger.debug(f"{epoch:4d} best_loss:{best_loss:7.5f} train_acc:{train_acc:7.5f} acc:{acc:7.5f}")
        return acc

    def calc_acc(self, x_hat, conv="GCN"):
        acc_list = []
        splits = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        logger.info(f"start model {conv}")
        for trn_nodes, val_nodes in splits.split(self.test_nodes):
            if conv == "GCN":
                model = GNN(self.num_attrs, self.num_classes, num_layers=3, hidden_size=64).to(x_hat.device)
            else:
                model = MLP(self.num_attrs, self.num_classes, num_layers=2, hidden_size=256).to(x_hat.device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            acc = self.train(model, loss_fn, optimizer, self.epoches, x_hat, self.edges, self.y_all, trn_nodes,
                             val_nodes, self.is_multi_task)
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

    def validate_from_scores(self, algo_name, apa_fn, est_scores: Scores, val_only_once):
        convs = ["MLP"]  # ["GCN", "MLP"]
        metrics, ks = ([self.metric], [self.k]) if val_only_once else (self.metrics, self.ks)
        for metric in metrics:
            for k in ks:
                row = f"{metric}_{algo_name}"
                col = f"{self.dataset_name}@{k}_params"

                if any([not self._is_executed(row, col, conv) for conv in convs]):
                    params = {} if algo_name.startswith("raw_") else est_scores.get_value(row, col)[self.seed_idx]
                    logger.info(f"{algo_name}, {row}, {col}, {params}")
                    x_hat = apa_fn(**params)

                    for conv in convs:
                        if not self._is_executed(row, col, conv):
                            acc = self.calc_acc(x_hat, conv)
                            self._add_score(row, col, conv, acc)
        self.scores.print()

    @staticmethod
    def run(file_name="class_all_umtp30", est_scores_file_name="all_umtp30",
            dataset_names=['cora', 'citeseer', 'computers', 'photo', 'pubmed', 'cs', 'arxiv'],
            run_algos=['raw', 'fp', 'pr', 'ppr', 'mtp', 'umtp', 'umtp2'], val_only_once=True, k_index=0):
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
                data = d.load_data(dataset_name, split=(0.4, 0.1, 0.5), seed=seed).to(default_device, True)
                apa = APA(data.edges, data.x, data.trn_mask, data.is_binary)
                induced_edge_index, _ = subgraph(data.test_mask, data.edges)
                c = ClassificationValidator(scores, data, epoches=epoches, seed=seed, seed_idx=seed, k_index=k_index)

                def raw_fn(**kw):
                    return data.x

                algos = {
                    'raw': raw_fn,
                    'fp': apa.fp,
                    'pr': apa.pr,
                    'ppr': apa.ppr,
                    'mtp': apa.mtp,
                    'umtp': apa.umtp,
                    'umtp_1_0': apa.umtp,
                    'mtp_partial': apa.mtp_partial,
                    'umtp2': apa.umtp2,
                    'umtp_beta': apa.umtp_beta
                }
                for algo_name in run_algos:
                    print(dataset_name, seed, algo_name)
                    c.validate_from_scores(algo_name, algos[algo_name], est_scores, val_only_once)

        scores.print()
        print("end")

