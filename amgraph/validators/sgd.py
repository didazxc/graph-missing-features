from ..data import data as d
from ..metrics import calc_single_score, greater_is_better
from ..utils import SearchPoints, Scores
from ..models.apa import APA, UMTPLoss, UMTPwithParams
from .estimation import EstimationValidator
import logging
import numpy as np
import torch
from torch import nn

logger = logging.getLogger('amgraph.validators.sgd')


class SGDValidator:

    @staticmethod
    def compare(dataset_name, apa: APA, val_nodes, params, metric, k, epochs):
        s = Scores("compare_rate")
        s.load()

        edge_index, x_all, trn_nodes = apa.edge_index, apa.x, apa.know_mask
        umtp = UMTPLoss(edge_index, x_all, trn_nodes, is_binary=apa.is_binary, **params)

        # analytical_solution
        x_hat = apa.umtp_analytical_solution(**params)
        if not s.has_value("analytical_solution_score", dataset_name, 0):
            score = calc_single_score(dataset_name, x_hat, x_all, val_nodes, metric, k)
            loss = umtp.get_loss(x_hat)
            s.add_score("analytical_solution_loss", dataset_name, loss.item())
            s.add_score("analytical_solution_score", dataset_name, score)
            print(f"umtp_analytical_solution score={score:7.5f} loss={loss:7.5f}, metric={metric}, k={k}")
            s.save()
        if not s.has_value("prop_mse", dataset_name, epochs - 1):
            # prop
            out = None
            for epoch in range(epochs):
                if epoch == 0:
                    out = apa.out
                else:
                    out = apa.umtp(out, **params)
                score = calc_single_score(dataset_name, out, x_all, val_nodes, metric, k)
                loss = umtp.get_loss(out)
                mse = torch.nn.functional.mse_loss(x_hat, out, reduction='sum')
                s.add_score("prop_score", dataset_name, score)
                s.add_score("prop_loss", dataset_name, loss.item())
                s.add_score("prop_mse", dataset_name, mse.item())
                print(f"prop epoch={epoch}, loss={loss}, mse={mse}, score={score}")
            s.save()
            print(f"end prop at epoch={epoch}")
        if not s.has_value("sgd_mse", dataset_name, epochs - 1):
            # sgd
            optimizer = torch.optim.Adam(umtp.parameters(), lr=0.01)
            for epoch in range(epochs):
                optimizer.zero_grad()
                loss = umtp()
                loss.backward()
                optimizer.step()
                score = calc_single_score(dataset_name, umtp.get_out(), x_all, val_nodes, metric, k)
                mse = torch.nn.functional.mse_loss(x_hat, umtp.get_out(), reduction='sum')
                s.add_score("sgd_score", dataset_name, score)
                s.add_score("sgd_loss", dataset_name, loss.item())  # before backward
                s.add_score("sgd_mse", dataset_name, mse.item())
                print(f"sgd epoch={epoch}, loss={loss}, mse={mse}, score={score}")
            s.save()
            print(f"end sgd at epoch={epoch}")
        s.print('raw')

    @staticmethod
    def sgd(file_name="sgd_score", dataset_names=['pubmed'], k_index=-1,
            params_range_kw={'alpha': (0.0, 1.0), 'beta': (0.0, 1.0)}):
        epochs = 400
        scores = Scores(file_name=file_name)
        scores.load()
        for seed in range(1):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            for dataset_name in dataset_names:
                s = SearchPoints(**params_range_kw)
                all_metrics, all_ks = EstimationValidator.metric_and_ks(dataset_name)
                metric, k = all_metrics[k_index], all_ks[k_index]
                data = d.load_data(dataset_name, split=( 0.4, 0.1, 0.5), seed=seed)

                def fn(point):
                    params = {tag: p for tag, p in zip(s.tags, point)}
                    umtp = UMTPLoss(data.edges, data.x, data.trn_mask, is_binary=data.is_binary, **params)
                    optimizer = torch.optim.Adam(umtp.parameters(), lr=0.01)
                    best_score = None
                    for epoch in range(epochs):
                        optimizer.zero_grad()
                        loss = umtp()
                        loss.backward()
                        optimizer.step()
                        score = calc_single_score(dataset_name, umtp.get_out(), data.x, data.val_mask, metric, k)
                        if best_score is None or (greater_is_better(metric) == (best_score < score)):
                            best_score = score
                        print(f"sgd epoch={epoch}, loss={loss}, score={score}, best_score={best_score}")

                    return best_score, (umtp.get_out(), epoch)

                s.search(fn, greater_is_better(metric))
                best_params = {tag: p for tag, p in zip(s.tags, s.best_points[0])}
                best_x_hat, iter_count = s.best_out[0]
                best_params['iter_count'] = iter_count

                for metric in all_metrics:
                    row = f"{metric}"
                    for ki in all_ks:
                        col = f"{dataset_name}@{ki}"
                        score = calc_single_score(dataset_name, best_x_hat, data.x, data.test_mask, metric, ki)
                        scores.append((row, col, score))
                    scores.append((row, f"{dataset_name}@{k}_params", best_params))

            scores.print()

    @staticmethod
    def run_compare(dataset_names=['pubmed', 'cs', 'arxiv'], k_index=-1, epochs=120):
        params_dict = {
            'pubmed': {'alpha': 1.0, 'beta': 0.625},
            'cs': {'alpha': 1.0, 'beta': 0.0859375},
            'arxiv': {'alpha': 1.0, 'beta': 0.375}
        }
        for seed in range(1):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            for dataset_name in dataset_names:
                params = params_dict[dataset_name]
                all_metrics, all_ks = EstimationValidator.metric_and_ks(dataset_name)
                metric, k = all_metrics[k_index], all_ks[k_index]

                data = d.load_data(dataset_name, split=( 0.4, 0.1, 0.5), seed=seed)

                apa = APA(data.edges, data.x, data.trn_mask, data.is_binary)

                SGDValidator.compare(dataset_name, apa, data.val_mask, params, metric, k, epochs)

    @staticmethod
    def sgd_params_search(dataset_names=['pubmed'], epochs=2000, k_index=-1):
        for seed in range(1):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            for dataset_name in dataset_names:

                all_metrics, all_ks = EstimationValidator.metric_and_ks(dataset_name)
                metric, k = all_metrics[k_index], all_ks[k_index]
                edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes, num_classes = d.load_data(dataset_name,
                                                                                                      split=(
                                                                                                      0.4, 0.1, 0.5),
                                                                                                      seed=seed)
                apa = APA(x_all, y_all, edge_index, trn_nodes, d.is_binary(dataset_name))
                alpha, beta = nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))

                loss_fn = nn.MSELoss()
                optimizer = torch.optim.Adam([alpha, beta], lr=0.1)
                best_score = None
                test_score = None

                no_improve_count = 0

                for epoch in range(epochs):
                    alpha1, beta1 = torch.sigmoid(alpha), torch.sigmoid(beta)
                    x_hat = apa.umtp(alpha=alpha1, beta=beta1, num_iter=30)
                    loss = loss_fn(x_hat[val_nodes], x_all[val_nodes])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    score = calc_single_score(dataset_name, x_hat, x_all, val_nodes, metric, k)
                    if best_score is None or (greater_is_better(metric) == (score > best_score)):
                        best_score = score
                        test_score = [calc_single_score(dataset_name, x_hat, x_all, test_nodes, metric, k) for k in
                                      all_ks]
                        test_score_str = ','.join([f"{s:7.5f}" for s in test_score])
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    print(
                        f"seed={seed:2d} dataset_name={dataset_name:8s} metric={metric:5s} epoch={epoch:4d} best_score={best_score:7.5f} test_score=({test_score_str}) loss={loss:7.5f} score={score:7.5f} alpha1={alpha1.item():7.5f} beta1={beta1.item():7.5f} {alpha.item():10f} {beta.item():10f}")

                    if no_improve_count >= 30:
                        break
                print(epoch - 30, alpha1.item(), beta1.item(), best_score, test_score_str)

    @staticmethod
    def sgd_params_vector(dataset_names=['pubmed'], epochs=2000, k_index=-1):
        for seed in range(1):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            for dataset_name in dataset_names:

                all_metrics, all_ks = EstimationValidator.metric_and_ks(dataset_name)
                metric, k = all_metrics[k_index], all_ks[k_index]
                edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes, num_classes = d.load_data(dataset_name,
                                                                                                      split=(
                                                                                                      0.4, 0.1, 0.5),
                                                                                                      seed=seed)
                umtp = UMTPwithParams(x_all, y_all, edge_index, trn_nodes, d.is_binary(dataset_name))

                loss_fn = nn.MSELoss()
                optimizer = torch.optim.Adam(umtp.parameters(), lr=0.05)
                best_score = None
                test_score = None

                no_improve_count = 0

                for epoch in range(epochs):
                    x_hat = umtp()
                    loss = loss_fn(x_hat[val_nodes], x_all[val_nodes])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    score = calc_single_score(dataset_name, x_hat, x_all, val_nodes, metric, k)
                    if best_score is None or (greater_is_better(metric) == (score > best_score)):
                        best_score = score
                        test_score = [calc_single_score(dataset_name, x_hat, x_all, test_nodes, metric, k) for k in
                                      all_ks]
                        test_score_str = ','.join([f"{s:7.5f}" for s in test_score])
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    print(
                        f"seed={seed:2d} dataset_name={dataset_name:8s} metric={metric:5s} epoch={epoch:4d} best_score={best_score:7.5f} test_score=({test_score_str}) loss={loss:7.5f} score={score:7.5f}")

                    if no_improve_count >= 30:
                        break
                print(epoch - 30, best_score, test_score_str)

