import logging
from typing import Callable, List, Any, Tuple
import json
import numpy as np
import pandas as pd
import functools
import time
import random
import numbers


logging.basicConfig(level=logging.WARN,
                    format='%(asctime)s %(name)-10s %(filename)-12s[line:%(lineno)-3d] %(funcName)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('amgraph.apa')
logger.setLevel(logging.DEBUG)


def get_time_cost(s_time):
    e_time = time.time()
    m, s = divmod(e_time - s_time, 60)
    h, m = divmod(m, 60)
    return f'{int(h):2d}h{int(m):2d}m{s:5.2f}s'


def print_time_cost(fn):
    @functools.wraps(fn)
    def wrapper(*arg, **kwarg):
        s_time = time.time()
        res = fn(*arg, **kwarg)
        e_time = time.time()
        m, s = divmod(e_time - s_time, 60)
        h, m = divmod(m, 60)
        logger.info(f'{fn.__qualname__:40s} time_cost-{int(h):2d}h{int(m):2d}m{s}s')
        return res

    return wrapper


class Scores:
    def __init__(self, file_name="scores.npy") -> None:
        self.dict = {}
        if not file_name.endswith(".npy"):
            self.file_name = f"{file_name}.npy"
        else:
            self.file_name = file_name

    def add_score(self, row, col, value, idx=None):
        if col not in self.dict:
            self.dict[col] = {}
        if idx is None:
            if row not in self.dict[col]:
                self.dict[col][row] = [value]
            else:
                self.dict[col][row].append(value)
        else:
            if row not in self.dict[col]:
                self.dict[col][row] = []
            s = len(self.dict[col][row])
            for _ in range(s, idx):
                self.dict[col][row].append(None)
            if len(self.dict[col][row])<=idx:
                self.dict[col][row].append(value)
            else:
                self.dict[col][row][idx] = value

    def get_value(self, row, col):
        return self.dict[col][row] if col in self.dict and row in self.dict[col] else None

    def has_value(self, row, col, idx):
        cnt = len(self.dict[col][row]) if col in self.dict and row in self.dict[col] else 0
        return cnt > idx

    def print(self, reduction='mean'):
        self.save()
        df_dict = {col: {
            row: np.mean(row_v) if isinstance(row_v, list) and all([isinstance(v, numbers.Number) for v in row_v]) and reduction=='mean' else row_v
            for row, row_v in col_v.items()
        } for col, col_v in self.dict.items()}
        df = pd.DataFrame(df_dict)
        df.to_csv(self.file_name.replace('.npy','.csv'))
        print(df)

    def save(self):
        np.save(self.file_name, self.dict)
        return self

    def combine(self, file_names):
        self.dict = {}
        for file_name in file_names:
            if not file_name.endswith(".npy"):
                file_name = f"{file_name}.npy"
            self.dict.update(np.load(file_name, allow_pickle=True).item())
        return self

    def load(self):
        try:
            self.dict = np.load(self.file_name, allow_pickle=True).item()
        except Exception:
            logger.info(f'cannot load data, maybe no {self.file_name}. So use an empty scores.')
        return self


class SearchPoints:
    def __init__(self, **params_range_kw):
        """
        Search for the optimal params in the range of *params_range_kw*, then save best params in self.best_points, and save the output of *score_fn* in self.best_score and self.best_out.
        params_range_kw = {'alpha':(0,1), 'beta':(0,1)}
        """
        self.tags = list(params_range_kw.keys())
        self.ranges = [params_range_kw[t] for t in self.tags]
        self.center_point = [(a[0] + a[1]) / 2 for a in self.ranges]
        self.steps = [a[1] - a[0] for a in self.ranges]
        self.searched_points = set()
        self.best_points = [self.center_point]
        self.best_score = None
        self.best_out = None
        self.max_best_points_num = 4

    def expand_neighbors(self, steps: List[float], ranges: List[List[Tuple[float, float]]]) -> List[list]:
        """
        steps: the step of every param
        expande neighbors for self.best_points in ranges 
        """
        points = set()
        for center_point in self.best_points:
            for i, (v, step) in enumerate(zip(center_point, steps)):
                if step == 0:
                    continue
                for s in [-step, step]:
                    value = v + s
                    for v_range in ranges:
                        if v_range[i][0] <= value <= v_range[i][1]:
                            point = center_point.copy()
                            point[i] = value
                            point_str = json.dumps(point)
                            if point_str not in self.searched_points:
                                points.add(point_str)
        return [json.loads(point_str) for point_str in points]

    def search_points(self, points: List[List[float]], score_fn: Callable[[List[float]], Tuple[float, Any, str]],
                      greater_is_better: bool = True):
        """
        Search for the optimal param in the *points*, then update self.best_points.
        """
        for idx, point in enumerate(points):
            t_start = time.time()
            score, out, *msg = score_fn(point)
            self.searched_points.add(json.dumps(point))
            if score == self.best_score:
                self.best_points.append(point)
                self.best_out.append(out)
                if len(self.best_points) > self.max_best_points_num:
                    self.best_points.pop(random.randint(0, self.max_best_points_num))
            elif self.best_score is None or greater_is_better == (score > self.best_score):
                self.best_score = score
                self.best_points = [point]
                self.best_out = [out]
            cost_str = get_time_cost(t_start)
            debug_msg = msg[0] if len(msg) > 0 else ''
            logger.debug(
                f"{debug_msg} {len(self.searched_points):4d}:{str(point):20s} {idx + 1:3d}/{len(points):3d} cost={cost_str} score={score:.6f}/{self.best_score:.6f} best_points={self.best_points}")

    def search(self, score_fn: Callable[[List[float]], Tuple[float, Any, str]], greater_is_better: bool = True,
               precision: float = 0.01):
        """
        score_fn returns score and x_hat
        """
        points = self.best_points.copy()
        self.search_points(points, score_fn, greater_is_better)
        steps = self.steps
        while max(steps) > precision:
            steps = [s / 2 for s in steps]
            ranges = [[(i - s if i - s > r[0] else r[0], i + s if i + s < r[1] else r[1]) for i, s, r in
                       zip(best_point, steps, self.ranges)] for best_point in self.best_points]
            points = self.expand_neighbors(steps, ranges)
            while len(points):
                self.search_points(points, score_fn, greater_is_better)
                points = self.expand_neighbors(steps, ranges)
