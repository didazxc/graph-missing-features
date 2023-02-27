import logging
from typing import Callable, List, Any, Tuple
import json
import numpy as np
import pandas as pd
import functools
import time


logging.basicConfig(level=logging.WARN,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('g.apa')
logger.setLevel(logging.DEBUG)


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
    def __init__(self) -> None:
        self.dict = {}
    
    def add_score(self, row, col, value):
        if col not in self.dict:
            self.dict[col] = {}
        if isinstance(value, str):
            if row not in self.dict[col]:
                self.dict[col][row] = value
            else:
                self.dict[col][row] += value
        else:
            if row not in self.dict[col]:
                self.dict[col][row] = [value]
            else:
                self.dict[col][row].append(value)

    def print(self, file_name=None):
        self.save()
        df_dict = {col:{row: np.mean(row_v) if isinstance(row_v, list) else row_v for row, row_v in col_v.items()} for col, col_v in self.dict.items()}
        df = pd.DataFrame(df_dict)
        if file_name is not None:
            df.to_csv(f"{file_name}.csv")
            self.save(file_name)
        print(df)
    
    def save(self, file_name='scores.npy'):
        np.save(file_name, self.dict)
    
    def load(self, file_name='scores.npy'):
        try:
            self.dict = np.load(file_name).item()
        except Exception:
            logger.info(f'cannot load data, maybe no {file_name}. So use an empty scores.')


class SearchPoints:
    def __init__(self, **params_range_kw):
        """
        params_range_kw = {'alpha':(0,1), 'beta':(0,1)}
        """
        self.tags = list(params_range_kw.keys())
        self.ranges = [params_range_kw[t] for t in self.tags]
        self.center_point = [(a[0]+a[1])/2 for a in self.ranges]
        self.steps = [(a[1]-a[0]) for a in self.ranges]
        self.searched_points = set()
        self.best_points = [self.center_point]
        self.best_score = None
        self.best_out = None

    def expand_neighbors(self, steps: List[float], ranges: List[Tuple[float, float]]):
        """
        steps: the step of every param
        """
        points = set()
        for center_point in self.best_points:
            for i, (v, step) in enumerate(zip(center_point, steps)):
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

    def search_points(self, points:List[List[float]], score_fn:Callable[[List[float]], Tuple[float, Any]], greater_is_better: bool=True):
        for point in points:
            score, out = score_fn(point)
            self.searched_points.add(json.dumps(point))
            if score == self.best_score:
                self.best_score = score
                self.best_points.append(point)
                self.best_out.append(out)
            elif self.best_score is None or greater_is_better == (score > self.best_score):
                self.best_score = score
                self.best_points = [point]
                self.best_out = [out]
            logger.debug(f"point={point}/{points}, best_points={self.best_points}, best_score={self.best_score}, searched_points={self.searched_points}")

    def search(self, score_fn:Callable[[List[float]], Tuple[float, Any]], greater_is_better:bool=True, precision:float=0.01):
        points = self.best_points.copy()
        self.search_points(points, score_fn, greater_is_better)
        steps = self.steps
        while max(steps) > precision:
            steps = [s / 2 for s in steps]
            ranges = [[(i-s if i-s > r[0] else r[0], i+s if i+s < r[1] else r[1]) for i, s, r in zip(best_point, steps, self.ranges)] for best_point in self.best_points]
            points = self.expand_neighbors(steps, ranges)
            while len(points):
                self.search_points(points, score_fn, greater_is_better)
                points = self.expand_neighbors(steps, ranges)
