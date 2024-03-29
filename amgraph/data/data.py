import os
from tqdm import tqdm
from argparse import Namespace
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.typing import Adj
from torch_geometric.utils import to_undirected, remove_self_loops, add_remaining_self_loops
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Yelp, Reddit
import torch_geometric.transforms as transforms
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


def is_binary(data):
    return data in ['cora', 'citeseer', 'computers', 'photo', 'steam', 'cs', 'steam20', 'steam1']


def is_continuous(data):
    return data in ['pubmed', 'coauthor', 'cs', 'arxiv', 'products', 'reddit', 'yelp']


def is_connect(data):
    return data in ['pubmed', 'coauthor', 'cs', 'arxiv', 'reddit']


def validate_edges(edges):
    """
    Validate the edges of a graph with various criteria.
    """
    # No self-loops
    for src, dst in edges.t():
        if src.item() == dst.item():
            raise ValueError(f"{src} has self-loops")

    # Each edge (a, b) appears only once.
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if dst in m[src]:
            raise ValueError(f"({src}->{dst}) appears more than once")
        m[src].add(dst)

    # Each pair (a, b) and (b, a) exists together.
    for src, neighbors in m.items():
        for dst in neighbors:
            if src not in m[dst]:
                raise ValueError(f"({src}->{dst}) lack of inverse edge")


def precess_steam(root):
    """
    this code is copied from SAT
    only change raw and processed directory and encoding with utf-8
    """
    raw_dir = os.path.join(root, '../data/Steam', 'raw')
    processed_dir = os.path.join(root, '../data/Steam', 'processed')
    if not os.path.exists(os.path.join(processed_dir, 'freq_item_mat.pkl')):
        itemID_userID_dict = {}
        with open(os.path.join(raw_dir, 'user_product_date.csv'), 'rb') as f:
            lines = f.readlines()[1:]
            for i in tqdm(range(len(lines))):
                line = str(lines[i].strip(), 'utf-8').split('|')
                userID = line[1]
                itemID = line[2]
                if itemID not in itemID_userID_dict:
                    itemID_userID_dict[itemID] = set()
                    itemID_userID_dict[itemID].add(userID)
                else:
                    itemID_userID_dict[itemID].add(userID)
        itemID_unique = list(itemID_userID_dict.keys())
        itemID_Idx_map = pd.Series(data=range(len(itemID_unique)), index=itemID_unique)
        freq_item_mat = np.zeros(shape=[len(itemID_userID_dict), len(itemID_userID_dict)])
        for i in tqdm(range(len(itemID_unique))):
            itemID1 = itemID_unique[i]
            for j in range(i + 1, len(itemID_unique)):
                itemID2 = itemID_unique[j]
                freq_item_mat[itemID_Idx_map[itemID1], itemID_Idx_map[itemID2]] = len(itemID_userID_dict[itemID1] & itemID_userID_dict[itemID2])
                freq_item_mat[itemID_Idx_map[itemID2], itemID_Idx_map[itemID1]] = len(itemID_userID_dict[itemID1] & itemID_userID_dict[itemID2])
        pickle.dump(itemID_Idx_map, open(os.path.join(processed_dir, 'itemID_Idx_map.pkl'), 'wb'))
        pickle.dump(freq_item_mat, open(os.path.join(processed_dir, 'freq_item_mat.pkl'), 'wb'))
    else:
        itemID_Idx_map = pickle.load(open(os.path.join(processed_dir, 'itemID_Idx_map.pkl'), 'rb'))

    # construct sparse feature matrix
    itemID_tagID_dict = {}
    with open(os.path.join(raw_dir, 'product_tags.csv'), 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for i in tqdm(range(len(lines))):
            line = lines[i].strip().split('|')
            itemID = line[0]
            tagIDSet = set(line[2].strip('[').strip(']').split(', '))
            if itemID in list(itemID_Idx_map.index):
                if itemID not in itemID_tagID_dict:
                    itemID_tagID_dict[itemID] = tagIDSet
                else:
                    itemID_tagID_dict[itemID] = itemID_tagID_dict[itemID] | tagIDSet
            else:
                pass

    all_tags = set()
    for ele in itemID_tagID_dict:
        all_tags = all_tags | itemID_tagID_dict[ele]
    all_tags = list(all_tags)
    tagID_Idx_map = pd.Series(data=range(len(all_tags)), index=all_tags)

    indices = []
    values = []
    for itemID in tqdm(itemID_tagID_dict):
        itemIdx = itemID_Idx_map[itemID]
        tagID_list = list(itemID_tagID_dict[itemID])
        for tagID in tagID_list:
            tagIdx = tagID_Idx_map[tagID]
            indices.append([itemIdx, tagIdx])
            values.append(1.0)
    indices = np.array(indices)
    values = np.array(values)
    sp_fts = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=[len(itemID_Idx_map), len(tagID_Idx_map)])
    pickle.dump(sp_fts, open(os.path.join(processed_dir, 'sp_fts.pkl'), 'wb'))
    pickle.dump(itemID_tagID_dict, open(os.path.join(processed_dir, 'itemID_tagID_dict.pkl'), 'wb'))
    pickle.dump(tagID_Idx_map, open(os.path.join(processed_dir, 'tagID_Idx_map.pkl'), 'wb'))


def load_steam(root, threshold: float = 10.0):
    if not os.path.exists(os.path.join(root, '../data/Steam', 'processed', 'sp_fts.pkl')):
        os.mkdir(os.path.join(root, '../data/Steam', 'processed'))
        precess_steam(root)

    freq_item_mat = pickle.load(open(os.path.join(root, '../data/Steam', 'processed', 'freq_item_mat.pkl'), 'rb'))
    features = pickle.load(open(os.path.join(root, '../data/Steam', 'processed', 'sp_fts.pkl'), 'rb'))
    features = torch.from_numpy(features.todense()).float()
    labels = torch.zeros(features.size(0), dtype=torch.int)

    adj = freq_item_mat.copy()
    adj[adj < threshold] = 0.0
    adj[adj >= threshold] = 1.0
    indices = np.where(adj != 0.0)
    rows = indices[0]
    cols = indices[1]
    edge_index = torch.from_numpy(np.stack([rows, cols], axis=0))
    return Namespace(data=Namespace(x=features, y=labels, edge_index=edge_index), num_classes=1)


class Dataset:

    def __init__(self, data_name: str, edges: Adj, x: torch.Tensor, y: torch.Tensor, trn_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, num_classes: int):
        self.data_name = data_name
        self.edges = edges
        self.x = x
        self.y = y
        self.trn_mask = trn_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = num_classes
        self.num_nodes = x.size(0)
        self.num_attrs = x.size(1) if len(x.size()) > 1 else 1
        self.is_binary = is_binary(data_name)
        self.is_continuous = is_continuous(data_name)
        self.is_connect = is_connect(data_name)
        self.is_multi_task = data_name == "yelp"

    def to(self, device, with_edges: bool = False):
        if with_edges:
            self.edges = self.edges.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.trn_mask = self.trn_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self


def load_data(data_name, split=None, seed=None, verbose=False) -> Dataset:
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    if data_name == 'steam':
        data = load_steam(root)
    elif data_name == 'steam1':
        data = load_steam(root, 1.0)
    elif data_name == 'steam20':
        data = load_steam(root, 20.0)
    elif data_name == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        data = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
        data.data.edge_index = to_undirected(data.data.edge_index)
    elif data_name == 'products':
        from ogb.nodeproppred import PygNodePropPredDataset
        data = PygNodePropPredDataset(name='ogbn-products', root=root)
        data.data.edge_index = remove_self_loops(to_undirected(data.data.edge_index))[0]
    elif data_name == 'papers100M':
        from ogb.nodeproppred import PygNodePropPredDataset
        data = PygNodePropPredDataset(name='ogbn-papers100M', root=root)
        data.data.edge_index = remove_self_loops(to_undirected(data.data.edge_index))
    elif data_name == 'cora':
        data = Planetoid(root, 'Cora')
    elif data_name == 'yelp':
        data = Yelp(f"{root}/Yelp")
    elif data_name == 'reddit':
        data = Reddit(f"{root}/Reddit")
    elif data_name == 'citeseer':
        data = Planetoid(root, 'CiteSeer')
    elif data_name == 'pubmed':
        data = Planetoid(root, 'PubMed')
    elif data_name == 'computers':
        data = Amazon(root, 'Computers')
    elif data_name == 'photo':
        data = Amazon(root, 'Photo')
    elif data_name == 'cs' or data_name == 'coauthor':
        data = Coauthor(root, 'CS')
    elif data_name == 'physics':
        data = Coauthor(root, 'Physics')
    else:
        raise ValueError(data_name)

    dat = data.data

    node_x = dat.x
    node_y = dat.y.squeeze()
    edges = dat.edge_index

    # validate_edges(edges)

    if split is None:
        if hasattr(dat, 'train_mask'):
            trn_mask = dat.train_mask
            val_mask = dat.val_mask
            trn_nodes = torch.nonzero(trn_mask).view(-1)
            val_nodes = torch.nonzero(val_mask).view(-1)
            test_nodes = torch.nonzero(~(trn_mask | val_mask)).view(-1)
        else:
            trn_nodes, val_nodes, test_nodes = None, None, None
    elif len(split) == 3 and sum(split) == 1:
        trn_size, val_size, test_size = split
        indices = np.arange(node_x.shape[0])
        trn_nodes, test_nodes = train_test_split(indices, test_size=test_size, random_state=seed)  # ,stratify=node_y)
        trn_nodes, val_nodes = train_test_split(trn_nodes, test_size=val_size / (trn_size + val_size),
                                                random_state=seed)  # ,stratify=node_y[trn_nodes])

        trn_nodes = torch.from_numpy(trn_nodes).to(torch.long)
        val_nodes = torch.from_numpy(val_nodes).to(torch.long)
        test_nodes = torch.from_numpy(test_nodes).to(torch.long)
    else:
        raise ValueError(split)

    if verbose:
        print('Data:', data_name)
        print('Number of nodes:', node_x.size(0))
        print('Number of edges:', edges.size(1) // 2)
        print('Number of features:', node_x.size(1))
        print('Ratio of nonzero features:', (node_x > 0).float().mean().item())
        print('Number of classes:', node_y.max().item() + 1 if node_y is not None else 0)
        print()

    # return data
    return Dataset(data_name=data_name, edges=edges, x=node_x, y=node_y, trn_mask=trn_nodes, val_mask=val_nodes,
                   test_mask=test_nodes, num_classes=data.num_classes)


if __name__ == '__main__':
    load_data('yelp', split=(0.4, 0.1, 0.5))
    print('Data process done!')
