import os
from tqdm import tqdm
from argparse import Namespace
import pickle
import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset  # ogb must import before pyg and scipy, or it will get stuck
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import scipy.sparse as sp


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


def load_steam(root):
    """
    this code is copied from SVGA
    """
    if not os.path.exists(os.path.join(root, '../data/Steam', 'processed', 'sp_fts.pkl')):
        precess_steam(root)

    freq_item_mat = pickle.load(open(os.path.join(root, '../data/Steam', 'processed', 'freq_item_mat.pkl'), 'rb'))
    features = pickle.load(open(os.path.join(root, '../data/Steam', 'processed', 'sp_fts.pkl'), 'rb'))
    features = torch.from_numpy(features.todense()).float()
    labels = torch.zeros(features.size(0), dtype=torch.int)

    adj = freq_item_mat.copy()
    adj[adj < 10.0] = 0.0
    adj[adj >= 10.0] = 1.0
    indices = np.where(adj != 0.0)
    rows = indices[0]
    cols = indices[1]
    edge_index = torch.from_numpy(np.stack([rows, cols], axis=0))
    return Namespace(data=Namespace(x=features, y=labels, edge_index=edge_index))


def load_data(data_name):
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if data_name == 'steam':
        data = load_steam(root)
    elif data_name == 'arxiv':
        data = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
    elif data_name == 'cora':
        data = Planetoid(root, 'Cora')
    elif data_name == 'citeseer':
        data = Planetoid(root, 'CiteSeer')
    elif data_name == 'pubmed':
        data = Planetoid(root, 'PubMed')
    elif data_name == 'computers':
        data = Amazon(root, 'Computers')
    elif data_name == 'photo':
        data = Amazon(root, 'Photo')
    elif data_name == 'cs':
        data = Coauthor(root, 'CS')
    elif data_name == 'physics':
        data = Coauthor(root, 'Physics')
    else:
        raise ValueError(data_name)

    return data


if __name__ == '__main__':
    load_data('arxiv')
