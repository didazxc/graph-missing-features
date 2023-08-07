from amgraph.data.data import load_data
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import assortativity, get_laplacian, convert, to_scipy_sparse_matrix
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")


def to_dirichlet_loss(attrs, laplacian):
    # return torch.trace(torch.mm(attrs.T, torch.spmm(laplacian, attrs)))
    return torch.bmm(attrs.t().unsqueeze(1), laplacian.matmul(attrs).t().unsqueeze(2)).view(-1).sum()
    

def attr_sparsity(node_x):
    n_nodes=node_x.size(0)
    n_attrs=node_x.size(1)
    return torch.count_nonzero(node_x).item()/n_nodes/n_attrs


def density(node_x, edges):
    n_nodes = node_x.size(0)
    return 2*edges.size(1)/(n_nodes*(n_nodes-1))


def num_components(node_x, edges):
    n_nodes = node_x.size(0)
    adj = to_scipy_sparse_matrix(edge_index=edges, num_nodes=n_nodes)
    num = sp.csgraph.connected_components(adj, directed=False, connection="weak", return_labels=False)
    return num


def homophily(node_x, edges, normalize:bool=True):
    mean = node_x.mean(dim=0) if normalize else 0
    std = node_x.std(dim=0) if normalize else 1
    x = (node_x - mean) / std
    x[torch.isinf(x)] = 0
    x[torch.isnan(x)] = 0
    n_nodes = x.size(0)
    edge_index, edge_weight = get_laplacian(edges, num_nodes=n_nodes, normalization="sym")
    L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes))
    return int(to_dirichlet_loss(x, L))


def draw(data_name, x, edge_index):
    pyg_data = Data(x=x, edge_index=edge_index)
    G = convert.to_networkx(pyg_data, to_undirected=True)
    plt.figure(1, figsize=(100, 100))
    nx.draw(G, pos=nx.spring_layout(G), node_size=100)
    plt.savefig(f"{data_name}_spring.svg", dpi=600)


def draw_spectral(data_name, x, edge_index):
    pyg_data = Data(x=x, edge_index=edge_index)
    G = convert.to_networkx(pyg_data, to_undirected=True)
    plt.figure(2, figsize=(100, 100))
    nx.draw(G, pos=nx.spectral_layout(G), node_size=100)
    plt.savefig(f"{data_name}_spectral.svg", dpi=600)


def main():
    need_draw_spring = False
    need_draw_spectral = False
    data_names = ['products']  # ['cora', 'citeseer', 'computers', 'photo', 'steam', 'pubmed', 'cs', 'arxiv']
    print(f'{"":10s} nodes_num edges_num attrs_num homophily num_components assortativity density attr_sparsity')
    for data_name in data_names:
        data = load_data(data_name)
        edges, node_x = data.edges, data.x
        print(f'{data_name:10s} {node_x.size(0):{len("nodes_num")}d} {edges.size(1):{len("edges_num")}d} {node_x.size(1):{len("attrs_num")}d} {homophily(node_x, edges):{len("homophily")}d} {num_components(node_x, edges):{len("num_components")}d} {assortativity(edges):{len("assortativity")}.4f} {density(node_x, edges):{len("density")}.4f} {attr_sparsity(node_x):{len("attr_sparsity")}.4f}')
        if need_draw_spring:
            draw(data_name, x=node_x, edge_index=edges)
        if need_draw_spectral:
            draw_spectral(data_name, x=node_x, edge_index=edges)
