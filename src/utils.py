# src/utils.py
import numpy as np
import torch
from torch_geometric.data import Data
from pygod.utils.utility import check_parameter

def generate_gt_neighbor(neighbor_dict, node_embeddings, neighbor_num_list, in_dim):
    max_neighbor_num = max(neighbor_num_list)
    all_gt_neighbor_embeddings = []
    for i, embedding in enumerate(node_embeddings):
        neighbor_indexes = neighbor_dict[i]
        neighbor_embeddings = []
        for index in neighbor_indexes:
            neighbor_embeddings.append(node_embeddings[index].tolist())
        if len(neighbor_embeddings) < max_neighbor_num:
            for _ in range(max_neighbor_num - len(neighbor_embeddings)):
                neighbor_embeddings.append(torch.zeros(in_dim).tolist())
        all_gt_neighbor_embeddings.append(neighbor_embeddings)
    return all_gt_neighbor_embeddings


# def gen_joint_structural_outliers(data, m, n, random_state=None):
#     """
#     We randomly select n nodes from the network which will be the anomalies 
#     and for each node we select m nodes from the network. 
#     We connect each of n nodes with the m other nodes.

#     Parameters
#     ----------
#     data : PyTorch Geometric Data instance (torch_geometric.data.Data)
#         The input data.
#     m : int
#         Number nodes in the outlier cliques.
#     n : int
#         Number of outlier cliques.
#     p : int, optional
#         Probability of edge drop in cliques. Default: ``0``.
#     random_state : int, optional
#         The seed to control the randomness, Default: ``None``.

#     Returns
#     -------
#     data : PyTorch Geometric Data instance (torch_geometric.data.Data)
#         The structural outlier graph with injected edges.
#     y_outlier : torch.Tensor
#         The outlier label tensor where 1 represents outliers and 0 represents
#         regular nodes.
#     """

#     if not isinstance(data, Data):
#         raise TypeError("data should be torch_geometric.data.Data")

#     if isinstance(m, int):
#         check_parameter(m, low=0, high=data.num_nodes, param_name='m')
#     else:
#         raise ValueError("m should be int, got %s" % m)

#     if isinstance(n, int):
#         check_parameter(n, low=0, high=data.num_nodes, param_name='n')
#     else:
#         raise ValueError("n should be int, got %s" % n)

#     check_parameter(m * n, low=0, high=data.num_nodes, param_name='m*n')

#     if random_state:
#         np.random.seed(random_state)


#     outlier_idx = np.random.choice(data.num_nodes, size=n, replace=False)
#     all_nodes = [i for i in range(data.num_nodes)]
#     rem_nodes = []
    
#     for node in all_nodes:
#         if node is not outlier_idx:
#             rem_nodes.append(node)
    
    
    
#     new_edges = []
    
#     # connect all m nodes in each clique
#     for i in range(0, n):
#         other_idx = np.random.choice(data.num_nodes, size=m, replace=False)
#         for j in other_idx:
#             new_edges.append(torch.tensor([[i, j]], dtype=torch.long))
                    

#     new_edges = torch.cat(new_edges)


#     outlier_idx = torch.tensor(outlier_idx, dtype=torch.long)
#     y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
#     y_outlier[outlier_idx] = 1

#     data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

#     return data, y_outlier


def gen_joint_structural_outliers(data, m, n, random_state=None):
    from torch_geometric.data import Data
    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")
    check_parameter(m, low=0, high=data.num_nodes, param_name='m')
    check_parameter(n, low=0, high=data.num_nodes, param_name='n')
    check_parameter(m * n, low=0, high=data.num_nodes, param_name='m*n')
    if random_state is not None:
        np.random.seed(random_state)

    # n개의 이상치 노드 선택
    outlier_idx = np.random.choice(data.num_nodes, size=n, replace=False)

    # 엣지 추가: 각 outlier v에 대해 v→(무작위 m개) 연결(자기 자신/중복 제외)
    new_src, new_dst = [], []
    for v in outlier_idx:
        candidates = np.setdiff1d(np.arange(data.num_nodes), np.array([v]))
        others = np.random.choice(candidates, size=m, replace=False)
        new_src.extend([v] * m)
        new_dst.extend(others.tolist())

    if len(new_src) > 0:
        add_e = torch.tensor([new_src, new_dst], dtype=torch.long)
        data.edge_index = torch.cat([data.edge_index, add_e], dim=1)

    # 라벨
    y_outlier = torch.zeros(data.num_nodes, dtype=torch.long)
    y_outlier[torch.tensor(outlier_idx, dtype=torch.long)] = 1
    return data, y_outlier

