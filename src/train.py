import torch
# src/train.py
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# === 외부 모듈(당신 노트북 코드 그대로 복붙) ===
from torch_geometric.utils import add_self_loops as pyg_add_self_loops
from pygod.utils import load_data
from pygod.generator import gen_contextual_outliers, gen_structural_outliers
from pygod.metrics import eval_roc_auc

# === 우리 프로젝트 모듈 ===
from src.utils import  gen_joint_structural_outliers  # utils.py에 만든 것
from src.losses import KL_neighbor_loss  # 필요 시 import (원본 쓰면 그대로)
from src.models.gad_nr import GNNStructEncoder

import numpy as np

def _to_1d_numpy(x):
    t = torch.as_tensor(x)
    t = t.reshape(-1)                 # 1D 보장
    return t.detach().cpu().numpy()

def safe_auc(y_true, y_score):
    y_true = _to_1d_numpy(y_true)
    y_score = _to_1d_numpy(y_score)
    ok = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true, y_score = y_true[ok], y_score[ok]
    if y_true.size == 0 or y_score.size == 0:
        raise ValueError(f"Empty inputs: y_true={y_true.size}, y_score={y_score.size}")
    cls = np.unique(y_true)
    if cls.size < 2:
        # 한 클래스만 존재 → 이 스텝은 스킵
        return np.nan
    return eval_roc_auc(y_true, y_score) * 100



def train(data,dataset_str, 
          y, yc, ys, yj, ysj, 
          lr, epoch, device, encoder,
          lambda_loss1, lambda_loss2, lambda_loss3, 
          hidden_dim, sample_size=10,
          loss_step=20,real_loss=False, neigh_loss = 'KL', 
          calculate_contextual=False,calculate_structural=False, h_loss_weight = 1.0, feature_loss_weight = 2.0, degree_loss_weight = 1.0 ):
    '''
     Main training function
     INPUT:
     -----------------------
     data : torch geometric dataset object
     lr    :    learning rate
     epoch     :    number of training epoch
     device     :   CPU or GPU
     encoder    :    GCN or GIN or GraphSAGE
     lambda_loss    :   Trade-off between degree loss and neighborhood reconstruction loss
     hidden_dim     :   latent variable dimension
    '''
    
    
    in_nodes = data.edge_index[0,:]
    out_nodes = data.edge_index[1,:]
    
    
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    # neighbor_num_list = []
    # for i in neighbor_dict:
    #     neighbor_num_list.append(len(neighbor_dict[i]))
    
    # neighbor_num_list = torch.tensor(neighbor_num_list).to(device)

    # train() 안
    in_nodes = data.edge_index[0]
    # 모든 노드 길이로 차수 카운트 (누락 0 자동 패딩)
    neighbor_num_list = torch.bincount(in_nodes, minlength=data.num_nodes).to(device)
        

    # neighbor_num_list = []
    # for i in neighbor_dict:
    #     neighbor_num_list.append(len(neighbor_dict[i]))
    
    # neighbor_num_list = torch.tensor(neighbor_num_list).to(device)
    
    in_dim = data.x.shape[1]
    GNNModel = GNNStructEncoder(in_dim, hidden_dim, hidden_dim, 2, sample_size, device=device, 
                    neighbor_num_list=neighbor_num_list, GNN_name=encoder, neigh_loss= neigh_loss,
                    lambda_loss1=lambda_loss1, lambda_loss2=lambda_loss2,lambda_loss3=lambda_loss3)
    GNNModel.to(device)
    degree_params = list(map(id, GNNModel.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params,
                         GNNModel.parameters())

    opt = torch.optim.Adam([{'params': base_params}, {'params': GNNModel.degree_decoder.parameters(), 'lr': 1e-2}],lr=lr, weight_decay=0.0003)
    min_loss = float('inf')
    arg_min_loss_per_node = None
    
    best_auc = 0
    best_auc_contextual = 0
    best_auc_dense_structural = 0
    best_auc_joint_structural = 0
    best_auc_structure_type = 0
    
        
    loss_values = []
    for i in tqdm(range(epoch)):
        
        if i%loss_step==0:
            GNNModel.lambda_loss2 = GNNModel.lambda_loss2 + 0.5
            GNNModel.lambda_loss3 = GNNModel.lambda_loss3 / 2
        
        loss,loss_per_node,h_loss,degree_loss,feature_loss = GNNModel(data.edge_index, data.x, neighbor_num_list, neighbor_dict, device=device)
        
        
        
        loss_per_node = loss_per_node.cpu().detach()
        
        h_loss = h_loss.cpu().detach()
        degree_loss = degree_loss.cpu().detach()
        feature_loss = feature_loss.cpu().detach()
        
        h_loss_norm = h_loss / (torch.max(h_loss) - torch.min(h_loss))
        degree_loss_norm = degree_loss / (torch.max(degree_loss) - torch.min(degree_loss))
        feature_loss_norm = feature_loss / (torch.max(feature_loss) - torch.min(feature_loss))
        
        comb_loss = h_loss_weight * h_loss_norm + degree_loss_weight *  degree_loss_norm + feature_loss_weight * feature_loss_norm
        
        if real_loss:
            comp_loss = loss_per_node
        else:
            comp_loss = comb_loss
            
        
        auc_score = safe_auc(y.cpu().detach().tolist(), comp_loss.cpu().detach().tolist()) * 100
        print("Dataset Name: ",dataset_str, ", AUC Score(benchmark/combined): ", auc_score)
        
        contextual_auc_score = safe_auc(yc.cpu().detach().tolist(), comp_loss.cpu().detach().tolist()) * 100
        print("Dataset Name: ",dataset_str, ", AUC Score (contextual): ", contextual_auc_score)


        def to_tensor(x):
            return torch.as_tensor(x) if isinstance(x, list) else x

        ys = to_tensor(ys)
        comp_loss = to_tensor(comp_loss)

        dense_structural_auc_score = safe_auc(ys.cpu().detach().tolist(), comp_loss.cpu().detach().tolist()) * 100
        print("Dataset Name: ",dataset_str, ", AUC Score (structural): ", dense_structural_auc_score)
        
        joint_structural_auc_score = safe_auc(yj.cpu().detach().tolist(), comp_loss.cpu().detach().tolist()) * 100
        print("Dataset Name: ",dataset_str, ", AUC Score (joint-type): ", joint_structural_auc_score)
        
        structure_type_auc_score = safe_auc(ysj.cpu().detach().tolist(), comp_loss.cpu().detach().tolist()) * 100
        print("Dataset Name: ",dataset_str, ", AUC Score (structure type): ", joint_structural_auc_score) 
        
        best_auc = max(best_auc, auc_score)
        best_auc_contextual = max(best_auc_contextual, contextual_auc_score)
        best_auc_dense_structural = max(best_auc_dense_structural, dense_structural_auc_score)
        best_auc_joint_structural = max(best_auc_joint_structural, joint_structural_auc_score)
        best_auc_structure_type = max(best_auc_structure_type, structure_type_auc_score)
        
        
        
        print("===========================================================================================")
        print("Dataset Name: ",dataset_str, " Best AUC Score(benchmark/combined): ", best_auc)
        
        contextual_auc_score = safe_auc(yc.cpu().detach().tolist(), comp_loss.cpu().detach().tolist()) * 100
        print("Dataset Name: ",dataset_str, " Best AUC Score (contextual): ", best_auc_contextual)

        dense_structural_auc_score = safe_auc(ys.cpu().detach().tolist(), comp_loss.cpu().detach().tolist()) * 100
        print("Dataset Name: ",dataset_str, " Best AUC Score (structural): ", best_auc_dense_structural)
        
        joint_structural_auc_score = safe_auc(yj.cpu().detach().tolist(), comp_loss.cpu().detach().tolist()) * 100
        print("Dataset Name: ",dataset_str, " Best AUC Score (joint-type): ", best_auc_joint_structural)
        
        structure_type_auc_score = safe_auc(ysj.cpu().detach().tolist(), comp_loss.cpu().detach().tolist()) * 100
        print("Dataset Name: ",dataset_str, " Best AUC Score (structure type): ", best_auc_structure_type) 
        print("===========================================================================================")
        
        
        if loss < min_loss:
            min_loss = loss
            arg_min_loss_per_node = loss_per_node
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        loss = loss.cpu().detach()
        loss_values.append(loss)

        return_value = {
            "loss": min_loss.item(),
            "loss_per_node": arg_min_loss_per_node.cpu().detach(),
            "best_auc": best_auc,
            "best_auc_contextual": best_auc_contextual,
            "best_auc_dense_structural": best_auc_dense_structural,
            "best_auc_joint_type": best_auc_joint_structural,
            "best_auc_structure_type": best_auc_structure_type,
            "loss_values": loss_values
        }
        
    
    return min_loss.item(), arg_min_loss_per_node.cpu().detach(), return_value



def evaluate(model, embeddings, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(embeddings)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    

def train_real_datasets(dataset_str, 
                        epoch_num = 10, lr = 5e-6, encoder = "GCN", 
                        lambda_loss1=1e-2, lambda_loss2=1e-3, lambda_loss3=1e-3, 
                        sample_size=8, loss_step=20, hiddens_dim=None,
                        real_loss=False,calculate_contextual=False,calculate_structural=False, 
                        structural_n = 70,contextual_n =70, 
                        contextual_k = 10,structural_m = 10, 
                        neigh_loss='KL', use_combine_outlier = False,
                        device = 'cpu', hidden_dim = 128,
                        h_loss_weight = 1.0, feature_loss_weight = 2.0, degree_loss_weight = 1.0 
                        ):
    
    data = load_data(dataset_str)
    node_features = data.x
    node_features_min = node_features.min()
    node_features_max = node_features.max()
    node_features = (node_features - node_features_min)/node_features_max
    data.x = node_features
    
    yc = []
    ys = []
    yj = []
    
    if calculate_contextual:
        
        if dataset_str == "inj_cora":
            yc = data.y >> 0 & 1 # contextual outliers
        else:
            data, yc = gen_contextual_outliers(data=data,n=contextual_n,k=contextual_k)
            
        yc = yc.cpu().detach()
    
    
    if calculate_structural:
        
        if dataset_str == "inj_cora":
            ys = data.y >> 1 & 1 # structural outliers
        else:
            data, ys = gen_structural_outliers(data=data,n=structural_n,m=structural_m,p=0.2)
            
        ys = ys.cpu().detach()
        data, yj = gen_joint_structural_outliers(data=data,n=structural_n,m=structural_m)
        
    
    if use_combine_outlier:
        data.y = torch.logical_or(ys, yc).int()
        
    # ysj = torch.logical_or(ys, yj).int()
    ysj = torch.logical_or(torch.as_tensor(ys), torch.as_tensor(yj)).int()

    y = data.y.bool()    # binary labels (inlier/outlier)
    y = y.cpu().detach()
    
    edge_index = data.edge_index.cpu()
    
    num_nodes = node_features.shape[0]
    self_edges = torch.tensor([[i for i in range(num_nodes)],[i for i in range(num_nodes)]])
    edge_index = torch.cat([edge_index,self_edges],dim=1)
    data.edge_index = edge_index
    data = data.to(device)
    

    loss, loss_per_node, result = train(data,dataset_str, 
                                        y, yc, ys, yj, ysj,
                                        lr=lr, epoch=epoch_num, device=device, encoder=encoder, 
                                        lambda_loss1=lambda_loss1, lambda_loss2=lambda_loss2, lambda_loss3=lambda_loss3, 
                                        hidden_dim=hidden_dim, sample_size=sample_size,
                                        loss_step=loss_step, real_loss=real_loss, neigh_loss=neigh_loss,
                                        calculate_contextual=calculate_contextual,calculate_structural=calculate_structural,
                                        h_loss_weight=h_loss_weight, feature_loss_weight=feature_loss_weight, degree_loss_weight=degree_loss_weight 
                                        )
    

    # 시간 기반 디렉토리 생성
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"results/{dataset_str}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 저장
    with open(f"{save_dir}/result.txt", "w") as f:
        f.write(f"Dataset: {dataset_str}\n")
        f.write(f"Best AUC (benchmark): {result['best_auc']:.2f}\n")
        f.write(f"Contextual AUC: {result['best_auc_contextual']:.2f}\n")
        f.write(f"Structural AUC: {result['best_auc_dense_structural']:.2f}\n")
        f.write(f"Joint-Type AUC: {result['best_auc_joint_type']:.2f}\n")
        f.write(f"Structure-Type AUC: {result['best_auc_structure_type']:.2f}\n")
        f.write(f"Final Loss: {result['loss']:.4f}\n")

    return result, save_dir
