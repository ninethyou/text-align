# main.py
import argparse
import torch
from src.train import train_real_datasets

def get_args():
    p = argparse.ArgumentParser(description="GAD-NR runner")
    p.add_argument('--dataset', type=str, default="inj_cora")
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--epoch_num', type=int, default=500)
    p.add_argument('--lambda_loss1', type=float, default=1e-2)
    p.add_argument('--lambda_loss2', type=float, default=0.5)
    p.add_argument('--lambda_loss3', type=float, default=0.8)
    p.add_argument('--sample_size', type=int, default=10)
    p.add_argument('--dimension', type=int, default=128)
    p.add_argument('--encoder', type=str, default="GCN", choices=["GCN","GIN","GAT","SAGE"])
    p.add_argument('--loss_step', type=int, default=30)
    p.add_argument('--real_loss', action='store_true')
    p.add_argument('--neigh_loss', type=str, default="KL", choices=["KL","W2"])
    p.add_argument('--h_loss_weight', type=float, default=1.0)
    p.add_argument('--feature_loss_weight', type=float, default=2.0)
    p.add_argument('--degree_loss_weight', type=float, default=1.0)
    p.add_argument('--calculate_contextual', type=bool, default=True)
    p.add_argument('--contextual_n', type=int, default=70)
    p.add_argument('--contextual_k', type=int, default=10)
    p.add_argument('--calculate_structural', type=bool, default=True)
    p.add_argument('--structural_n', type=int, default=70)
    p.add_argument('--structural_m', type=int, default=10)
    p.add_argument('--use_combine_outlier', type=bool, default=False)
    p.add_argument('--save_dir_root', type=str, default="results")
    p.add_argument('--device',type=str,default="cuda",choices=["cpu", "cuda"], help="Select device: cpu or cuda")
    return p.parse_args()

def get_device(device_str: str) -> torch.device:
    """문자열을 torch.device로 안전하게 변환"""
    # 사용자가 cuda만 넣으면 cuda:0으로 치환
    if device_str == "cuda":
        device_str = "cuda:0"

    # cuda 요청인데 GPU가 없는 경우 → cpu로 fallback
    if "cuda" in device_str and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Using CPU instead.")
        return torch.device("cpu")

    return torch.device(device_str)



def main():
    args = get_args()
    device = get_device(args.device)
    print("GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction")
    print("Dataset: ", args.dataset, "lr:", args.lr, "lambda_loss1 (neighbor):",args.lambda_loss1, "lambda_loss2 (feature):", args.lambda_loss2, "lambda_loss3 (degree):", args.lambda_loss3, "sample_size:", args.sample_size, "dimension:",args.dimension, "encoder:", args.encoder, "loss_step:", args.loss_step,"real_loss:", args.real_loss, "h_loss_weight:",args.h_loss_weight,"feature_loss_weight",args.feature_loss_weight,"degree_loss_weight:",args.degree_loss_weight,
    "calculate_contextual",args.calculate_contextual,"calculate_structural",args.calculate_structural)

    result, save_dir = train_real_datasets(
        dataset_str=args.dataset,
        lr=args.lr,
        epoch_num=args.epoch_num, 
        lambda_loss1=args.lambda_loss1, 
        lambda_loss2=args.lambda_loss2, 
        lambda_loss3=args.lambda_loss3, 
        encoder=args.encoder, 
        sample_size=args.sample_size, 
        loss_step=args.loss_step, 
        hidden_dim=args.dimension,
        real_loss=args.real_loss,
        calculate_contextual=args.calculate_contextual,
        calculate_structural=args.calculate_structural,
        structural_n = args.structural_n,
        structural_m = args.structural_m,
        neigh_loss = args.neigh_loss,
        contextual_n = args.contextual_n,
        contextual_k = args.contextual_k,
        h_loss_weight = args.h_loss_weight,
        degree_loss_weight = args.degree_loss_weight,
        feature_loss_weight = args.feature_loss_weight,
        use_combine_outlier = args.use_combine_outlier,
        device = device,
        hiddens_dim=args.dimension,
        )


    print(f"[DONE] Saved results to: {save_dir}")
    print(result)

if __name__ == "__main__":
    main()
