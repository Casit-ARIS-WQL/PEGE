import argparse
import os
import torch
import numpy as np
from models.new_model import build
from util.misc import NestedTensor
from data_utils.dataset import PoETDataset
from torch.utils.data import DataLoader

# ========== 配置参数 ==========
def get_args():
    parser = argparse.ArgumentParser(description='PoET Training Script')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_path', default='output', help='Model save path')
    # Transformer
    parser.add_argument('--enc_layers', default=1, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=128, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=64, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.2, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=2, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--dec_n_points', default=3, type=int)
    parser.add_argument('--enc_n_points', default=3, type=int)
    # Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone_cfg', default='configs/uav_rcnn.yaml', type=str, help="Path to the backbone config file to use")
    parser.add_argument('--backbone_weights', default=None, type=str, help="Path to the pretrained weights for the backbone. None if no weights should be loaded.")
    parser.add_argument('--backbone', default='maskrcnn')
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--num_feature_levels', type=int, default=4)
    # Misc
    parser.add_argument('--bbox_mode', default='gt', choices=('gt', 'jitter'))
    parser.add_argument('--reference_points', default='bbox')
    parser.add_argument('--query_embedding', default='bbox')
    parser.add_argument('--rotation_representation', default='6d')
    parser.add_argument('--class_mode', default='agnostic')
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--translation_loss_coef', type=float, default=1.0)
    parser.add_argument('--rotation_loss_coef', type=float, default=1.0)
    parser.add_argument('--optimization_loss_coef', type=float, default=0.1)
    # Matcher
    parser.add_argument('--matcher_type', default='pose', choices=['pose'], type=str)
    parser.add_argument('--set_cost_bbox', default=1, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=1, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--enable_tensor_opt', action='store_true')
    parser.add_argument('--tensor_rank', type=int, default=64)
    parser.add_argument('--enable_topology_opt', action='store_true')
    parser.add_argument('--enable_riemannian_refine', action='store_true')
    parser.add_argument('--riemann_steps', type=int, default=3)
    parser.add_argument('--riemann_step_scale', type=float, default=0.5)
    parser.add_argument('--riemann_refiner_type', default='learnable')
    return parser.parse_args()

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    if targets is not None:
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets

# ========== 主训练流程 ==========
def train():
    args = get_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 构建模型
    model, criterion, matcher = build(args)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ====== 数据集路径配置 ======
    images_dir = os.path.join('data', 'train', 'images')
    ann_path = os.path.join('data', 'train', 'annotations', 'train.json')
    # 支持jitter参数
    dataset = PoETDataset(images_dir, ann_path, device=device, jitter=(args.bbox_mode=='jitter'))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    # dataloader输出的bbox为cxcywh格式
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        step = 0
        for images, targets in dataloader:
            images, targets = to_cuda(images, targets, device)
            optimizer.zero_grad()
            outputs, n_boxes = model(images, targets)
            loss_dict = criterion(outputs, targets, n_boxes)
            total_loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict.keys())
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            step += 1
            print(f"Epoch {epoch+1}/{args.epochs} | Step {step} | Loss: {total_loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"============Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f}============")

        # 保存模型
        torch.save(model.state_dict(), args.save_path)
        output_path = os.path.join(args.save_path, f"checkpoint{epoch+1:d}")
        print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train()
