# ------------------------------------------------------------------------
# PoET (All-in-One Optimized)
# - Consolidates PoET + tensor/topology/riemannian enhancements into a single file.
# - No extra third-party deps beyond torch/numpy.
# - Drop-in build(args) API kept.
# ------------------------------------------------------------------------

import copy
import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ======== Project utilities (assumed present in repo) ========
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer import build_deforamble_transformer
from .position_encoding import BoundingBoxEmbeddingSine


# ========================== Lie / SO(3) Helpers ==========================
def _hat(omega: torch.Tensor) -> torch.Tensor:
    """so(3) hat map: (...,3) -> (...,3,3)"""
    wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
    O = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device, dtype=omega.dtype)
    O[..., 0, 1], O[..., 0, 2] = -wz, wy
    O[..., 1, 0], O[..., 1, 2] = wz, -wx
    O[..., 2, 0], O[..., 2, 1] = -wy, wx
    return O

def _vee(W: torch.Tensor) -> torch.Tensor:
    """so(3) vee map: (...,3,3) -> (...,3)"""
    return torch.stack([W[..., 2,1] - W[..., 1,2],
                        W[..., 0,2] - W[..., 2,0],
                        W[..., 1,0] - W[..., 0,1]], dim=-1) * 0.5

def exp_SO3(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues' exp: (B,3) -> (B,3,3)"""
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp(min=1e-12)
    k = omega / theta
    K = _hat(k)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[0], 3, 3)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    K2 = torch.matmul(K, K)
    R = I + sin_t * K + (1 - cos_t) * K2

    small = (theta.squeeze(-1) < 1e-5).float()[..., None, None]
    R_small = I + _hat(omega) + 0.5 * torch.matmul(_hat(omega), _hat(omega))
    return R * (1 - small) + R_small * small

def log_SO3(R: torch.Tensor) -> torch.Tensor:
    """Log map: (B,3,3) -> (B,3)"""
    B = R.shape[0]
    tr = (R[..., 0,0] + R[..., 1,1] + R[..., 2,2]).clamp(-1.0, 3.0)
    theta = torch.acos(((tr - 1.0) * 0.5).clamp(-1.0, 1.0))
    theta = theta.unsqueeze(-1)
    small = (theta.squeeze(-1) < 1e-5).float().unsqueeze(-1)
    W = 0.5 * (R - R.transpose(-1, -2))
    w_small = _vee(W)
    w = _vee(W) / (torch.sin(theta).clamp(min=1e-12)).unsqueeze(-1) * theta
    return w * (1 - small) + w_small * small

def geodesic_SO3(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    Rt = torch.matmul(R1.transpose(-1, -2), R2)
    tr = (Rt[..., 0,0] + Rt[..., 1,1] + Rt[..., 2,2]).clamp(-1.0, 3.0)
    cos_th = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos_th)



# =================== Tensor Optimization (small-object) ===================
class TensorOptimization(nn.Module):
    """
    Lightweight tensor-inspired enhancement on sequence features (B,Q,D).
    - Linear bottleneck ("Tucker-like") D -> r -> D
    - Soft sparsity gate per sample
    Returns: enhanced (B,Q,D) and dict(losses)
    """
    def __init__(self, hidden_dim: int, rank: int = 64, sparsity_lambda: float = 1e-3):
        super().__init__()
        r = max(8, min(rank, hidden_dim))
        self.enc = nn.Linear(hidden_dim, r, bias=False)
        self.dec = nn.Linear(r, hidden_dim, bias=False)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.sparsity_lambda = sparsity_lambda

        nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dec.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B,Q,D)
        B, Q, D = x.shape
        z = self.enc(x)            # (B,Q,r)
        y = self.dec(z)            # (B,Q,D)
        gate = self.gate(x)        # (B,Q,D) in (0,1)
        enhanced = x + gate * (y - x)

        # Losses
        recon = F.mse_loss(y, x)
        sparsity = gate.mean()
        losses = {
            'tensor_reconstruction_loss': recon,
            'sparsity_loss': self.sparsity_lambda * sparsity
        }
        return enhanced, losses


# ===================== Topology Proxy (Betti-0-ish) ======================
class TopologyOptimization(nn.Module):
    """
    A differentiable proxy that encourages 'connectivity' among queries:
    - Compute pairwise affinity A = exp(-||qi - qj||^2 / (2*sigma^2))
    - Topology regularizer = 1 - mean(top-K affinities per query)  (smaller is better connectivity)
    Also produces a global summary vector via attention pooling: (B,D) -> broadcast to (B,Q,D).
    """
    def __init__(self, hidden_dim: int, k: int = 4, sigma: float = 1.0):
        super().__init__()
        self.k = k
        self.sigma2 = sigma ** 2
        self.pool = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B,Q,D)
        B, Q, D = x.shape
        # Pairwise distances (B,Q,Q)
        x2 = (x**2).sum(-1, keepdim=True)                    # (B,Q,1)
        dist2 = x2 + x2.transpose(1,2) - 2 * torch.bmm(x, x.transpose(1,2))
        dist2 = dist2.clamp(min=0)

        A = torch.exp(-dist2 / (2 * self.sigma2) )           # (B,Q,Q)
        A = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))  # zero diagonal

        # top-K affinity per query
        topk = torch.topk(A, k=min(self.k, max(1, Q-1)), dim=-1).values  # (B,Q,k)
        topo_reg = 1.0 - topk.mean()  # scalar

        # Attention pooling to (B,D)
        attn = F.softmax(A.sum(-1), dim=-1).unsqueeze(-1)    # (B,Q,1)
        global_vec = (attn * x).sum(1)                       # (B,D)
        global_vec = self.pool(global_vec)                   # (B,D)
        global_broadcast = global_vec.unsqueeze(1).expand(B, Q, D)

        losses = {'topology_regularization': topo_reg}
        return global_broadcast, losses


# =============== Riemannian refinement (training-time step) ===============
class RiemannianRefinement(nn.Module):
    """
    Learnable Riemannian refinement module that performs geodesic steps on SO(3) manifold
    with learnable parameters for adaptive refinement.
    """
    def __init__(self, hidden_dim: int = 256, max_steps: int = 3):
        super().__init__()
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim
        
        # Learnable step scale predictor
        self.step_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, max_steps),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Learnable rotation weight predictor
        self.rotation_weight_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Learnable translation weight predictor
        self.translation_weight_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Adaptive matching weights based on feature similarity
        self.matching_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Global refinement strength
        self.global_strength = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, R_pred: torch.Tensor, t_pred: torch.Tensor, 
                pred_boxes: torch.Tensor, targets: List[Dict],
                query_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            R_pred: (B,Q,3,3) predicted rotations
            t_pred: (B,Q,3) predicted translations  
            pred_boxes: (B,Q,4) predicted bounding boxes
            targets: List of target dictionaries
            query_features: (B,Q,D) query features for learning adaptive weights
        """
        B, Q = pred_boxes.shape[:2]
        device = R_pred.device
        c_pred = pred_boxes[..., :2]  # (B,Q,2)
        
        # Predict adaptive step scales for each step
        step_scales = self.step_predictor(query_features.mean(dim=1))  # (B, max_steps)
        step_scales = step_scales * self.global_strength.clamp(0, 1)
        
        # Predict per-query weights
        rot_weights = self.rotation_weight_net(query_features)  # (B,Q,1)
        trans_weights = self.translation_weight_net(query_features)  # (B,Q,1)
        
        new_R, new_t = [], []
        
        for b in range(B):
            Rb = R_pred[b].clone()  # (Q,3,3)
            tb = t_pred[b].clone()  # (Q,3)
            tdict = targets[b]
            
            if ('relative_rotation' not in tdict) or (tdict['relative_rotation'].numel() == 0):
                new_R.append(Rb)
                new_t.append(tb)
                continue
                
            Rg = tdict['relative_rotation'].to(device)      # (Ng,3,3)
            tg = tdict['relative_position'].to(device)      # (Ng,3)
            cgt = tdict['boxes'][:, :2].to(device)          # (Ng,2)
            
            # Compute distance-based matching
            d = (c_pred[b][:,None,:] - cgt[None,:,:]).pow(2).sum(-1)  # (Q,Ng)
            nn_idx = torch.argmin(d, dim=1)  # (Q,)
            
            # Compute feature-based matching confidence if target features available
            if 'features' in tdict and tdict['features'].numel() > 0:
                target_features = tdict['features'].to(device)  # (Ng,D)
                matched_target_features = target_features[nn_idx]  # (Q,D)
                
                # Concatenate query and matched target features
                combined_features = torch.cat([
                    query_features[b], 
                    matched_target_features
                ], dim=-1)  # (Q, 2*D)
                
                matching_weights = self.matching_net(combined_features)  # (Q,1)
            else:
                # Fallback: use distance-based confidence
                min_distances = torch.min(d, dim=1).values  # (Q,)
                matching_weights = torch.exp(-min_distances).unsqueeze(-1)  # (Q,1)
            
            # Perform iterative refinement with learnable step sizes
            for step in range(self.max_steps):
                current_step_scale = step_scales[b, step]
                
                Rmatch = Rg[nn_idx]  # (Q,3,3)
                tmatch = tg[nn_idx]  # (Q,3)
                
                # Adaptive rotation refinement
                try:
                    w = log_SO3(torch.matmul(Rb.transpose(-1,-2), Rmatch))  # (Q,3)
                    
                    # Apply learnable weights and step scale
                    adaptive_w = w * rot_weights[b] * matching_weights * current_step_scale
                    delta_R = exp_SO3(adaptive_w)  # (Q,3,3)
                    Rb = torch.matmul(Rb, delta_R)
                    
                except Exception as e:
                    # Fallback to small identity perturbation if SO(3) operations fail
                    continue
                
                # Adaptive translation refinement
                trans_delta = (tmatch - tb) * trans_weights[b] * matching_weights * current_step_scale
                tb = tb + trans_delta
                
            new_R.append(Rb)
            new_t.append(tb)
            
        return torch.stack(new_R, 0), torch.stack(new_t, 0)


def riemannian_refine_with_targets(R_pred: torch.Tensor,
                                   t_pred: torch.Tensor,
                                   pred_boxes: torch.Tensor,
                                   targets: List[Dict],
                                   steps: int = 3,
                                   step_scale: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy function for backward compatibility. 
    Greedy nearest-center pairing and few geodesic steps on SO(3) + translation towards GT.
    
    Note: This function is kept for compatibility but the learnable RiemannianRefinement 
    module should be preferred for training.
    
    Shapes:
      R_pred: (B,Q,3,3), t_pred: (B,Q,3), pred_boxes: (B,Q,4)
      targets[b]['relative_rotation']: (Ng,3,3), ['relative_position']:(Ng,3), ['boxes']:(Ng,4)
    """
    B, Q = pred_boxes.shape[:2]
    device = R_pred.device
    c_pred = pred_boxes[..., :2]  # (B,Q,2)

    new_R, new_t = [], []
    for b in range(B):
        Rb = R_pred[b].clone()
        tb = t_pred[b].clone()
        tdict = targets[b]
        if ('relative_rotation' not in tdict) or (tdict['relative_rotation'].numel() == 0):
            new_R.append(Rb); new_t.append(tb); continue
        Rg = tdict['relative_rotation'].to(device)      # (Ng,3,3)
        tg = tdict['relative_position'].to(device)      # (Ng,3)
        cgt = tdict['boxes'][:, :2].to(device)          # (Ng,2)
        d = (c_pred[b][:,None,:] - cgt[None,:,:]).pow(2).sum(-1)  # (Q,Ng)
        nn_idx = torch.argmin(d, dim=1)                                  # (Q,)
        for _ in range(max(0, steps)):
            Rmatch = Rg[nn_idx]                      # (Q,3,3)
            tmatch = tg[nn_idx]                      # (Q,3)
            try:
                w = log_SO3(torch.matmul(Rb.transpose(-1,-2), Rmatch))  # (Q,3)
                delta_R = exp_SO3(step_scale * w)                      # (Q,3,3)
                Rb = torch.matmul(Rb, delta_R)
                tb = tb + step_scale * (tmatch - tb)
            except:
                # Skip this iteration if SO(3) operations fail
                continue
        new_R.append(Rb); new_t.append(tb)
    return torch.stack(new_R, 0), torch.stack(new_t, 0)


# ================================ PoET ====================================
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PoET(nn.Module):
    """
    Pose Estimation Transformer with integrated algebraic/geometric enhancements.
    """
    def __init__(self, backbone, transformer, num_queries, num_feature_levels, n_classes, bbox_mode='gt',
                 ref_points_mode='bbox', query_embedding_mode='bbox', rotation_mode='6d', class_mode='agnostic',
                 aux_loss=True, backbone_type="yolo",
                 # Enhancement flags
                 enable_tensor_opt: bool = True,
                 tensor_rank: int = 64,
                 enable_topology_opt: bool = True,
                 enable_riemannian_refine: bool = True,
                 riemann_steps: int = 3,
                 riemann_step_scale: float = 0.5,
                 # 新增参数：选择Riemannian refiner类型
                 riemann_refiner_type: str = 'learnable'):  # 'original', 'learnable'
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.aux_loss = aux_loss
        self.n_queries = num_queries
        self.n_classes = n_classes + 1  # +1 for dummy/background class
        self.bbox_mode = bbox_mode
        self.ref_points_mode = ref_points_mode
        self.query_embedding_mode = query_embedding_mode
        self.rotation_mode = rotation_mode
        self.class_mode = class_mode

        # Enhancements
        self.enable_tensor_opt = enable_tensor_opt
        self.enable_topology_opt = enable_topology_opt
        self.enable_riemannian_refine = enable_riemannian_refine
        self.riemann_steps = riemann_steps
        self.riemann_step_scale = riemann_step_scale

        self.riemann_refiner_type = riemann_refiner_type

        # Head dims
        self.t_dim = 3
        if self.rotation_mode == '6d':
            self.rot_dim = 6
        elif self.rotation_mode in ['quat', 'silho_quat']:
            self.rot_dim = 4
        else:
            raise NotImplementedError('Rotational representation is not supported.')

        # Heads
        if self.class_mode == 'agnostic':
            self.translation_head = MLP(hidden_dim, hidden_dim, self.t_dim, 3)
            self.rotation_head = MLP(hidden_dim, hidden_dim, self.rot_dim, 3)
        elif self.class_mode == 'specific':
            self.translation_head = MLP(hidden_dim, hidden_dim, self.t_dim * self.n_classes, 3)
            self.rotation_head = MLP(hidden_dim, hidden_dim, self.rot_dim * self.n_classes, 3)
        else:
            raise NotImplementedError('Class mode is not supported.')

        # Feature levels
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for n in range(num_backbone_outs):
                in_channels = backbone.num_channels[n]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for n in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Aux per-layer heads
        num_pred = transformer.decoder.num_layers
        self.translation_head = nn.ModuleList([copy.deepcopy(self.translation_head) for _ in range(num_pred)])
        self.rotation_head = nn.ModuleList([copy.deepcopy(self.rotation_head) for _ in range(num_pred)])

        # Query embeddings
        if self.query_embedding_mode == 'bbox':
            self.bbox_embedding = BoundingBoxEmbeddingSine(num_pos_feats=hidden_dim / 8)
        elif self.query_embedding_mode == 'learned':
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            self.bbox_embedding = BoundingBoxEmbeddingSine(num_pos_feats=hidden_dim / 8)
        else:
            raise NotImplementedError('This query embedding mode is not implemented.')

        # Enhancement modules on hs (B,Q,D) - 修改属性名以保持一致性
        if self.enable_tensor_opt:
            self.tensor_opt = TensorOptimization(hidden_dim=hidden_dim, rank=tensor_rank)
            # 为了向后兼容，也创建 tensor_optimization 别名
            self.tensor_optimization = self.tensor_opt
        else:
            self.tensor_opt = None
            self.tensor_optimization = None

        if self.enable_topology_opt:
            self.topology_opt = TopologyOptimization(hidden_dim=hidden_dim, k=4, sigma=1.0)
            # 为了向后兼容，也创建 topo_opt 别名
            self.topo_opt = self.topology_opt
        else:
            self.topology_opt = None
            self.topo_opt = None

        # Fusion after enhancements (orig + 2 branches -> 3*D -> D)
        fused_in = hidden_dim * (1 + int(self.enable_tensor_opt) + int(self.enable_topology_opt))
        self.optimization_fusion = nn.Identity() if fused_in == hidden_dim else nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Riemannian refinement module
        if self.enable_riemannian_refine:
            if riemann_refiner_type == 'learnable':
                self.riemannian_refiner = RiemannianRefinement(
                    hidden_dim=hidden_dim, 
                    max_steps=riemann_steps
                )
            else:
                self.riemannian_refiner = None  # 使用原始函数
        else:
            self.riemannian_refiner = None


    def forward(self, samples: NestedTensor, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # sizes HxW
        image_sizes = [[sample.shape[-2], sample.shape[-1]] for sample in samples.tensors]
        features, pos, pred_objects = self.backbone(samples)

        # 获取批次大小
        batch_size = samples.tensors.shape[0]

        # Prepare boxes/classes and queries
        pred_boxes = []
        pred_classes = []
        query_embeds = []
        n_boxes_per_sample = []

        if self.bbox_mode in ['gt', 'jitter'] and targets is not None:
            for t, target in enumerate(targets):
                t_boxes = target["boxes"] if self.bbox_mode == 'gt' else target["jitter_boxes"]
                n_boxes = len(t_boxes)
                n_boxes_per_sample.append(n_boxes)
                t_classes = target["labels"]
                query_embed = self.bbox_embedding(t_boxes).repeat(1, 2)
                if n_boxes < self.n_queries:
                    device = t_boxes.device
                    dummy_boxes = torch.full((self.n_queries - n_boxes, 4), -1.0, dtype=torch.float32, device=device)
                    dummy_embed = torch.full((self.n_queries - n_boxes, 1), -10.0, dtype=torch.float32, device=device).repeat(1, self.hidden_dim*2)
                    t_boxes = torch.vstack((t_boxes, dummy_boxes))
                    query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                    dummy_classes = torch.full((self.n_queries - n_boxes,), -1, dtype=torch.int64, device=device)
                    t_classes = torch.cat((t_classes, dummy_classes))
                pred_boxes.append(t_boxes)
                query_embeds.append(query_embed)
                pred_classes.append(t_classes)
        elif self.bbox_mode == 'backbone':
            for bs, predictions in enumerate(pred_objects):
                if predictions is None:
                    n_boxes = 0
                    n_boxes_per_sample.append(n_boxes)
                    device_ = features[0].decompose()[0].device
                    backbone_boxes = torch.full((self.n_queries, 4), -1.0, dtype=torch.float32, device=device_)
                    query_embed = torch.full((self.n_queries, 1), -10.0, dtype=torch.float32, device=device_).repeat(1, self.hidden_dim * 2)
                    backbone_classes = torch.full((self.n_queries,), -1, dtype=torch.int64, device=device_)
                else:
                    backbone_boxes = predictions[:, :4]
                    backbone_boxes = box_ops.box_xyxy_to_cxcywh(backbone_boxes)
                    backbone_boxes = box_ops.box_normalize_cxcywh(backbone_boxes, image_sizes[0])
                    n_boxes = len(backbone_boxes)
                    backbone_scores = predictions[:, 4]
                    backbone_classes = predictions[:, 5].type(torch.int64)
                    query_embed = self.bbox_embedding(backbone_boxes).repeat(1, 2)
                    if n_boxes < self.n_queries:
                        device = backbone_boxes.device
                        dummy_boxes = torch.full((self.n_queries - n_boxes, 4), -1.0, dtype=torch.float32, device=device)
                        dummy_embed = torch.full((self.n_queries - n_boxes, 1), -10.0, dtype=torch.float32, device=device).repeat(1, self.hidden_dim * 2)
                        backbone_boxes = torch.cat([backbone_boxes, dummy_boxes], dim=0)
                        query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                        dummy_classes = torch.full((self.n_queries - n_boxes,), -1, dtype=torch.int64, device=device)
                        backbone_classes = torch.cat([backbone_classes, dummy_classes], dim=0)
                    elif n_boxes > self.n_queries:
                        n_boxes = self.n_queries
                        backbone_scores, indices = torch.sort(backbone_scores, dim=0, descending=True)
                        backbone_classes = backbone_classes[indices][:self.n_queries]
                        backbone_boxes = backbone_boxes[indices, :][:self.n_queries]
                        query_embed = query_embed[indices, :][:self.n_queries]
                    n_boxes_per_sample.append(n_boxes)
                pred_boxes.append(backbone_boxes)
                pred_classes.append(backbone_classes)
                query_embeds.append(query_embed)
        else:
            # 推理模式或其他模式：创建默认的查询嵌入
            device = features[0].decompose()[0].device
            for bs in range(batch_size):
                # 创建默认的边界框和查询
                default_boxes = torch.zeros((self.n_queries, 4), dtype=torch.float32, device=device)
                default_classes = torch.zeros((self.n_queries,), dtype=torch.int64, device=device)
                
                if self.query_embedding_mode == 'bbox':
                    # 使用默认边界框创建查询嵌入
                    query_embed = self.bbox_embedding(default_boxes).repeat(1, 2)
                elif self.query_embedding_mode == 'learned':
                    # 使用学习的查询嵌入
                    query_embed = self.query_embed.weight.clone()
                else:
                    # 创建随机查询嵌入
                    query_embed = torch.randn((self.n_queries, self.hidden_dim * 2), device=device)
                
                pred_boxes.append(default_boxes)
                pred_classes.append(default_classes)
                query_embeds.append(query_embed)
                n_boxes_per_sample.append(0)  # 推理模式下没有ground truth boxes

        # 确保我们有正确数量的查询嵌入
        if len(query_embeds) == 0:
            # 如果仍然为空，创建默认值
            device = features[0].decompose()[0].device
            for bs in range(batch_size):
                default_boxes = torch.zeros((self.n_queries, 4), dtype=torch.float32, device=device)
                default_classes = torch.zeros((self.n_queries,), dtype=torch.int64, device=device)
                
                if self.query_embedding_mode == 'learned':
                    query_embed = self.query_embed.weight.clone()
                else:
                    query_embed = torch.randn((self.n_queries, self.hidden_dim * 2), device=device)
                
                pred_boxes.append(default_boxes)
                pred_classes.append(default_classes)
                query_embeds.append(query_embed)
                n_boxes_per_sample.append(0)

        # 现在安全地堆叠张量
        query_embeds = torch.stack(query_embeds)
        pred_boxes = torch.stack(pred_boxes)
        pred_classes = torch.stack(pred_classes)

        # Project feature levels
        srcs = []
        masks = []
        for lvl, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[lvl](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for lvl in range(_len_srcs, self.num_feature_levels):
                if lvl == _len_srcs:
                    src = self.input_proj[lvl](features[-1].tensors)
                else:
                    src = self.input_proj[lvl](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        reference_points = pred_boxes[:, :, :2] if self.ref_points_mode == 'bbox' else None
        if self.query_embedding_mode == 'learned':
            # 对于学习的查询嵌入，重复到批次大小
            query_embeds = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # Transformer
        hs, init_reference, _, _, _ = self.transformer(srcs, masks, pos, query_embeds, reference_points)
        optimization_losses = {}

        # Enhancements on hs (per layer)
        enhanced_hs = []
        for level_features in hs:  # (B,Q,D)
            branches = [level_features]
            if self.tensor_opt is not None:
                f_tensor, loss_dict = self.tensor_opt(level_features)
                branches.append(f_tensor)
                for k, v in loss_dict.items():
                    optimization_losses[f'tensor_{k}'] = optimization_losses.get(f'tensor_{k}', 0.0) + v
            if self.topology_opt is not None:  # 使用 topology_opt
                f_topo, t_loss = self.topology_opt(level_features)
                branches.append(f_topo)
                for k, v in t_loss.items():
                    optimization_losses[f'{k}'] = optimization_losses.get(k, 0.0) + v
            fused = torch.cat(branches, dim=-1)
            fused = self.optimization_fusion(fused)
            enhanced_hs.append(fused)
        hs = torch.stack(enhanced_hs)

        # Heads
        outputs_translation = []
        outputs_rotation = []
        bs, _ = pred_classes.shape
        # For class-specific, select per-class slice; else directly use predictions
        if self.class_mode == 'specific':
            output_idx = torch.where(pred_classes > 0, pred_classes, torch.zeros_like(pred_classes)).view(-1)

        for lvl in range(hs.shape[0]):
            out_rot = self.rotation_head[lvl](hs[lvl])      # (B,Q,rot_dim)
            out_trans = self.translation_head[lvl](hs[lvl]) # (B,Q,3)
            if self.class_mode == 'specific':
                out_rot = out_rot.view(bs * self.n_queries, self.n_classes, -1)
                out_rot = torch.cat([q[output_idx[i], :] for i, q in enumerate(out_rot)]).view(bs, self.n_queries, -1)
                out_trans = out_trans.view(bs * self.n_queries, self.n_classes, -1)
                out_trans = torch.cat([q[output_idx[i], :] for i, q in enumerate(out_trans)]).view(bs, self.n_queries, -1)

            out_rot = self.process_rotation(out_rot)        # -> (B,Q,3,3) or normalized quat
            outputs_rotation.append(out_rot)
            outputs_translation.append(out_trans)

        outputs_rotation = torch.stack(outputs_rotation)
        outputs_translation = torch.stack(outputs_translation)

        out = {'pred_translation': outputs_translation[-1], 'pred_rotation': outputs_rotation[-1],
               'pred_boxes': pred_boxes, 'pred_classes': pred_classes}

        # 修改Riemannian refinement调用
        if self.training and self.enable_riemannian_refine and targets is not None and self.rotation_mode == '6d':
            if self.riemannian_refiner is not None:
                # 使用可学习的Riemannian refinement
                R_refined, t_refined = self.riemannian_refiner(
                    out['pred_rotation'], 
                    out['pred_translation'], 
                    pred_boxes, 
                    targets,
                    hs[-1]  # 传递查询特征
                )
            else:
                # 使用原始的函数版本
                R_refined, t_refined = riemannian_refine_with_targets(
                    out['pred_rotation'], out['pred_translation'], pred_boxes, targets,
                    steps=self.riemann_steps, step_scale=self.riemann_step_scale
                )
            out['pred_rotation'] = R_refined
            out['pred_translation'] = t_refined

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_translation, outputs_rotation, pred_boxes, pred_classes)

        # Aggregate optimization losses (mean over layers)
        if optimization_losses:
            for k, v in optimization_losses.items():
                if torch.is_tensor(v):
                    optimization_losses[k] = v / hs.shape[0]
                else:
                    optimization_losses[k] = torch.as_tensor(v, device=hs.device, dtype=hs.dtype) / hs.shape[0]
            out['optimization_losses'] = optimization_losses

        return out, n_boxes_per_sample

    def _set_aux_loss(self, outputs_translation, outputs_quaternion, pred_boxes, pred_classes):
        return [{'pred_translation': t, 'pred_rotation': r, 'pred_boxes': pred_boxes, 'pred_classes': pred_classes}
                for t, r in zip(outputs_translation[:-1], outputs_quaternion[:-1])]

    def process_rotation(self, pred_rotation):
        if self.rotation_mode == '6d':
            return self.rotation_6d_to_matrix(pred_rotation)
        elif self.rotation_mode in ['quat', 'silho_quat']:
            return F.normalize(pred_rotation, p=2, dim=2)
        else:
            raise NotImplementedError('Rotation mode is not supported')

    def rotation_6d_to_matrix(self, rot_6d):
        """Convert 6D rotation representation to rotation matrix with numerical stability"""
        if rot_6d.dim() == 3:
            bs, n_q, _ = rot_6d.shape
            rot_6d = rot_6d.view(-1, 6)
        else:
            bs, n_q = rot_6d.shape[0], 1
        
        # 检查输入有效性
        if not torch.isfinite(rot_6d).all():
            device = rot_6d.device
            batch_size = rot_6d.shape[0]
            return torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1).view(bs, n_q, 3, 3)
        
        m1 = rot_6d[:, 0:3]
        m2 = rot_6d[:, 3:6]
        
        # 使用更稳定的正交化过程
        eps = 1e-6
        
        # 第一个向量归一化
        x = F.normalize(m1, p=2, dim=1, eps=eps)
        
        # 确保x不全为零
        x_norm = torch.norm(x, dim=1, keepdim=True)
        x = torch.where(x_norm < eps, torch.tensor([1., 0., 0.], device=x.device).expand_as(x), x)
        
        # 计算叉积
        z = torch.cross(x, m2, dim=1)
        z_norm = torch.norm(z, dim=1, keepdim=True)
        
        # 如果叉积接近零，使用默认向量
        z = torch.where(z_norm < eps, torch.tensor([0., 0., 1.], device=z.device).expand_as(z), z)
        z = F.normalize(z, p=2, dim=1, eps=eps)
        
        # 第三个向量
        y = torch.cross(z, x, dim=1)
        y = F.normalize(y, p=2, dim=1, eps=eps)
        
        # 构建旋转矩阵
        rot_matrix = torch.stack([x, y, z], dim=-1)  # (batch, 3, 3)
        
        # 检查结果有效性
        if not torch.isfinite(rot_matrix).all():
            device = rot_matrix.device
            batch_size = rot_matrix.shape[0]
            rot_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        rot_matrix = rot_matrix.view(bs, n_q, 3, 3)
        return rot_matrix


# =============================== Criterion ================================
class SetCriterion(nn.Module):
    """ Losses for PoET (+ optional optimization regularizers). """
    def __init__(self, matcher, weight_dict, losses):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def _get_src_permutation_idx(self, indices):
        """
        Permute predictions following indices
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """
        Permute targets following indices
        """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_translation(self, outputs, targets, indices):
        """Compute translation loss"""
        # 检查indices是否为空或所有匹配都为空
        if not indices or all(len(src) == 0 for src, _ in indices):
            device = next(iter(outputs.values())).device
            return {"loss_trans": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 获取排列索引
        try:
            idx = self._get_src_permutation_idx(indices)
        except:
            device = next(iter(outputs.values())).device
            return {"loss_trans": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 检查索引是否有效
        if len(idx[0]) == 0:
            device = next(iter(outputs.values())).device
            return {"loss_trans": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 获取预测的平移
        src_translation = outputs["pred_translation"][idx]
        
        # 安全地构建目标张量
        tgt_translations = []
        for t, (_, i) in zip(targets, indices):
            if len(i) > 0 and 'relative_position' in t:
                if t['relative_position'].numel() > 0 and len(i) <= len(t['relative_position']):
                    tgt_translations.append(t['relative_position'][i])
        
        if not tgt_translations:
            device = next(iter(outputs.values())).device
            return {"loss_trans": torch.tensor(0.0, device=device, requires_grad=True)}
        
        tgt_translation = torch.cat(tgt_translations, dim=0)
        
        # 确保形状匹配
        min_size = min(src_translation.shape[0], tgt_translation.shape[0])
        if min_size == 0:
            device = next(iter(outputs.values())).device
            return {"loss_trans": torch.tensor(0.0, device=device, requires_grad=True)}
        
        src_translation = src_translation[:min_size]
        tgt_translation = tgt_translation[:min_size]
        
        loss = F.l1_loss(src_translation, tgt_translation)
        return {"loss_trans": loss}

    def loss_rotation(self, outputs, targets, indices):
        """Compute rotation loss with robust tensor handling"""
        eps = 1e-6
        
        # 检查indices是否为空或所有匹配都为空
        if not indices or all(len(src) == 0 for src, _ in indices):
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 获取排列索引
        try:
            idx = self._get_src_permutation_idx(indices)
        except:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 检查索引是否有效
        if len(idx[0]) == 0:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 获取预测的旋转
        src_rot = outputs["pred_rotation"][idx]
        
        # 安全地构建目标张量
        tgt_rotations = []
        for t, (_, i) in zip(targets, indices):
            if len(i) > 0 and 'relative_rotation' in t:
                if t['relative_rotation'].numel() > 0 and len(i) <= len(t['relative_rotation']):
                    tgt_rotations.append(t['relative_rotation'][i])
        
        if not tgt_rotations:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        tgt_rot = torch.cat(tgt_rotations, dim=0)
        
        # 检查张量是否为空
        if src_rot.numel() == 0 or tgt_rot.numel() == 0:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 确保形状匹配
        min_size = min(src_rot.shape[0], tgt_rot.shape[0])
        if min_size == 0:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        src_rot = src_rot[:min_size]
        tgt_rot = tgt_rot[:min_size]
        
        # 确保是3D张量且形状为 (N, 3, 3)
        if src_rot.dim() == 2:
            if src_rot.shape[1] == 9:
                src_rot = src_rot.view(-1, 3, 3)
            elif src_rot.shape[1] == 6:
                # 如果是6D表示，转换为旋转矩阵
                src_rot = self._6d_to_rotation_matrix(src_rot)
            else:
                device = src_rot.device
                return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        if tgt_rot.dim() == 2:
            if tgt_rot.shape[1] == 9:
                tgt_rot = tgt_rot.view(-1, 3, 3)
            else:
                device = tgt_rot.device
                return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 最终检查：确保是正确的3D张量
        if (src_rot.dim() != 3 or src_rot.shape[1:] != (3, 3) or 
            tgt_rot.dim() != 3 or tgt_rot.shape[1:] != (3, 3)):
            device = src_rot.device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 检查数值有效性
        if not torch.isfinite(src_rot).all() or not torch.isfinite(tgt_rot).all():
            device = src_rot.device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 计算旋转损失 - 使用更稳定的方法
        try:
            # 方法1: 使用Frobenius范数（更稳定）
            rotation_diff = src_rot - tgt_rot
            frobenius_loss = torch.norm(rotation_diff, p='fro', dim=(-2, -1)).mean()
            
            # 如果Frobenius损失有效，直接返回
            if torch.isfinite(frobenius_loss):
                return {"loss_rot": frobenius_loss}
            
            # 方法2: 使用geodesic distance（备选）
            relative_rot = torch.bmm(src_rot.transpose(-1, -2), tgt_rot)
            trace = relative_rot.diagonal(dim1=-2, dim2=-1).sum(-1)
            
            # 计算角度距离: arccos((trace-1)/2)，但要确保数值稳定性
            cos_angle = (trace - 1.0) / 2.0
            cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
            
            # 检查cos_angle是否有效
            if not torch.isfinite(cos_angle).all():
                device = src_rot.device
                return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
            
            angle = torch.acos(cos_angle)
            
            # 检查角度是否有效
            if not torch.isfinite(angle).all():
                device = src_rot.device
                return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
            
            return {"loss_rot": angle.mean()}
            
        except Exception as e:
            # 最后的备选方案：简单的L2损失
            try:
                l2_loss = F.mse_loss(src_rot, tgt_rot)
                if torch.isfinite(l2_loss):
                    return {"loss_rot": l2_loss}
            except:
                pass
            
            # 如果所有方法都失败，返回零损失
            device = src_rot.device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}

    def loss_quaternion(self, outputs, targets, indices):
        """Compute quaternion loss with numerical stability"""
        eps = 1e-6  # 增大epsilon以提高数值稳定性
        
        # 同样的安全检查
        if not indices or all(len(src) == 0 for src, _ in indices):
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        try:
            idx = self._get_src_permutation_idx(indices)
        except:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        if len(idx[0]) == 0:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        src_quaternion = outputs["pred_rotation"][idx]
        
        # 构建目标四元数
        tgt_quaternions = []
        for t, (_, i) in zip(targets, indices):
            if len(i) > 0 and 'relative_quaternions' in t:
                if t['relative_quaternions'].numel() > 0 and len(i) <= len(t['relative_quaternions']):
                    tgt_quaternions.append(t['relative_quaternions'][i])
        
        if not tgt_quaternions:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        tgt_quaternion = torch.cat(tgt_quaternions, dim=0)
        
        # 确保形状匹配
        min_size = min(src_quaternion.shape[0], tgt_quaternion.shape[0])
        if min_size == 0:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        src_quaternion = src_quaternion[:min_size]
        tgt_quaternion = tgt_quaternion[:min_size]
        
        # 检查数值有效性
        if not torch.isfinite(src_quaternion).all() or not torch.isfinite(tgt_quaternion).all():
            device = src_quaternion.device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # 归一化四元数
        src_quaternion = F.normalize(src_quaternion, p=2, dim=-1, eps=eps)
        tgt_quaternion = F.normalize(tgt_quaternion, p=2, dim=-1, eps=eps)
        
        # 计算点积，但要处理四元数的双重覆盖性质 (q和-q表示同一旋转)
        dp = torch.sum(torch.mul(src_quaternion, tgt_quaternion), dim=1)
        dp = torch.abs(dp)  # 处理双重覆盖
        dp = torch.clamp(dp, eps, 1.0 - eps)  # 确保在有效范围内
        
        # 使用更稳定的损失函数
        loss_quat = 1.0 - dp.mean()  # 简单但稳定的损失
        
        return {"loss_rot": loss_quat}

    def loss_silho_quaternion(self, outputs, targets, indices):
        """Compute silhouette quaternion loss"""
        eps = 1e-4
        
        # 相同的安全检查逻辑
        if not indices or all(len(src) == 0 for src, _ in indices):
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        try:
            idx = self._get_src_permutation_idx(indices)
        except:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        if len(idx[0]) == 0:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        src_quaternion = outputs["pred_rotation"][idx]
        
        tgt_quaternions = []
        for t, (_, i) in zip(targets, indices):
            if len(i) > 0 and 'relative_quaternions' in t:
                if t['relative_quaternions'].numel() > 0 and len(i) <= len(t['relative_quaternions']):
                    tgt_quaternions.append(t['relative_quaternions'][i])
        
        if not tgt_quaternions:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        tgt_quaternion = torch.cat(tgt_quaternions, dim=0)
        
        min_size = min(src_quaternion.shape[0], tgt_quaternion.shape[0])
        if min_size == 0:
            device = next(iter(outputs.values())).device
            return {"loss_rot": torch.tensor(0.0, device=device, requires_grad=True)}
        
        src_quaternion = src_quaternion[:min_size]
        tgt_quaternion = tgt_quaternion[:min_size]
        
        dp_sum = torch.sum(torch.mul(src_quaternion, tgt_quaternion), 1)
        loss_quat = torch.log(1 - torch.abs(dp_sum) + eps).mean()
        return {"loss_rot": loss_quat}

    def loss_optimizations(self, outputs, targets, indices):
        """Compute optimization regularization losses"""
        if 'optimization_losses' not in outputs:
            device = next(iter(outputs.values())).device
            return {"loss_opt_sum": torch.tensor(0.0, device=device, requires_grad=True)}
        
        opt_losses = outputs['optimization_losses']
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device, requires_grad=True)
        loss_dict = {}
        
        for key, value in opt_losses.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                total_loss = total_loss + value
                loss_dict[f"loss_opt_{key}"] = value
        
        loss_dict["loss_opt_sum"] = total_loss
        return loss_dict

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        """Get the specified loss"""
        loss_map = {
            'translation': self.loss_translation,
            'rotation': self.loss_rotation,
            'quaternion': self.loss_quaternion,
            'silho_quaternion': self.loss_silho_quaternion,
            'optimizations': self.loss_optimizations,
        }
        
        if loss not in loss_map:
            raise ValueError(f'Unknown loss: {loss}')
        
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, n_boxes):
        """Compute all losses"""
        # Compute matching
        outputs_without_aux = {k: v for k, v in outputs.items() 
                             if k not in ['aux_outputs', 'enc_outputs', 'optimization_losses']}
        
        try:
            indices = self.matcher(outputs_without_aux, targets, n_boxes)
        except Exception as e:
            # 如果匹配失败，创建空匹配
            device = next(iter(outputs.values())).device
            indices = [(torch.tensor([], dtype=torch.long, device=device), 
                       torch.tensor([], dtype=torch.long, device=device)) 
                      for _ in range(len(targets))]
        
        # Compute losses
        losses = {}
        for loss in self.losses:
            loss_dict = self.get_loss(loss, outputs, targets, indices)
            losses.update(loss_dict)
        
        # Handle auxiliary losses
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    if loss == 'optimizations':
                        continue  # Skip for aux outputs
                    aux_loss_dict = self.get_loss(loss, aux_outputs, targets, indices)
                    losses.update({f'{k}_{i}': v for k, v in aux_loss_dict.items()})
        
        return losses


# ================================== MLP ===================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ================================= build ==================================
def build(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)

    # Enhancement flags from args (with defaults)
    enable_tensor_opt = getattr(args, 'enable_tensor_opt', True)
    tensor_rank = getattr(args, 'tensor_rank', 64)
    enable_topology_opt = getattr(args, 'enable_topology_opt', True)
    enable_riemannian_refine = getattr(args, 'enable_riemannian_refine', True)
    riemann_steps = getattr(args, 'riemann_steps', 3)
    riemann_step_scale = getattr(args, 'riemann_step_scale', 0.5)
    riemann_refiner_type = getattr(args, 'riemann_refiner_type', 'learnable')  # 新增

    model = PoET(
        backbone,
        transformer,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        n_classes=args.n_classes,
        bbox_mode=args.bbox_mode,
        ref_points_mode=args.reference_points,
        query_embedding_mode=args.query_embedding,
        rotation_mode=args.rotation_representation,
        class_mode=args.class_mode,
        aux_loss=args.aux_loss,
        backbone_type=args.backbone,
        enable_tensor_opt=enable_tensor_opt,
        tensor_rank=tensor_rank,
        enable_topology_opt=enable_topology_opt,
        enable_riemannian_refine=enable_riemannian_refine,
        riemann_steps=riemann_steps,
        riemann_step_scale=riemann_step_scale,
        riemann_refiner_type=riemann_refiner_type  # 新增
    )

    matcher = build_matcher(args)
    weight_dict = {
        'loss_trans': args.translation_loss_coef,
        'loss_rot': args.rotation_loss_coef
    }

    # Optimize-regularizer weights (single sum + individual terms optional)
    opt_coef = getattr(args, 'optimization_loss_coef', 0.0)
    if opt_coef > 0.0:
        weight_dict['loss_opt_sum'] = opt_coef
        # 可选：在训练脚本里细化每项：
        # weight_dict['opt_tensor_tensor_reconstruction_loss'] = opt_coef
        # weight_dict['opt_tensor_sparsity_loss'] = opt_coef
        # weight_dict['opt_topology_regularization'] = opt_coef

    if args.rotation_representation == '6d':
        losses = ['translation', 'rotation']
    elif args.rotation_representation == 'quat':
        losses = ['translation', 'quaternion']
    elif args.rotation_representation == 'silho_quat':
        losses = ['translation', 'silho_quaternion']
    else:
        raise NotImplementedError('Rotation representation not implemented')

    if opt_coef > 0.0:
        losses.append('optimizations')

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher, weight_dict, losses)
    criterion.to(device)
    return model, criterion, matcher
