import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import numpy as np

# =============================================================================
# 1. 核心组件 (完全来自 train.py，未修改)
# =============================================================================

class PatchEmbedding(nn.Module):
    """Layer that divides images into patches and embeds them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)        # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (B, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        
    def forward(self, x):
        # Pre-norm residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PyTorchViT(nn.Module):
    """PyTorch native Vision Transformer (Exact copy from train.py)"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, num_layers=12, num_heads=12, mlp_dim=3072, 
                 num_classes=1000, dropout=0.1):
        super(PyTorchViT, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        # self.head = nn.Linear(embed_dim, num_classes) # STAR 不需要分类头，这里可以保留或注释
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_features=False):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks with feature extraction
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        x = self.ln(x)
        
        if return_features:
            return x, features  # Return features from all layers
        else:
            return x


# =============================================================================
# 2. 适配器 ViTSceneEncoder (做最小修改以适配 STAR)
# =============================================================================

class ViTSceneEncoder(nn.Module):
    """
    Adapter class to connect PyTorchViT with STAR.
    Retains the structure of 'train.py' ViTSceneEncoder but simplifies forward pass.
    """
    def __init__(self, embedding_dim=64, img_size=224, patch_size=16, 
                 embed_dim=768, num_layers=6, num_heads=8, freeze_vit=False, 
                 use_multi_scale=True):
        super(ViTSceneEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim

          # EfficientNet-B0，预训练权重
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # 保留 features + avgpool，去掉分类头
        self.backbone = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool,   # 输出 (B, 1280, 1, 1)
        )
        
        if freeze_vit:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen (EfficientNet-B0)")
        
        # 1280 -> embedding_dim，只训练这个投影层
        self.projector = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )
        # self.use_multi_scale = use_multi_scale
        
        # --- 使用 train.py 同款的 PyTorchViT ---
        # self.vit = PyTorchViT(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_channels=3,
        #     embed_dim=embed_dim,
        #     num_layers=num_layers,
        #     num_heads=num_heads,
        #     mlp_dim=embed_dim * 4,
        #     num_classes=1000, 
        #     dropout=0.1
        # )
        
        print(f"✅ EfficientNet-B0 backbone initialized (frozen={freeze_vit})")
        
        # # 冻结逻辑
        # if freeze_vit:
        #     for param in self.vit.parameters():
        #         param.requires_grad = False
        #     print("🔒 ViT weights are frozen.")
        # else:
        #     print("🔓 ViT weights are trainable.")
        
        # # 多尺度融合权重 (train.py 逻辑)
        # if use_multi_scale:
        #     self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        #     feature_dim = embed_dim
        # else:
        #     feature_dim = embed_dim
        
        # # Attention pooling (train.py 逻辑)
        # self.attention_pool = nn.MultiheadAttention(
        #     embed_dim=feature_dim,
        #     num_heads=8,
        #     dropout=0.1,
        #     batch_first=True
        # )
        
        # 投影层: 将 ViT 维度 (768) 映射到 STAR 维度 (64)
        # self.feature_projector = nn.Sequential(
        #     nn.Linear(feature_dim, embedding_dim * 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(embedding_dim * 2, embedding_dim),
        #     nn.ReLU()
        # )
        
        # 上下文增强 (train.py 逻辑)
        # self.context_enhancer = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.LayerNorm(embedding_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )

    def forward(self, scene_input):
        """
        Input: (B, 3, 224, 224)
        Output: (B, embedding_dim) <- 全局池化后的向量。
        """
        # if self.use_multi_scale:
        #     features, all_layer_features = self.vit(scene_input, return_features=True)
        #     weights = F.softmax(self.layer_weights, dim=0)
        #     weighted_features = 0
        #     for i, layer_feat in enumerate(all_layer_features):
        #         weighted_features = weighted_features + weights[i] * layer_feat
        #     features = weighted_features
        # else:
        #     features = self.vit(scene_input)
            
        # # 1. 移除 CLS token，只保留 14x14 = 196 个 patch 的特征
        # patch_features = features[:, 1:, :]  # (B, 196, embed_dim)
        # B, N, C = patch_features.shape
        # H = W = int(np.sqrt(N)) # 14 
            
        # # 2. 对每个 patch 分别进行降维映射 (768 -> 64)
        # projected_features = self.feature_projector(patch_features)  # (B, 196, 64)
        # enhanced_features = self.context_enhancer(projected_features) # (B, 196, 64)
            
        # # 3. 折叠回 2D 空间特征图的形状
        # # (B, 196, 64) -> (B, 64, 196) -> (B, 64, 14, 14)
        # out_spatial = enhanced_features.transpose(1, 2).reshape(B, self.embedding_dim, H, W)
        with torch.no_grad():
            feat = self.backbone(scene_input)  # (B, 1280, 1, 1)
        feat = feat.flatten(1)                  # (B, 1280)        
        return self.projector(feat)             # (B, 32) 
        