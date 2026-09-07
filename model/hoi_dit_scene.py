"""DiT backbone with gated local and ViT global scene conditioning."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict

try:
    from vit_pytorch import ViT
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False
    print("Warning: vit_pytorch not installed. Global scene encoding with ViT will not be available.")


def modulate(x, shift, scale):
    return x * (1 + scale) + (shift if shift is not None else 0)


class TMRAlignmentLoss(nn.Module):
    """TMR-style contrastive alignment loss for text-motion correspondence."""
    def __init__(self, temperature: float = 0.7, learnable_temp: bool = True):
        super().__init__()
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        else:
            self.register_buffer('logit_scale', torch.ones([]) * np.log(1 / temperature))

    def forward(self, text_emb: torch.Tensor, motion_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_emb = F.normalize(text_emb, dim=-1, eps=1e-8)
        motion_emb = F.normalize(motion_emb, dim=-1, eps=1e-8)
        logit_scale = self.logit_scale.exp().clamp(min=0.01, max=100)
        logits = logit_scale * torch.matmul(text_emb, motion_emb.T)
        bs = logits.shape[0]
        labels = torch.arange(bs, device=logits.device)
        loss_t2m = F.cross_entropy(logits, labels)
        loss_m2t = F.cross_entropy(logits.T, labels)
        loss = (loss_t2m + loss_m2t) / 2
        with torch.no_grad():
            acc_t2m = (logits.argmax(dim=-1) == labels).float().mean()
            acc_m2t = (logits.T.argmax(dim=-1) == labels).float().mean()
        return {
            'loss': loss,
            'loss_t2m': loss_t2m,
            'loss_m2t': loss_m2t,
            'acc_t2m': acc_t2m,
            'acc_m2t': acc_m2t,
        }


class MotionEncoder(nn.Module):
    """Motion encoder for TMR alignment."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 256,
                 output_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, output_dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def forward(self, motion: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if motion.dim() == 4:
            bs, njoints, nfeats, nframes = motion.shape
            motion = motion.reshape(bs, njoints * nfeats, nframes)
        else:
            bs, d, nframes = motion.shape
        motion = motion.permute(2, 0, 1)
        x = self.input_proj(motion)
        cls_tokens = self.cls_token.expand(-1, bs, -1)
        x = torch.cat([cls_tokens, x], dim=0)
        x = self.pos_encoder(x)
        if mask is not None:
            cls_mask = torch.ones(bs, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_output = x[0]
        output = self.output_proj(cls_output)
        return output


class TextPooler(nn.Module):
    """Pool T5 token sequence to single embedding for alignment."""
    def __init__(self, input_dim: int, output_dim: int = 512, pool_type: str = 'mean'):
        super().__init__()
        self.pool_type = pool_type
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, text_tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self.pool_type == 'mean':
            if mask is not None:
                mask = mask.unsqueeze(-1)
                mask_sum = mask.sum(dim=1).clamp(min=1)
                pooled = (text_tokens * mask).sum(dim=1) / mask_sum
            else:
                pooled = text_tokens.mean(dim=1)
        elif self.pool_type == 'first':
            pooled = text_tokens[:, 0]
        else:
            pooled = text_tokens.mean(dim=1)
        return self.proj(pooled)


class GatedSceneFusion(nn.Module):
    """Fuse 1536D spatial and 768D semantic scene features with a gated O(N) path."""

    def __init__(
        self,
        spatial_dim: int = 1536,
        semantic_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 512,
        xyz_fourier_freqs: int = 16,
        xyz_enc_dim: int = 64,
        dropout: float = 0.1,
        gate_init_value: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.xyz_fourier_freqs = xyz_fourier_freqs
        self.xyz_enc_dim = xyz_enc_dim

        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.gate = nn.Parameter(torch.full((hidden_dim,), gate_init_value))

        xyz_fourier_dim = 3 * 2 * xyz_fourier_freqs
        self.xyz_proj = nn.Sequential(
            nn.Linear(xyz_fourier_dim, xyz_enc_dim),
            nn.GELU(),
            nn.Linear(xyz_enc_dim, xyz_enc_dim),
        )

        fusion_dim = hidden_dim + xyz_enc_dim
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def fourier_encode_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        """Encode [B, N, 3] xyz coordinates as [B, N, 6 * num_freqs] Fourier features."""
        device = xyz.device
        freqs = 2.0 ** torch.arange(self.xyz_fourier_freqs, device=device, dtype=xyz.dtype)

        xyz_scaled = xyz.unsqueeze(-1) * freqs * np.pi

        encoded = torch.cat([xyz_scaled.sin(), xyz_scaled.cos()], dim=-1)
        encoded = encoded.flatten(-2)

        return encoded

    def forward(
        self,
        spatial_features: torch.Tensor,
        semantic_features: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Fuse [B, N, 1536] spatial and [B, N, 768] semantic features; mask is API-only."""
        spatial_emb = self.spatial_proj(spatial_features)
        semantic_emb = self.semantic_proj(semantic_features)

        gate = torch.sigmoid(self.gate)

        fused = spatial_emb + gate * semantic_emb

        xyz_enc = self.fourier_encode_xyz(coords)
        xyz_enc = self.xyz_proj(xyz_enc)

        combined = torch.cat([fused, xyz_enc], dim=-1)
        output = self.output_proj(combined)

        return output



class PerceiverLayer(nn.Module):
    """Apply cross-attention, self-attention, and an FFN in one Perceiver layer."""
    def __init__(self, latent_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, context: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.norm1(latents)
        x, _ = self.cross_attn(x, context, context, key_padding_mask=key_padding_mask)
        latents = latents + self.dropout1(x)

        x = self.norm2(latents)
        x, _ = self.self_attn(x, x, x)
        latents = latents + self.dropout2(x)

        x = self.norm3(latents)
        latents = latents + self.ffn(x)

        return latents


class FusedPerceiverEncoder(nn.Module):
    """Compress [B, N, feature_dim] scene features into [K, B, latent_dim] tokens."""
    def __init__(self,
                 feature_dim: int = 512,
                 latent_dim: int = 512,
                 num_latents: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(feature_dim, latent_dim) if feature_dim != latent_dim else nn.Identity()

        self.latent_queries = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        self.layers = nn.ModuleList([
            PerceiverLayer(latent_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Return [K, B, latent_dim] tokens using a [B, N] valid-point mask."""
        B = features.shape[0]

        context = self.input_proj(features)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        latents = self.latent_queries.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            latents = layer(latents, context, key_padding_mask)

        latents = self.norm(latents)

        return latents.permute(1, 0, 2)


class GlobalSceneEncoder(nn.Module):
    """ViT-based global scene encoder."""
    def __init__(
        self,
        scene_size: int = 48,
        scene_channels: int = 24,
        embed_dim: int = 512,
        vit_dim: int = 1024,
        vit_depth: int = 6,
        vit_heads: int = 16,
        vit_mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scene_size = scene_size
        self.scene_channels = scene_channels
        self.embed_dim = embed_dim

        if not VIT_AVAILABLE:
            raise ImportError(
                "vit_pytorch is required for GlobalSceneEncoder. "
                "Install with: pip install vit-pytorch"
            )

        patch_size = scene_size // 4

        self.vit = ViT(
            image_size=scene_size,
            patch_size=patch_size,
            channels=scene_channels,
            num_classes=embed_dim,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            dropout=dropout,
            emb_dropout=dropout,
        )

    def forward(self, occ_map: Tensor) -> Tensor:
        return self.vit(occ_map)


class DiTBlockPreNormCrossAttnScene(nn.Module):
    """DiT block with AdaLN-Zero plus text and scene cross-attention."""
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.text_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_text = nn.LayerNorm(d_model)
        self.dropout_text = nn.Dropout(dropout)

        self.scene_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_scene = nn.LayerNorm(d_model)
        self.dropout_scene = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 12 * d_model, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self,
                src: Tensor,
                c: Tensor,
                text_tokens: Tensor = None,
                text_key_padding_mask: Tensor = None,
                scene_tokens: Tensor = None,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        assert len(c.shape) == 3

        mod_params = self.adaLN_modulation(c).chunk(12, dim=-1)
        shift_sa, scale_sa, gate_sa = mod_params[0], mod_params[1], mod_params[2]
        shift_ta, scale_ta, gate_ta = mod_params[3], mod_params[4], mod_params[5]
        shift_sc, scale_sc, gate_sc = mod_params[6], mod_params[7], mod_params[8]
        shift_ff, scale_ff, gate_ff = mod_params[9], mod_params[10], mod_params[11]

        x = modulate(self.norm1(src), shift_sa, scale_sa)
        x = gate_sa * self.self_attn(x, x, x, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(x)

        if text_tokens is not None:
            x = modulate(self.norm_text(src), shift_ta, scale_ta)
            x = gate_ta * self.text_cross_attn(
                x, text_tokens, text_tokens,
                key_padding_mask=text_key_padding_mask)[0]
            src = src + self.dropout_text(x)

        if scene_tokens is not None:
            x = modulate(self.norm_scene(src), shift_sc, scale_sc)
            x = gate_sc * self.scene_cross_attn(x, scene_tokens, scene_tokens)[0]
            src = src + self.dropout_scene(x)

        x = modulate(self.norm2(src), shift_ff, scale_ff)
        x = gate_ff * self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout2(x)

        return src


class DiTEncoderCrossAttnScene(nn.Module):
    """Encoder stack for DiT blocks with text and scene cross-attention."""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src: Tensor,
                c: Tensor,
                text_tokens: Tensor = None,
                text_key_padding_mask: Tensor = None,
                scene_tokens: Tensor = None,
                skip: Tensor = None,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output,
                         c=c,
                         text_tokens=text_tokens,
                         text_key_padding_mask=text_key_padding_mask,
                         scene_tokens=scene_tokens,
                         skip=skip,
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class PositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to sequence features."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """Embed diffusion timesteps in the model latent space."""
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps):
        from diffusion.nn import timestep_embedding
        if timesteps.dtype in [torch.float32, torch.float64, torch.float16]:
            t_emb = timestep_embedding(timesteps, self.latent_dim)
            t_emb = t_emb.unsqueeze(1)
        else:
            t_emb = self.sequence_pos_encoder.pe[timesteps]
        return self.time_embed(t_emb).permute(1, 0, 2)


class InputProcessConcat(nn.Module):
    """Project concatenated motion conditions into latent tokens."""
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        bs, _, total_dim, nframes = x.shape
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        x = self.poseEmbedding(x)
        return x


class FinalLayer(nn.Module):
    """Project latent tokens with adaptive layer normalization."""
    def __init__(self, in_channels, out_channels, cond_channels, norm=True, zero=True, scale_only=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(in_channels, elementwise_affine=False, eps=1e-6) if norm else nn.Identity()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.scale_only = scale_only
        adaLN_width = in_channels if scale_only else 2 * in_channels
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_channels, adaLN_width, bias=True))
        if zero:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        assert len(c.shape) == 3
        if self.scale_only:
            scale = self.adaLN_modulation(c)
            shift = None
        else:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class OutputProcess(nn.Module):
    """Project latent tokens back to motion features."""
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats, norm, zero, skip=False, scale_only=False):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.skip = skip
        self.effective_dim = latent_dim * 2 if skip else latent_dim
        self.poseFinal = FinalLayer(self.effective_dim, input_feats, latent_dim, norm, zero=zero, scale_only=scale_only)

    def forward(self, output, c, skip=None):
        if self.skip:
            output = torch.cat([output, skip], dim=-1)
        nframes, bs, d = output.shape
        output = self.poseFinal(output, c=c)
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)
        return output


class SceneHOIDiT(nn.Module):
    """Generate hand-object motion from T5, TMR, gated point scenes, and global ViT context."""
    def __init__(self,
                 modeltype,
                 njoints,
                 nfeats,
                 translation,
                 pose_rep,
                 glob,
                 glob_rot,
                 latent_dim=256,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=4,
                 dropout=0.1,
                 ablation=None,
                 activation="gelu",
                 data_rep='rot6d',
                 dataset='amass',
                 clip_dim=512,
                 arch='dit_v19_bidirectional_fusion',
                 emb_trans_dec=False,
                 two_head=False,
                 t5_model_name="google/flan-t5-large",
                 text_feature_dim=None,
                 text_max_length=77,
                 use_tmr_alignment=True,
                 tmr_align_dim=512,
                 tmr_temperature=0.7,
                 tmr_loss_weight=0.1,
                 tmr_motion_encoder_layers=4,
                 use_pose_cond=True,
                 pose_cond_type='grasp',
                 grasp_frame=49,
                 use_bps=False,
                 bps_dim=3072,
                 bps_hidden_dim=256,
                 use_scene_fusion=True,
                 scene_fusion_spatial_dim=1536,
                 scene_fusion_semantic_dim=768,
                 scene_fusion_hidden_dim=512,
                 scene_fusion_main_heads=8,
                 scene_fusion_main_layers=2,
                 scene_fusion_aux_heads=4,
                 scene_fusion_aux_layers=1,
                 scene_fusion_xyz_freqs=16,
                 scene_fusion_xyz_enc_dim=64,
                 scene_perceiver_num_latents=256,
                 scene_perceiver_num_layers=2,
                 scene_perceiver_num_heads=8,
                 max_concerto_points=25000,
                 use_scene_global=False,
                 scene_size=48,
                 scene_channels=24,
                 vit_dim=1024,
                 vit_depth=6,
                 vit_heads=16,
                 vit_mlp_dim=2048,
                 **kargs):
        super().__init__()

        self.t5_model_name = t5_model_name
        self.text_max_length = text_max_length

        if "xxl" in t5_model_name:
            self.text_feature_dim = 4096
        elif "xl" in t5_model_name:
            self.text_feature_dim = 2048
        elif "large" in t5_model_name:
            self.text_feature_dim = 1024
        elif "base" in t5_model_name:
            self.text_feature_dim = 768
        elif "small" in t5_model_name:
            self.text_feature_dim = 512
        else:
            self.text_feature_dim = 1024

        self.use_tmr_alignment = use_tmr_alignment
        self.tmr_align_dim = tmr_align_dim
        self.tmr_loss_weight = tmr_loss_weight

        self.use_pose_cond = use_pose_cond
        self.pose_cond_type = pose_cond_type
        self.grasp_frame = grasp_frame
        self.use_bps = use_bps
        self.bps_dim = bps_dim
        self.bps_hidden_dim = bps_hidden_dim

        self.use_scene_fusion = use_scene_fusion
        self.scene_fusion_spatial_dim = scene_fusion_spatial_dim
        self.scene_fusion_semantic_dim = scene_fusion_semantic_dim
        self.max_concerto_points = max_concerto_points
        self.scene_perceiver_num_latents = scene_perceiver_num_latents

        self.use_scene_global = use_scene_global
        self.scene_size = scene_size
        self.scene_channels = scene_channels
        self.vit_dim = vit_dim
        self.vit_depth = vit_depth
        self.vit_heads = vit_heads
        self.vit_mlp_dim = vit_mlp_dim

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.two_head = two_head
        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.input_feats = njoints * nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = latent_dim if arch == 'gru' else 0

        concat_input_dim = self.input_feats + self.gru_emb_dim
        if self.use_pose_cond:
            concat_input_dim += self.input_feats
        if self.use_bps:
            concat_input_dim += self.bps_hidden_dim

        self.input_process = InputProcessConcat(data_rep, concat_input_dim, latent_dim)
        print(f'[SceneHOIDiT] Concat dim: {concat_input_dim} (motion={self.input_feats}, '
              f'pose_cond={self.input_feats if self.use_pose_cond else 0}, '
              f'bps={self.bps_hidden_dim if self.use_bps else 0})')

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.emb_trans_dec = emb_trans_dec

        print(f"SceneHOIDiT (T5-large + TMR + gated fusion + Perceiver + global ViT: {t5_model_name})")
        seqTransEncoderLayer = DiTBlockPreNormCrossAttnScene(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)

        self.output_process = OutputProcess(
            data_rep, self.input_feats, latent_dim, njoints, nfeats,
            norm=True, zero=True, skip=False, scale_only=False)

        if two_head:
            self.output_process2 = OutputProcess(
                data_rep, self.input_feats, latent_dim, njoints, nfeats,
                norm=True, zero=True, skip=False, scale_only=False)

        self.seqTransEncoder = DiTEncoderCrossAttnScene(seqTransEncoderLayer, num_layers=num_layers)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.text_proj = nn.Sequential(
                    nn.Linear(self.text_feature_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                )
                print(f'[SceneHOIDiT] T5: {t5_model_name} ({self.text_feature_dim} -> {latent_dim})')
                print(f'Loading T5 encoder: {t5_model_name}...')
                self.text_model = self.load_and_freeze_text_encoder()

        if self.use_tmr_alignment:
            self.motion_encoder = MotionEncoder(
                input_dim=self.input_feats,
                latent_dim=latent_dim,
                output_dim=tmr_align_dim,
                num_layers=tmr_motion_encoder_layers,
                num_heads=num_heads,
                dropout=dropout
            )
            print(f'[SceneHOIDiT] Motion Encoder: {self.input_feats} -> {tmr_align_dim}')

            self.text_pooler = TextPooler(
                input_dim=self.text_feature_dim,
                output_dim=tmr_align_dim,
                pool_type='mean'
            )
            print(f'[SceneHOIDiT] Text Pooler: {self.text_feature_dim} -> {tmr_align_dim}')

            self.tmr_loss = TMRAlignmentLoss(temperature=tmr_temperature, learnable_temp=True)
            print(f'[SceneHOIDiT] Contrastive Loss: temp={tmr_temperature}, weight={tmr_loss_weight}')

        if self.use_pose_cond:
            if self.pose_cond_type == 'grasp':
                print('[Concat] POSE COND: frame 0 (grasp model)')
            else:
                print(f'[Concat] POSE COND: frames 0 and {self.grasp_frame} (full model)')

        if self.use_bps:
            self.bps_encoder = nn.Sequential(
                nn.Linear(bps_dim, 512),
                nn.ReLU(),
                nn.Linear(512, bps_hidden_dim),
            )
            print(f'[Concat] BPS ENCODER ({bps_dim} -> 512 -> {bps_hidden_dim})')

        if self.use_scene_fusion:
            self.scene_fusion = GatedSceneFusion(
                spatial_dim=scene_fusion_spatial_dim,
                semantic_dim=scene_fusion_semantic_dim,
                hidden_dim=scene_fusion_hidden_dim,
                output_dim=scene_fusion_hidden_dim,
                xyz_fourier_freqs=scene_fusion_xyz_freqs,
                xyz_enc_dim=scene_fusion_xyz_enc_dim,
                dropout=dropout,
            )
            print(f'[SceneHOIDiT] Gated: spatial({scene_fusion_spatial_dim}D) + '
                  f'semantic({scene_fusion_semantic_dim}D) -> {scene_fusion_hidden_dim}D')
            print('  Formula: fused = spatial_emb + sigmoid(gate) * semantic_emb')
            print(f'  XYZ encoding: {scene_fusion_xyz_freqs} freqs -> {scene_fusion_xyz_enc_dim}D')

            self.scene_encoder = FusedPerceiverEncoder(
                feature_dim=scene_fusion_hidden_dim,
                latent_dim=latent_dim,
                num_latents=scene_perceiver_num_latents,
                num_layers=scene_perceiver_num_layers,
                num_heads=scene_perceiver_num_heads,
                dropout=dropout
            )
            print(f'[SceneHOIDiT] {scene_fusion_hidden_dim}D x {max_concerto_points} pts '
                  f'-> {scene_perceiver_num_latents} tokens')

        if self.use_scene_global:
            self.embed_scene_global = GlobalSceneEncoder(
                scene_size=scene_size,
                scene_channels=scene_channels,
                embed_dim=latent_dim,
                vit_dim=vit_dim,
                vit_depth=vit_depth,
                vit_heads=vit_heads,
                vit_mlp_dim=vit_mlp_dim,
                dropout=dropout,
            )
            print(f'[SceneHOIDiT] ViT ({scene_size}x{scene_size}x{scene_channels} -> {latent_dim})')

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters()
                if not name.startswith('clip_model.') and not name.startswith('text_model.')]

    def load_and_freeze_text_encoder(self):
        from model.text_encoders import T5TextEncoder
        text_encoder = T5TextEncoder(
            model_name=self.t5_model_name,
            max_length=self.text_max_length,
            device="cpu",
        )
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad = False
        return text_encoder

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob)
            if cond.dim() == 3:
                mask = mask.view(bs, 1, 1)
            else:
                mask = mask.view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def encode_text_with_mask(self, raw_text):
        device = next(self.parameters()).device
        text_model_device = next(self.text_model.parameters()).device
        if text_model_device != device:
            self.text_model = self.text_model.to(device)
        return self.text_model.encode_text_with_mask(raw_text)

    def create_cond_mask(self, bs, nframes, device):
        cond_mask = torch.ones(bs, nframes, device=device)
        if self.pose_cond_type == 'grasp':
            cond_mask[:, 0] = 0
        else:
            cond_mask[:, 0] = 0
            if self.grasp_frame < nframes:
                cond_mask[:, self.grasp_frame] = 0
        return cond_mask

    def compute_tmr_loss(self, x_start: torch.Tensor, text_emb: torch.Tensor,
                         text_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if not self.use_tmr_alignment:
            return {'tmr_loss': torch.tensor(0.0, device=x_start.device)}
        motion_emb = self.motion_encoder(x_start)
        text_pooled = self.text_pooler(text_emb, text_mask)
        loss_dict = self.tmr_loss(text_pooled, motion_emb)
        loss_dict['tmr_loss'] = loss_dict['loss'] * self.tmr_loss_weight
        return loss_dict

    def forward(self, x, timesteps, y=None):
        """Predict [B, J, F, T] motion from cached or on-demand T5, point-scene, and occupancy features."""
        bs, njoints, nfeats, nframes = x.shape
        device = next(self.parameters()).device
        force_mask = y.get('uncond', False)

        emb = self.embed_timestep(timesteps)

        if self.use_scene_global and 'occ_map' in y and y['occ_map'] is not None:
            occ_map = y['occ_map'].to(device).float()
            scene_global_emb = self.embed_scene_global(occ_map)
            scene_global_emb = self.mask_cond(scene_global_emb, force_mask=force_mask)
            emb = emb + scene_global_emb.unsqueeze(0)

        text_tokens = None
        text_key_padding_mask = None
        enc_text_raw = None
        text_mask_raw = None

        if 'text' in self.cond_mode:
            if 't5_emb' in y and y['t5_emb'] is not None:
                enc_text = y['t5_emb'].to(device).float()
                text_mask = y.get('t5_mask', None)
                if text_mask is not None:
                    text_mask = text_mask.to(device)
                enc_text_raw = enc_text
                text_mask_raw = text_mask
                if text_mask is not None:
                    text_key_padding_mask = (text_mask == 0).to(device)
            else:
                enc_text, text_mask = self.encode_text_with_mask(y['text'])
                enc_text = enc_text.to(device)
                text_mask = text_mask.to(device)
                enc_text_raw = enc_text
                text_mask_raw = text_mask
                text_key_padding_mask = (text_mask == 0).to(device)

            enc_text = self.mask_cond(enc_text, force_mask=force_mask)
            text_tokens = self.text_proj(enc_text)
            text_tokens = text_tokens.permute(1, 0, 2)

        scene_tokens = None
        if self.use_scene_fusion:
            concerto_feat = y.get('concerto_features', None)
            semantic_feat = y.get('semantic_features', None)
            concerto_coords = y.get('concerto_coords', None)
            concerto_mask = y.get('concerto_mask', None)

            if concerto_feat is not None and semantic_feat is not None and concerto_coords is not None:
                concerto_feat = concerto_feat.to(device).float()
                semantic_feat = semantic_feat.to(device).float()
                concerto_coords = concerto_coords.to(device).float()
                if concerto_mask is not None:
                    concerto_mask = concerto_mask.to(device)

                fused_features = self.scene_fusion(
                    spatial_features=concerto_feat,
                    semantic_features=semantic_feat,
                    coords=concerto_coords,
                    mask=concerto_mask,
                )

                scene_tokens = self.scene_encoder(fused_features, concerto_mask)

        if self.use_pose_cond:
            if 'x_start' in y and y['x_start'] is not None:
                x_start = y['x_start'].to(device).float()
                if 'cond_mask' in y and y['cond_mask'] is not None:
                    cond_mask = y['cond_mask'].to(device).float()
                else:
                    cond_mask = self.create_cond_mask(bs, nframes, device)
                cond_mask_expanded = cond_mask.unsqueeze(1).unsqueeze(2)
                x_pose_cond = x_start * (1 - cond_mask_expanded)
            else:
                x_pose_cond = torch.zeros_like(x)

        if self.use_bps:
            if 'object_bps' in y and y['object_bps'] is not None:
                if isinstance(y['object_bps'], list):
                    obj_bps = torch.stack([
                        torch.from_numpy(b) if isinstance(b, np.ndarray) else b
                        for b in y['object_bps']
                    ]).to(device).float()
                else:
                    obj_bps = y['object_bps'].to(device).float()
                if obj_bps.dim() == 1:
                    obj_bps = obj_bps.unsqueeze(0)
                if obj_bps.dim() > 2:
                    obj_bps = obj_bps.reshape(obj_bps.shape[0], -1)
                bps_feat = self.bps_encoder(obj_bps)
                bps_feat = bps_feat.unsqueeze(-1).unsqueeze(-1)
                bps_feat = bps_feat.expand(-1, -1, 1, nframes)
            else:
                bps_feat = torch.zeros(bs, self.bps_hidden_dim, 1, nframes, device=device)

        x_flat = x.reshape(bs, njoints * nfeats, nframes)
        concat_flat = [x_flat]

        if self.use_pose_cond:
            x_pose_cond_flat = x_pose_cond.reshape(bs, njoints * nfeats, nframes)
            concat_flat.append(x_pose_cond_flat)

        if self.use_bps:
            bps_flat = bps_feat.reshape(bs, self.bps_hidden_dim, nframes)
            concat_flat.append(bps_flat)

        x_concat = torch.cat(concat_flat, dim=1)
        x_concat = x_concat.unsqueeze(1)

        x = self.input_process(x_concat)

        skip = x = self.sequence_pos_encoder(x)
        x = self.seqTransEncoder(
            x, c=emb,
            text_tokens=text_tokens,
            text_key_padding_mask=text_key_padding_mask,
            scene_tokens=scene_tokens,
            skip=skip
        )

        output = self.output_process(x, c=emb, skip=skip)

        tmr_loss_dict = None
        if self.training and self.use_tmr_alignment and 'x_start' in y and y['x_start'] is not None:
            if enc_text_raw is not None:
                tmr_loss_dict = self.compute_tmr_loss(
                    y['x_start'].to(device).float(),
                    enc_text_raw,
                    text_mask_raw
                )

        if self.two_head:
            output2 = self.output_process2(x, c=emb, skip=skip)
            if tmr_loss_dict is not None:
                return output, output2, tmr_loss_dict
            return output, output2
        else:
            if tmr_loss_dict is not None:
                return output, tmr_loss_dict
            return output

    def _apply(self, fn):
        super()._apply(fn)
        return self

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        return self
