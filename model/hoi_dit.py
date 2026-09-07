"""DiT backbone for hand-object interaction generation."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict


def modulate(x, shift, scale):
    return x * (1 + scale) + (shift if shift is not None else 0)


class TMRAlignmentLoss(nn.Module):
    """TMR-style text-motion contrastive loss with the paper's 0.7 default temperature."""
    def __init__(self, temperature: float = 0.7, learnable_temp: bool = True):
        super().__init__()
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        else:
            self.register_buffer('logit_scale', torch.ones([]) * np.log(1 / temperature))

    def forward(self, text_emb: torch.Tensor, motion_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute symmetric InfoNCE loss for [B, D] text and motion embeddings."""
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
    """Encode motion sequences for TMR alignment."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 256,
                 output_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

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
        """Encode [B, D, T] or [B, J, F, T] motion with a [B, T] validity mask."""
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
    """Pool a T5 token sequence into one alignment embedding."""
    def __init__(self, input_dim: int, output_dim: int = 512, pool_type: str = 'mean'):
        super().__init__()
        self.pool_type = pool_type
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, text_tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Pool [B, S, D] tokens into [B, output_dim] with a [B, S] validity mask."""
        if self.pool_type == 'mean':
            if mask is not None:
                mask = mask.unsqueeze(-1)
                mask_sum = mask.sum(dim=1).clamp(min=1)
                pooled = (text_tokens * mask).sum(dim=1) / mask_sum
            else:
                pooled = text_tokens.mean(dim=1)
        elif self.pool_type == 'first':
            pooled = text_tokens[:, 0]
        elif self.pool_type == 'last':
            if mask is not None:
                lengths = mask.sum(dim=1).long() - 1
                pooled = text_tokens[torch.arange(text_tokens.size(0)), lengths]
            else:
                pooled = text_tokens[:, -1]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        return self.proj(pooled)


class DiTBlockPreNormCrossAttn(nn.Module):
    """DiT block with AdaLN-Zero and cross-attention."""
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_ca = nn.LayerNorm(d_model)
        self.dropout_ca = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 9 * d_model, bias=True))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self,
                src,
                c,
                text_tokens=None,
                text_key_padding_mask=None,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs):
        assert len(c.shape) == 3

        mod_params = self.adaLN_modulation(c).chunk(9, dim=-1)
        shift_sa, scale_sa, gate_sa = mod_params[0], mod_params[1], mod_params[2]
        shift_ca, scale_ca, gate_ca = mod_params[3], mod_params[4], mod_params[5]
        shift_ff, scale_ff, gate_ff = mod_params[6], mod_params[7], mod_params[8]

        x = modulate(self.norm1(src), shift_sa, scale_sa)
        x = gate_sa * self.self_attn(x, x, x, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(x)

        if text_tokens is not None:
            x = modulate(self.norm_ca(src), shift_ca, scale_ca)
            x = gate_ca * self.cross_attn(
                x, text_tokens, text_tokens,
                key_padding_mask=text_key_padding_mask)[0]
            src = src + self.dropout_ca(x)

        x = modulate(self.norm2(src), shift_ff, scale_ff)
        x = gate_ff * self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout2(x)

        return src


class DiTEncoderCrossAttn(nn.Module):
    """Encoder stack for DiT blocks with cross-attention."""
    def __init__(self, encoder_layer, num_layers, norm=None, encoder_layers=None):
        super().__init__()
        if encoder_layers is not None:
            self.layers = nn.ModuleList(encoder_layers)
        else:
            self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src: Tensor,
                c: Tensor,
                text_tokens: Tensor = None,
                text_key_padding_mask: Tensor = None,
                skip: Tensor = None,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output,
                         c=c,
                         text_tokens=text_tokens,
                         text_key_padding_mask=text_key_padding_mask,
                         skip=skip,
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class HOIDiT(nn.Module):
    """Generate hand-object motion with T5, TMR, pose, and BPS conditioning."""
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
                 arch='dit_v12_prenorm',
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
        if two_head:
            print('Using two-head output')

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0

        concat_input_dim = self.input_feats + self.gru_emb_dim
        if self.use_pose_cond:
            concat_input_dim += self.input_feats
        if self.use_bps:
            concat_input_dim += self.bps_hidden_dim

        self.input_process = InputProcessConcat(
            self.data_rep,
            concat_input_dim,
            self.latent_dim
        )
        print(f'[HOIDiT] Concat dim: {concat_input_dim} (motion={self.input_feats}, '
              f'pose_cond={self.input_feats if self.use_pose_cond else 0}, '
              f'bps={self.bps_hidden_dim if self.use_bps else 0})')

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        print(f"HOIDiT (T5-large + TMR alignment: {t5_model_name})")
        seqTransEncoderLayer = DiTBlockPreNormCrossAttn(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation)
        add_norm_before_pred = True
        use_skip_connection = False

        self.output_process = OutputProcess(self.data_rep,
                                            self.input_feats,
                                            self.latent_dim,
                                            self.njoints,
                                            self.nfeats,
                                            norm=add_norm_before_pred,
                                            zero=True,
                                            skip=use_skip_connection,
                                            scale_only=False)
        if self.two_head:
            self.output_process2 = OutputProcess(self.data_rep,
                                                 self.input_feats,
                                                 self.latent_dim,
                                                 self.njoints,
                                                 self.nfeats,
                                                 norm=add_norm_before_pred,
                                                 zero=True,
                                                 skip=use_skip_connection,
                                                 scale_only=False)

        self.seqTransEncoder = DiTEncoderCrossAttn(
            seqTransEncoderLayer,
            num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.text_proj = nn.Sequential(
                    nn.Linear(self.text_feature_dim, self.latent_dim),
                    nn.LayerNorm(self.latent_dim),
                )
                print(f'[HOIDiT] T5: {t5_model_name} ({self.text_feature_dim} -> {self.latent_dim})')
                print(f'Loading T5 encoder: {t5_model_name}...')
                self.text_model = self.load_and_freeze_text_encoder()

        if self.use_tmr_alignment:
            self.motion_encoder = MotionEncoder(
                input_dim=self.input_feats,
                latent_dim=self.latent_dim,
                output_dim=self.tmr_align_dim,
                num_layers=tmr_motion_encoder_layers,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            print(f'[HOIDiT] Motion Encoder: {self.input_feats} -> {self.tmr_align_dim}')

            self.text_pooler = TextPooler(
                input_dim=self.text_feature_dim,
                output_dim=self.tmr_align_dim,
                pool_type='mean'
            )
            print(f'[HOIDiT] Text Pooler: {self.text_feature_dim} -> {self.tmr_align_dim}')

            self.tmr_loss = TMRAlignmentLoss(
                temperature=tmr_temperature,
                learnable_temp=True
            )
            print(f'[HOIDiT] Contrastive Loss: temp={tmr_temperature}, weight={tmr_loss_weight}')

        if self.use_pose_cond:
            if self.pose_cond_type == 'grasp':
                print('[Concat] POSE COND: frame 0 (grasp model)')
            else:
                print(f'[Concat] POSE COND: frames 0 and {self.grasp_frame} (full model)')

        if self.use_bps:
            self.bps_encoder = nn.Sequential(
                nn.Linear(self.bps_dim, 512),
                nn.ReLU(),
                nn.Linear(512, self.bps_hidden_dim),
            )
            print(f'[Concat] BPS ENCODER ({self.bps_dim} -> 512 -> {self.bps_hidden_dim})')

    def parameters_wo_clip(self):
        return [
            p for name, p in self.named_parameters()
            if not name.startswith('clip_model.') and not name.startswith('text_model.')
        ]

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
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob)
            if cond.dim() == 3:
                mask = mask.view(bs, 1, 1)
            else:
                mask = mask.view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        device = next(self.parameters()).device

        text_model_device = next(self.text_model.parameters()).device
        if text_model_device != device:
            self.text_model = self.text_model.to(device)
        return self.text_model.encode_text(raw_text)

    def encode_text_with_mask(self, raw_text):
        device = next(self.parameters()).device

        text_model_device = next(self.text_model.parameters()).device
        if text_model_device != device:
            self.text_model = self.text_model.to(device)
        return self.text_model.encode_text_with_mask(raw_text)

    def create_cond_mask(self, bs, nframes, device):
        """Create conditioning mask for pose inpainting."""
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
        """Align [B, J, F, T] motion with [B, S, D] T5 features."""
        if not self.use_tmr_alignment:
            return {'tmr_loss': torch.tensor(0.0, device=x_start.device)}

        motion_emb = self.motion_encoder(x_start)

        text_pooled = self.text_pooler(text_emb, text_mask)

        loss_dict = self.tmr_loss(text_pooled, motion_emb)

        loss_dict['tmr_loss'] = loss_dict['loss'] * self.tmr_loss_weight

        return loss_dict

    def forward(self, x, timesteps, y=None):
        """Predict [B, J, F, T] motion with cached or on-demand T5 features and optional TMR metrics."""
        bs, njoints, nfeats, nframes = x.shape
        device = next(self.parameters()).device
        force_mask = y.get('uncond', False)

        emb = self.embed_timestep(timesteps)

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
        x = self.seqTransEncoder(x, c=emb, text_tokens=text_tokens,
                                  text_key_padding_mask=text_key_padding_mask, skip=skip)

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


class PositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to sequence features."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
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

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
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
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, _, total_dim, nframes = x.shape
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        x = self.poseEmbedding(x)
        return x


class FinalLayer(nn.Module):
    """Project latent tokens with adaptive layer normalization."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_channels,
                 norm=True,
                 zero=True,
                 scale_only=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(in_channels,
                                       elementwise_affine=False,
                                       eps=1e-6) if norm else nn.Identity()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.scale_only = scale_only
        adaLN_width = in_channels if self.scale_only else 2 * in_channels
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_channels, adaLN_width, bias=True))
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
    def __init__(self,
                 data_rep,
                 input_feats,
                 latent_dim,
                 njoints,
                 nfeats,
                 norm,
                 zero,
                 skip=False,
                 scale_only=False):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.skip = skip

        self.effective_dim = self.latent_dim * 2 if self.skip else self.latent_dim
        self.poseFinal = FinalLayer(self.effective_dim,
                                    self.input_feats,
                                    self.latent_dim,
                                    norm,
                                    zero=zero,
                                    scale_only=scale_only)
        if self.data_rep == 'rot_vel':
            self.velFinal = FinalLayer(self.effective_dim,
                                       self.input_feats,
                                       self.latent_dim,
                                       norm,
                                       zero=zero,
                                       scale_only=scale_only)

    def forward(self, output, c, skip=None):
        if self.skip:
            output = torch.cat([output, skip], dim=-1)

        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output, c=c)
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]
            first_pose = self.poseFinal(first_pose, c=c)
            vel = output[1:]
            vel = self.velFinal(vel, c=c)
            output = torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)
        return output
