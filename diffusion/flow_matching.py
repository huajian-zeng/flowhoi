"""Flow matching with x1 prediction based on the FlowMotion approach."""

from typing import Optional, Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from diffusion.nn import sum_flat


class FlowMatching:
    """Predict x1 and Euler-sample from noise at t=0 to data at t=1."""

    def __init__(
        self,
        num_sampling_steps: int = 50,
        sigma_min: float = 0.0,
        traj_extra_weight: float = 1.0,
        drop_redundant: bool = False,
        lambda_vel: float = 0.0,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.sigma_min = sigma_min
        self.traj_extra_weight = traj_extra_weight
        self.drop_redundant = drop_redundant
        self.lambda_vel = lambda_vel
        self.l2_loss = lambda a, b: (a - b) ** 2

    def masked_l2(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked L2 loss."""
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        non_zero_elements = torch.clamp(non_zero_elements, min=1.0)
        return loss / non_zero_elements

    def masked_l2_weighted(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor,
        time_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute DDPM-normalized weighted L2 over [B, J, Jdim, T] tensors."""
        loss = self.l2_loss(a, b)

        weights = weights / weights.sum(dim=[1, 2], keepdims=True)
        loss = loss * weights

        if time_weights is not None:
            loss = loss * time_weights

        loss = sum_flat(loss * mask.float())

        non_zero_elements = sum_flat(mask)
        non_zero_elements = torch.clamp(non_zero_elements, min=1.0)
        return loss / non_zero_elements

    def _get_traj_weights(
        self,
        target: torch.Tensor,
        model_kwargs: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Weight 81D left-hand/object pose slices [:6]/[-9:] or 117D L/R-hand/object positions [:3]/[63:66]/[-15:-6]."""
        weights = torch.ones(
            *target.shape[:-1], 1,
            device=target.device,
            dtype=target.dtype
        )

        if self.traj_extra_weight == 1.0:
            return weights

        w2 = self.traj_extra_weight ** 2

        has_motion_target = (
            model_kwargs is not None and
            'y' in model_kwargs and
            'motion_target' in model_kwargs['y']
        )

        if has_motion_target or self.drop_redundant:
            weights[:, :6] *= w2
            weights[:, -9:] *= w2
        else:
            weights[:, :3] *= w2
            weights[:, 63:66] *= w2
            weights[:, -15:-6] *= w2

        return weights

    def get_x_t(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate x_t = t*x1 + (1 - (1 - sigma_min)*t)*x0 along the OT path."""
        if t.dim() == 1:
            t = t[:, None, None, None]

        x_t = t * x_start + (1 - (1 - self.sigma_min) * t) * noise
        return x_t

    def get_velocity_from_x1(
        self,
        x1_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Derive v = (x1 - (1 - sigma_min)*x_t) / (1 - (1 - sigma_min)*t)."""
        if t.dim() == 1:
            t = t[:, None, None, None]

        denom = 1 - (1 - self.sigma_min) * t
        denom = torch.clamp(denom, min=1e-6)

        v_t = (x1_pred - (1 - self.sigma_min) * x_t) / denom
        return v_t

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        model_kwargs: Optional[dict] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train direct x1 prediction with trajectory-weighted L2 and optional velocity/TMR losses."""
        if model_kwargs is None:
            model_kwargs = {}

        mask = model_kwargs.get('y', {}).get('mask', None)
        if mask is None:
            mask = torch.ones(x_start.shape[0], 1, 1, x_start.shape[-1],
                            device=x_start.device, dtype=x_start.dtype)

        B = x_start.shape[0]
        device = x_start.device

        if noise is None:
            noise = torch.randn_like(x_start)

        t = torch.rand(B, device=device, dtype=x_start.dtype)
        t_expanded = t[:, None, None, None]

        x_t = self.get_x_t(x_start, noise, t_expanded)

        target = x_start

        if 'y' in model_kwargs:
            model_kwargs['y']['x_start'] = x_start

        model_output = model(x_t, t, **model_kwargs)

        tmr_loss_dict = None
        if isinstance(model_output, tuple):
            if len(model_output) >= 2 and isinstance(model_output[-1], dict):
                if len(model_output) == 2:
                    model_output, tmr_loss_dict = model_output
                elif len(model_output) == 3:
                    model_output, output2, tmr_loss_dict = model_output
            else:
                model_output = model_output[0]

        weights = self._get_traj_weights(target, model_kwargs)

        time_weights = torch.ones(
            *target.shape,
            device=target.device,
            dtype=target.dtype
        )

        terms = {}

        terms["rot_mse"] = self.masked_l2_weighted(
            target, model_output, mask, weights, time_weights
        )

        if self.lambda_vel > 0.0:
            target_vel = x_start[..., 1:] - x_start[..., :-1]
            model_output_vel = model_output[..., 1:] - model_output[..., :-1]
            terms["vel_mse"] = self.masked_l2(
                target_vel,
                model_output_vel,
                mask[:, :, :, 1:],
            )

        terms["loss"] = terms["rot_mse"] + self.lambda_vel * terms.get("vel_mse", 0.0)

        if tmr_loss_dict is not None:
            terms["tmr_loss"] = tmr_loss_dict.get("tmr_loss", 0.0)
            terms["tmr_acc_t2m"] = tmr_loss_dict.get("acc_t2m", 0.0)
            terms["tmr_acc_m2t"] = tmr_loss_dict.get("acc_m2t", 0.0)
            terms["loss"] = terms["loss"] + terms["tmr_loss"]

        return terms

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
        cond_fn=None,
    ) -> torch.Tensor:
        """Euler-sample [B, C, J, T] from t=0 noise to t=1 data with optional two-stage inpainting."""
        use_guidance = cond_fn is not None

        if device is None:
            device = next(model.parameters()).device
        if model_kwargs is None:
            model_kwargs = {}

        with torch.no_grad():
            if noise is None:
                x = torch.randn(*shape, device=device)
            else:
                x = noise.to(device)

        B = x.shape[0]
        dt = 1.0 / self.num_sampling_steps

        y = model_kwargs.get('y', {})
        inpainted_motion = y.get('inpainted_motion', None)
        inpainting_mask = y.get('inpainting_mask', None)
        impute_until = y.get('impute_until', 0.0)

        inpainted_motion_s2 = y.get('inpainted_motion_second_stage', None)
        inpainting_mask_s2 = y.get('inpainting_mask_second_stage', None)
        impute_until_s2 = y.get('impute_until_second_stage', 0.0)

        inpainting_enabled = y.get('do_inpainting', False)
        do_inpainting = inpainting_enabled and inpainted_motion is not None and inpainting_mask is not None

        inpainted_in_pose_space = y.get('inpainted_in_pose_space', False)

        if do_inpainting:
            inpaint_noise = torch.randn_like(inpainted_motion)
            if inpainted_motion_s2 is not None:
                inpaint_noise_s2 = torch.randn_like(inpainted_motion_s2)
            else:
                inpaint_noise_s2 = None

        N = self.num_sampling_steps
        iterator = range(N)
        if progress:
            iterator = tqdm(iterator, desc="Flow Matching Sampling")

        def get_inpaint_params(t_val):
            """Select stage two after the first cutoff and stage one before it."""
            if not do_inpainting:
                return None, None, None
            if inpainted_motion_s2 is not None and impute_until < t_val <= impute_until_s2:
                mask = inpainting_mask_s2 if inpainting_mask_s2 is not None else inpainting_mask
                noise = inpaint_noise_s2 if inpaint_noise_s2 is not None else inpaint_noise
                return inpainted_motion_s2, mask, noise
            elif t_val <= impute_until:
                return inpainted_motion, inpainting_mask, inpaint_noise
            return None, None, None

        def apply_inpainting(x_curr, next_t_val, step_idx):
            """Inpaint normalized targets, transforming pose-space targets but not Gaussian noise."""
            target, mask, fixed_noise = get_inpaint_params(next_t_val)

            if target is None:
                return x_curr

            if inpainted_in_pose_space and hasattr(self, 'data_transform_fn') and self.data_transform_fn is not None:
                target_normalized = self.data_transform_fn(
                    target.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                target_normalized = target

            noise_for_inpaint = fixed_noise

            next_t_expanded = torch.ones(B, 1, 1, 1, device=device) * next_t_val
            inpainted_x_next = self.get_x_t(target_normalized, noise_for_inpaint, next_t_expanded)

            x_curr = x_curr * (~mask) + inpainted_x_next * mask

            return x_curr

        for i in iterator:
            t_val = i / N
            t = torch.ones(B, device=device) * t_val
            t_expanded = t[:, None, None, None]

            with torch.no_grad():
                x1_pred = model(x, t, **model_kwargs)
                v = self.get_velocity_from_x1(x1_pred, x, t_expanded)

            if use_guidance:
                x = x.detach().requires_grad_(True)
                with torch.enable_grad():
                    grad = cond_fn(x, t, y=y)
                if grad is not None:
                    guidance_scale = (1.0 - t_val)

                    grad = torch.clamp(grad, -1.0, 1.0)
                    v = v + grad.detach() * guidance_scale
                x = x.detach()

            x = x + v * dt

            if do_inpainting:
                next_t_val = (i + 1) / N
                x = apply_inpainting(x, next_t_val, i)

        if do_inpainting and inpainted_motion is not None:
            if inpainted_in_pose_space and hasattr(self, 'data_transform_fn') and self.data_transform_fn is not None:
                target_normalized = self.data_transform_fn(
                    inpainted_motion.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                target_normalized = inpainted_motion

            x_masked = x[inpainting_mask]
            target_masked = target_normalized[inpainting_mask]

            diff = (x_masked - target_masked).abs()
            max_diff = diff.max().item()

            if max_diff > 0.01:
                print(f"[WARNING FlowMatching] Inpainting mismatch! Max diff {max_diff:.6f} > 0.01")
                print(f"  x_masked sample: {x_masked[:5].tolist()}")
                print(f"  target_masked sample: {target_masked[:5].tolist()}")

        return x





def create_flow_matching(
    num_sampling_steps: int = 50,
    sigma_min: float = 0.0,
    traj_extra_weight: float = 1.0,
    drop_redundant: bool = False,
    lambda_vel: float = 0.0,
) -> FlowMatching:
    """Create an x1-predicting FlowMatching instance."""
    return FlowMatching(
        num_sampling_steps=num_sampling_steps,
        sigma_min=sigma_min,
        traj_extra_weight=traj_extra_weight,
        drop_redundant=drop_redundant,
        lambda_vel=lambda_vel,
    )
