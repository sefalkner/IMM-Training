# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from training.unets import *  
from training.dit import *  
from training.preconds import *  
 
from einops import rearrange

class IMMPrecondTraining(IMMPrecond):

    def __init__(
        self,
        *args, 
        **kwargs,  
    ):
        super().__init__(*args, **kwargs)
    
        assert self.noise_schedule == "fm" 
        assert self.f_type == "euler_fm" 

    def get_eta_t(self, t):
        """ Returns eta = sigma_t/alpha_t for the calculation of the r(s,t) mapping function."""

        if self.noise_schedule == "fm":
            eta_t = t/(1-t)
        else:
            raise NotImplementedError

        return eta_t
    
    def get_inv_eta_t(self, t):
        """ Returns the inverse of eta(t) for the calculation of the r(s,t) mapping function."""

        if self.noise_schedule == "fm":
            inv_eta_t = t / (1 + t)
        else:
            raise NotImplementedError

        return inv_eta_t
    
    def get_ddt_log_snr(self, t):
        """ Returns the derivative of the log SNR for the calculation of w(s,t) in Eq. (13) main text of https://arxiv.org/pdf/2503.07565"""

        if self.noise_schedule == "fm":
            ddt_log_snr = -2 / (t*(1-t))
        else:
            raise NotImplementedError
        
        return ddt_log_snr
    

    def get_r_ts(self, t, s, k=12):
        """ Calculates the mapping function r(s,t) assuming a constant decrement in eta(t) (see section C.6. in the SI of https://arxiv.org/pdf/2503.07565)"""

        eta_t = self.get_eta_t(t)

        eps = (self.nt_high - self.nt_low) / 2**k
        r_trial =  self.get_inv_eta_t(eta_t - eps)

        # Sort out r values smaller than s, these don't make sense
        r = torch.where(s > r_trial, s, r_trial)

        return r
    

    def get_c_out(self, t, s):
        """ Calculates c_out used in w_tilde(s,t) in the kernel function (see section C.8. in the SI of https://arxiv.org/pdf/2503.07565)
        
        WARNING: This is not specified explicitly in the text, not sure if there should also be a sigma_d! 
                 However, this would just scale the final loss by exp(-sigma_d), should not change too much.
        
        """

        if self.noise_schedule == "fm":
            c_out = t-s
        else:
            raise NotImplementedError
        
        return c_out
    
    
    def kernel_func(self, dist, c_out, dim, eps=1e-8):
        """Applies the kernel function to a distance, required for the MMD calculation. (see section C.8. in the SI of https://arxiv.org/pdf/2503.07565)"""
        
        d_reg = torch.where(dist < eps, dist.new_ones(1,) * eps, dist)
        
        return torch.exp(-(1/c_out) * d_reg / dim)
    

    def get_weight_st(self, t, s, a=1, b=4):
        """ Returns the w(s,t) for the MMD calculation specified in Eq. (13) main text of https://arxiv.org/pdf/2503.07565"""
        
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        lamda = self.get_logsnr(t)

        ddt_log_snr = self.get_ddt_log_snr(t)
        weight = 0.5 * torch.sigmoid(b - lamda) * (-ddt_log_snr) * alpha_t**a / (alpha_t**2 + sigma_t**2)

        return weight
    

    def MMD(self, f_st, f_sr, t, s):
        """ Returns the MMD loss specified in Eq. (67) SI of https://arxiv.org/pdf/2503.07565"""

        # cout needed for w_tilde(s, t)
        c_out = self.get_c_out(t, s)

        # Get distance matrix (contains distance from every image to every other)
        f_st_f_st_dist = torch.cdist(f_st, f_st)
        f_sr_f_sr_dist = torch.cdist(f_sr, f_sr)
        f_st_f_sr_dist = torch.cdist(f_st, f_sr)

        # Apply kernel function
        k_st = self.kernel_func(f_st_f_st_dist, c_out, f_st.shape[-1])
        k_sr = self.kernel_func(f_sr_f_sr_dist, c_out, f_sr.shape[-1] )
        k_stsr = self.kernel_func(f_st_f_sr_dist, c_out, f_sr.shape[-1])

        MMD_loss = k_st + k_sr - 2*k_stsr

        weight = self.get_weight_st(t, s)

        MMD_loss = MMD_loss * weight

        return MMD_loss.mean()


    def loss(self, x, class_labels=None, group_size=4, ema_model=None, force_fp32=False, **model_kwargs):
        """ Trains the model on the MMD loss specified in Eq. (67) SI of https://arxiv.org/pdf/2503.07565"""
        
        assert x.shape[0] % group_size == 0, "Batch size must be divisible by group size."
        
        n_groups = x.shape[0] // group_size

        # Sample s, t and r in groups, repeat within each group
        t = x.new_empty((n_groups,)).uniform_(self.eps, self.T)
        s = x.new_empty((n_groups,)).uniform_(0, 1) * t

        r = self.get_r_ts(t, s)
        
        srt = torch.cat([s.unsqueeze(1),r.unsqueeze(1),t.unsqueeze(1)], dim=1).repeat(1, group_size).reshape(-1, 3)
        s, r, t = srt.chunk(chunks=3, dim=-1)

        # Add some dimensions to fit to BCHW images
        s = s.reshape(-1,1,1,1)
        r = r.reshape(-1,1,1,1)
        t = t.reshape(-1,1,1,1)

        # Add noise, get x_r from x_t
        x_t, noise = self.add_noise(x, t)
        x_r = self.ddim(x_t, x, t, r)

        # Get denoised sample from t to s
        f_st = self.forward(x_t, t=t, s=s, class_labels=class_labels, force_fp32=force_fp32, model_kwargs=model_kwargs)

        # Get denoised sample from r to s, optionally coming from an EMA model
        with torch.no_grad():
            f_minus_model = ema_model if ema_model is not None else self.forward
            f_sr = f_minus_model(x_r, t=r, s=s, class_labels=class_labels, force_fp32=force_fp32, model_kwargs=model_kwargs)

        # Reshape images into groups
        f_st = rearrange(f_st, "(n m) c h w -> n m (c h w)", n=n_groups, m=group_size)
        f_sr = rearrange(f_sr, "(n m) c h w -> n m (c h w)", n=n_groups, m=group_size)

        # Reshape s,r,t into groups
        s = rearrange(s.squeeze(), "(n m)-> n m", n=n_groups, m=group_size).unsqueeze(-1)
        r = rearrange(r.squeeze(), "(n m)-> n m", n=n_groups, m=group_size).unsqueeze(-1)
        t = rearrange(t.squeeze(), "(n m)-> n m", n=n_groups, m=group_size).unsqueeze(-1)

        # Here we are
        MMD_loss = self.MMD(f_st, f_sr, t, s)

        return MMD_loss