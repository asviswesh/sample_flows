import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_utils import *

class RealNVPCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, reverse_simple_mask, mask_type, device):
        super(RealNVPCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reverse_simple_mask = reverse_simple_mask
        self.mask_type = mask_type
        self.device = device

        self.scale_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),  
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),  
            nn.Linear(self.hidden_dim, self.input_dim)
        )

        self.translate_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),  
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),  
            nn.Linear(self.hidden_dim, self.input_dim)
        )

        self.rescale = nn.utils.parametrizations.weight_norm(DataRescale(input_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, log_det_jacobian=None, reverse=True):
        batch_size, width = x.size()
        if self.mask_type == "checkerboard":
            mask = create_checkerboard_mask(batch_size, width).to(self.device)
        elif self.mask_type == "simple":
            mask = create_half_split_mask(batch_size, width, self.reverse_simple_mask).to(self.device)
        x_masked = apply_mask(x, mask) 
        s = self.scale_net(x_masked)
        t = self.translate_net(x_masked) 
        s = self.rescale(torch.tanh(s))
        s = apply_mask(s, 1 - mask)
        t = apply_mask(t, 1 - mask) 

        if reverse: 
            y = x * torch.exp(s * -1) - t
        else:
            y = x * torch.exp(s) + t 
            if log_det_jacobian is not None:  
                log_det_jacobian += s.view(s.size(0), -1).sum(dim=1)
        return y, log_det_jacobian

class RealFlowNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, num_blocks=8):
        super(RealFlowNVP, self).__init__()
        self.num_scales = 2
        self.device = device
        self.flows = _RealNVP(0, self.num_scales, input_dim, hidden_dim, num_blocks, device)

    def _preprocess(self, x):
        x = normalize_all_columns(x)
        ldj = F.softplus(x) + F.softplus(-x)
        sldj = ldj.view(ldj.size(0), -1).sum(-1)
        return x, sldj
    
    def forward(self, x, reverse=False, **kwargs):
        log_det_jacobian = None
        if not reverse:
            x = normalize_all_columns(x)  
            x, log_det_jacobian = self._preprocess(x)
        x, sldj = self.flows(x, log_det_jacobian, reverse)
        return x, sldj

    def inverse(self, z, **kwargs):
        return self.forward(z, reverse=True)

class _RealNVP(nn.Module):
    def __init__(self, scale_idx, num_scales, input_dim, hidden_dim, num_blocks, device):
        super(_RealNVP, self).__init__()

        self.is_last_block = (scale_idx == num_scales - 1)

        self.in_couplings = nn.ModuleList(
            [
                RealNVPCouplingLayer(
                    input_dim,
                    hidden_dim,
                    reverse_simple_mask=False,
                    mask_type='checkerboard',
                    device=device
                ),
                RealNVPCouplingLayer(
                    input_dim,
                    hidden_dim,
                    reverse_simple_mask=True,
                    mask_type='checkerboard',
                    device=device
                ),
                RealNVPCouplingLayer(
                    input_dim,
                    hidden_dim,
                    reverse_simple_mask=False,
                    mask_type='checkerboard',
                    device=device
                ),
            ]
        )

        if self.is_last_block:
            self.in_couplings.append(
                RealNVPCouplingLayer(
                    input_dim,
                    hidden_dim,
                    reverse_simple_mask=True,
                    mask_type='checkerboard',
                    device=device
                )
            )
        else:
            self.out_couplings = nn.ModuleList(
                [
                    RealNVPCouplingLayer(
                        input_dim,
                        hidden_dim,
                        reverse_simple_mask=False,
                        mask_type='simple',
                        device=device
                    ),
                    RealNVPCouplingLayer(
                        input_dim,
                        hidden_dim,
                        reverse_simple_mask=True,
                        mask_type='simple',
                        device=device
                    ),
                    RealNVPCouplingLayer(
                        input_dim,
                        hidden_dim,
                        reverse_simple_mask=False,
                        mask_type='simple',
                        device=device
                    ),
                ]
            )
            self.next_block = _RealNVP(
                scale_idx + 1, num_scales, input_dim, hidden_dim, num_blocks, device
            )
    
    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                x = x.view(x.size(0), -1) 
                x, sldj = self.next_block(x, sldj, reverse)
                x = x.view(x.size(0), -1) 

                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = x.view(x.size(0), -1) 
                x, sldj = self.next_block(x, sldj, reverse)
                x = x.view(x.size(0), -1)  

        return x, sldj
