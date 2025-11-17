import torch
import torch.nn as nn

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss(nn.Module):
    
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
    def tensor2freq(self, x):
        
        x=x.float()
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(x,dim=1, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(x, 1, normalized=True)
        return freq

    

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
            Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)#32x512x2
        target_freq = self.tensor2freq(target)

        
        diff = (pred_freq - target_freq).pow(2).sum(-1)  # [32, 512]
        weight = (diff.sqrt() ** self.alpha).detach()
        weight = weight / (weight.max(dim=1, keepdim=True)[0] + 1e-8) 
        
        # frequency weight matrix
        loss = weight * diff
        return loss.mean() * self.loss_weight
