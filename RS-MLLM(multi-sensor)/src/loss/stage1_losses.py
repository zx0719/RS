import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage1Loss(nn.Module):
    def __init__(self, lambda_align: float = 0.05):
        super().__init__()
        self.lambda_align = lambda_align

    def forward(
        self,
        lm_loss: torch.Tensor,
        ms_global: torch.Tensor,
        sar_global: torch.Tensor,
    ):
        ms_global = F.normalize(ms_global, dim=-1)
        sar_global = F.normalize(sar_global, dim=-1)

        align_loss = 1.0 - F.cosine_similarity(ms_global, sar_global, dim=-1).mean()
        total_loss = lm_loss + self.lambda_align * align_loss

        return {
            "loss": total_loss,
            "lm_loss": lm_loss.detach(),
            "align_loss": align_loss.detach(),
        }
