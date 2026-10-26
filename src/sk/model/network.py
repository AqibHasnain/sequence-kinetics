"""Sequence network (section 6.2) with full-covariance Gaussian posterior heads.

A convolutional tower shared by both segments (residual blocks with dilated convolutions and
max pooling, as in Basenji and Saluki), attention pooling within each segment, concatenation
with the gene covariates, and a head that outputs the mean and a 3 x 3 Cholesky factor of
q(a, b, c | sequence). Both pooled segments reach every output, so no rate is tied to a segment
by architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sk.model.inputs import FRAME, JUNCTION, PROMOTER_LEN

N_CHANNELS_IN = 7  # ACGT one-hot, codon frame, splice junction, segment identity
_TRIL = torch.tril_indices(3, 3, offset=-1)


@dataclass
class NetConfig:
    n_covariates: int
    stem_channels: int = 48
    channels: tuple[int, ...] = (64, 64, 96, 96, 128, 128, 128)
    kernel: int = 5
    hidden: int = 128
    dropout: float = 0.1


def one_hot_inputs(codes: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """[B, L] uint8 codes and flags -> [B, 7, L] float input."""
    x = F.one_hot(codes.long(), 6)[..., :4].float()
    f = flags.long()
    frame = (f & FRAME).clamp(max=1).float()
    junction = ((f & JUNCTION) >> 1).float()
    region = torch.zeros_like(frame)
    region[:, PROMOTER_LEN:] = 1.0
    return torch.cat([x, frame[..., None], junction[..., None], region[..., None]], -1).transpose(
        1, 2
    )


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(c_out)
        self.norm2 = nn.BatchNorm1d(c_out)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class AttentionPool(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.score = nn.Conv1d(c, 1, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(h).squeeze(1).masked_fill(~mask, -1e4)
        w = torch.softmax(logits, -1)
        return torch.einsum("bcl,bl->bc", h, w)


class SequenceNet(nn.Module):
    def __init__(self, cfg: NetConfig, prior_mean: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Conv1d(N_CHANNELS_IN, cfg.stem_channels, 15, padding=7)
        blocks, c = [], cfg.stem_channels
        for i, c_out in enumerate(cfg.channels):
            blocks.append(ResBlock(c, c_out, cfg.kernel, dilation=2 ** (i % 3 + 1)))
            c = c_out
        self.blocks = nn.ModuleList(blocks)
        self.pool_factor = 2 ** len(cfg.channels)
        self.attn_promoter = AttentionPool(c)
        self.attn_transcript = AttentionPool(c)
        self.cov = nn.Sequential(nn.Linear(cfg.n_covariates, 32), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(2 * c + 32, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(cfg.hidden, 9)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.zero_()
            self.head.bias[:3] = prior_mean.float()
            self.head.bias[3:6] = -1.0

    def forward(
        self, codes: torch.Tensor, flags: torch.Tensor, covariates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean [B, 3] and Cholesky factor [B, 3, 3]."""
        x = one_hot_inputs(codes, flags)
        valid = codes != 5
        h = F.gelu(self.stem(x))
        for block in self.blocks:
            h = F.max_pool1d(block(h), 2)
        valid = F.max_pool1d(valid.float()[:, None], self.pool_factor)[:, 0] > 0
        split = PROMOTER_LEN // self.pool_factor
        pooled = torch.cat(
            [
                self.attn_promoter(h[..., :split], valid[:, :split]),
                self.attn_transcript(h[..., split:], valid[:, split:]),
                self.cov(covariates),
            ],
            -1,
        )
        out = self.head(self.mlp(pooled))
        mean = out[:, :3]
        log_diag = out[:, 3:6].clamp(-7.0, 3.0)
        tril = torch.diag_embed(torch.exp(log_diag))
        lower = torch.zeros_like(tril)
        lower[:, _TRIL[0], _TRIL[1]] = out[:, 6:9]
        return mean, tril + lower
