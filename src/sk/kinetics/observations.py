"""Observation model of section 6.4: what each assay measures, its noise, and each design.

Rates are per hour. Each gene carries theta = (a, b, c). Three assays enter:

- Whole-cell steady state (GSE264556, four replicates): exonic and intronic counts, negative
  binomial with mean s_j L (u* + m*) for exons and s_j L_i u* for introns. Pre-mRNA carries
  exons too, so exonic reads come from both pools.
- TimeLapse-seq (GSE207924, pulses of 15, 30, 60 and 120 min, two replicates): the deposited
  per-gene fraction new, as a Gaussian observation in probability space with the deposited
  posterior spread scaled by kappa plus an extra standard deviation tau. The modeled fraction
  is that of exonic reads, the read-weighted mix of the pre-mRNA and mature fractions over the
  exons. The 0 min sample measures each gene's background (bg), and the observed fraction is
  bg + (1 - bg) h(t).
- TT-seq (GSE75792, 5 min pulse, two replicates): labeled counts in a gene's exonic and
  intronic bins, negative binomial with mean k_j T (h + eps (1 - h)), where T is the gene's
  normalized total-RNA count in the same bins, k_j the labeled library scale and eps the share
  of unlabeled RNA carried into the pulldown.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import NamedTuple

import torch
from torch import nn

from sk.kinetics.closed_form import region_labeled_fraction

TIMELAPSE_PULSES_H = (0.25, 0.5, 1.0, 2.0)
TTSEQ_PULSE_H = 5.0 / 60.0

DESIGNS: dict[str, tuple[str, ...]] = {
    "SS-E": ("wc_exon",),
    "SS-EI": ("wc_exon", "wc_intron"),
    "KIN": ("wc_exon", "wc_intron", "timelapse", "ttseq"),
    "KIN-1T": ("wc_exon", "wc_intron", "ttseq"),
}


@dataclass
class GeneData:
    """Per-gene observations and covariates; the first axis of every tensor is the gene.

    Lengths are in kb. tt_* tensors have a region axis (exonic bins, intronic bins).
    """

    wc_exon: torch.Tensor  # [G, S]
    wc_intron: torch.Tensor  # [G, S]
    len_exon: torch.Tensor  # [G]
    len_intron: torch.Tensor  # [G]
    tl_y: torch.Tensor  # [G, T, R]
    tl_sd: torch.Tensor  # [G, T, R]
    tl_mask: torch.Tensor  # [G, T, R]
    tl_bg: torch.Tensor  # [G]
    tt_y: torch.Tensor  # [G, 2, R2]
    tt_total: torch.Tensor  # [G, 2]
    tt_mask: torch.Tensor  # [G, 2]
    tt_len_u: torch.Tensor  # [G, 2]
    tt_len_m: torch.Tensor  # [G, 2]

    def __len__(self) -> int:
        return self.wc_exon.shape[0]

    def subset(self, idx: torch.Tensor) -> GeneData:
        return GeneData(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def to(self, dtype: torch.dtype) -> GeneData:
        return GeneData(
            **{
                f.name: (v.to(dtype) if v.is_floating_point() else v)
                for f in fields(self)
                for v in [getattr(self, f.name)]
            }
        )


class NoiseParams(nn.Module):
    """Global assay parameters, shared by all genes and fitted as point estimates."""

    def __init__(self, n_wc: int = 4, n_tt: int = 2, dtype: torch.dtype = torch.float64):
        super().__init__()

        def p(*shape: int, value: float = 0.0) -> nn.Parameter:
            return nn.Parameter(torch.full(shape, value, dtype=dtype))

        self.log_size_factor = p(n_wc)
        self.log_nb_size = nn.ParameterDict(
            {k: p(1, value=2.0) for k in ("wc_exon", "wc_intron", "tt_exon", "tt_intron")}
        )
        self.log_tt_scale = p(n_tt)
        self.logit_tt_contam = p(1, value=-3.0)
        self.log_tl_sd_scale = p(1)
        self.log_tl_extra_sd = p(1, value=-3.0)

    def nb_size(self, key: str) -> torch.Tensor:
        return torch.exp(self.log_nb_size[key])


class Term(NamedTuple):
    """Expected observations of one likelihood term for every gene.

    For "nb", scale is the negative binomial size; for "normal", the standard deviation.
    """

    kind: str
    mean: torch.Tensor  # [..., G, n]
    scale: torch.Tensor  # broadcastable to mean
    y: torch.Tensor  # [G, n]
    mask: torch.Tensor  # [G, n]


def expected(
    theta: torch.Tensor, data: GeneData, noise: NoiseParams, terms: tuple[str, ...]
) -> dict[str, Term]:
    """Expected observations under theta [..., G, 3] for each requested term."""
    a, b, c = theta.unbind(-1)
    out: dict[str, Term] = {}
    g = len(data)
    ones = torch.ones(g, dtype=torch.bool)
    if "wc_exon" in terms:
        log_mu = (torch.log(data.len_exon) + torch.logaddexp(a - b, a - c)).unsqueeze(
            -1
        ) + noise.log_size_factor
        out["wc_exon"] = Term(
            "nb",
            torch.exp(log_mu),
            noise.nb_size("wc_exon"),
            data.wc_exon,
            ones.unsqueeze(-1).expand_as(data.wc_exon),
        )
    if "wc_intron" in terms:
        log_mu = (torch.log(data.len_intron) + a - b).unsqueeze(-1) + noise.log_size_factor
        out["wc_intron"] = Term(
            "nb",
            torch.exp(log_mu),
            noise.nb_size("wc_intron"),
            data.wc_intron,
            ones.unsqueeze(-1).expand_as(data.wc_intron),
        )
    if "timelapse" in terms:
        t = torch.tensor(TIMELAPSE_PULSES_H, dtype=theta.dtype)
        len_exon = data.len_exon.unsqueeze(-1)
        h = region_labeled_fraction(b.unsqueeze(-1), c.unsqueeze(-1), t, len_exon, len_exon)
        bg = data.tl_bg.unsqueeze(-1)
        f = bg + (1 - bg) * h  # [..., G, T]
        n_rep = data.tl_y.shape[-1]
        mean = f.unsqueeze(-1).expand(*f.shape, n_rep).flatten(-2)
        sd = torch.sqrt(
            (torch.exp(noise.log_tl_sd_scale) * data.tl_sd) ** 2
            + torch.exp(2 * noise.log_tl_extra_sd)
        ).flatten(-2)
        out["timelapse"] = Term("normal", mean, sd, data.tl_y.flatten(-2), data.tl_mask.flatten(-2))
    if "ttseq" in terms:
        h = region_labeled_fraction(
            b.unsqueeze(-1),
            c.unsqueeze(-1),
            torch.tensor(TTSEQ_PULSE_H, dtype=theta.dtype),
            data.tt_len_u,
            data.tt_len_m,
        )  # [..., G, 2]
        eps = torch.sigmoid(noise.logit_tt_contam)
        captured = h + eps * (1 - h)
        mu = (data.tt_total * captured).unsqueeze(-1) * torch.exp(noise.log_tt_scale)
        size = torch.cat([noise.nb_size("tt_exon"), noise.nb_size("tt_intron")]).unsqueeze(
            -1
        )  # [2, 1]
        n_rep = data.tt_y.shape[-1]
        out["ttseq"] = Term(
            "nb",
            mu.flatten(-2),
            size.expand(2, n_rep).flatten(),
            data.tt_y.flatten(-2),
            data.tt_mask.unsqueeze(-1).expand_as(data.tt_y).flatten(-2),
        )
    return out


def _nb_log_prob(y: torch.Tensor, mu: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    log_mu_r = torch.log(mu + size)
    return (
        torch.lgamma(y + size)
        - torch.lgamma(size)
        - torch.lgamma(y + 1)
        + size * (torch.log(size) - log_mu_r)
        + y * (torch.log(mu) - log_mu_r)
    )


def log_likelihood(
    theta: torch.Tensor, data: GeneData, noise: NoiseParams, terms: tuple[str, ...]
) -> torch.Tensor:
    """log p(y_g | theta_g) summed over the design's terms, shape [..., G]."""
    total = torch.zeros(theta.shape[:-1], dtype=theta.dtype)
    for term in expected(theta, data, noise, terms).values():
        if term.kind == "nb":
            lp = _nb_log_prob(term.y, term.mean, term.scale)
        else:
            lp = torch.distributions.Normal(term.mean, term.scale).log_prob(term.y)
        total = total + torch.where(term.mask, lp, torch.zeros_like(lp)).sum(-1)
    return total


@torch.no_grad()
def simulate(theta: torch.Tensor, data: GeneData, noise: NoiseParams) -> GeneData:
    """New observations for every term, drawn at theta [G, 3] with data's covariates.

    Draws use the global torch generator, which the caller seeds.
    """
    sims = {}
    for name, term in expected(theta, data, noise, tuple(DESIGNS["KIN"])).items():
        if term.kind == "nb":
            size = term.scale.expand_as(term.mean)
            y = torch.distributions.NegativeBinomial(
                total_count=size, logits=torch.log(term.mean) - torch.log(size)
            ).sample()
        else:
            y = torch.distributions.Normal(term.mean, term.scale).sample()
        sims[name] = y
    return replace(
        data,
        wc_exon=sims["wc_exon"],
        wc_intron=sims["wc_intron"],
        tl_y=sims["timelapse"].reshape(data.tl_y.shape),
        tt_y=sims["ttseq"].reshape(data.tt_y.shape),
    )
