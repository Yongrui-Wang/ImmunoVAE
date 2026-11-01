"""Full ImmunoVAE: directional interaction and mask-aware variational branches."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    peptide_length: int = 15
    hla_length: int = 180
    esm_dim: int = 1280
    peptide_latent_dim: int = 64
    hla_latent_dim: int = 32
    dropout: float = 0.2


def masked_mean(values, mask, dim):
    weights = mask.to(values.dtype)
    weights = weights.unsqueeze(1 if dim == 2 else -1)
    return (values * weights).sum(dim) / weights.sum(dim).clamp_min(1)


class FusionEncoder(nn.Module):
    def __init__(self, esm_dim, latent_dim):
        super().__init__()
        # Encoder-local dropout in the existing architecture; interaction dropout is 0.2.
        self.onehot_proj = nn.Sequential(nn.Linear(21, 256), nn.LayerNorm(256),
                                         nn.ELU(), nn.Dropout(0.3))
        self.esm_proj = nn.Sequential(nn.Linear(esm_dim, 256), nn.LayerNorm(256),
                                      nn.ELU(), nn.Dropout(0.3))
        self.fusion_layer = nn.Sequential(nn.Linear(512, 512), nn.ELU(), nn.Dropout(0.3))
        self.mu_layer = nn.Sequential(nn.Linear(64, latent_dim), nn.ELU())
        self.logvar_layer = nn.Sequential(nn.Linear(64, latent_dim), nn.ELU())
        self.conv1 = nn.Sequential(nn.Conv1d(512, 512, 5, padding=2))
        self.conv2 = nn.Sequential(nn.Dropout(0.3), nn.Conv1d(256, 256, 5, padding=2))
        self.conv3 = nn.Sequential(nn.Dropout(0.3), nn.Conv1d(128, 128, 5, padding=2))

    def forward(self, onehot, esm, mask=None):
        fused = self.fusion_layer(torch.cat((self.onehot_proj(onehot), self.esm_proj(esm)), -1))
        if mask is not None:
            fused = fused * mask.unsqueeze(-1)
        x = fused.transpose(1, 2)
        for conv in (self.conv1, self.conv2, self.conv3):
            values, gates = conv(x).chunk(2, 1)
            x = values * gates.sigmoid()
            if mask is not None:
                x = x * mask.unsqueeze(1)
        pooled = masked_mean(x, mask, 2) if mask is not None else x.mean(2)
        return self.mu_layer(pooled), self.logvar_layer(pooled), fused


class ResidueDecoder(nn.Module):
    def __init__(self, latent_dim, length):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_length = length - 12
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, latent_dim * 2, 5), nn.ELU(),
            nn.ConvTranspose1d(latent_dim * 2, latent_dim * 4, 5), nn.ELU(),
            nn.ConvTranspose1d(latent_dim * 4, 128, 5), nn.ELU())
        self.layer = nn.Sequential(nn.Linear(latent_dim, latent_dim * self.initial_length), nn.ELU())
        self.final_projection = nn.Sequential(nn.Linear(128, 64), nn.ELU(),
                                               nn.Dropout(0.1), nn.Linear(64, 21))

    def forward(self, latent):
        x = self.layer(latent).reshape(-1, self.latent_dim, self.initial_length)
        return self.final_projection(self.convt(x).transpose(1, 2))


class CrossInteraction(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.contact_attention = nn.MultiheadAttention(512, 8, dropout=dropout, batch_first=True)
        self.interaction_proj = nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm(512),
                                               nn.ELU(), nn.Dropout(dropout))

    def forward(self, peptide, hla, mask):
        attended, _ = self.contact_attention(peptide, hla, hla, need_weights=False)
        return self.interaction_proj(torch.cat((masked_mean(attended, mask, 1), hla.mean(1)), -1))


def prediction_head():
    layers = []
    for dim_in, dim_out, dropout in zip((608, 256, 128, 64), (256, 128, 64, 32), (.5, .4, .3, .2)):
        layers.extend((nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out), nn.ELU(), nn.Dropout(dropout)))
    layers.append(nn.Linear(32, 1))
    return nn.Sequential(*layers)


class ImmunoVAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if (c.peptide_length, c.hla_length, c.esm_dim, c.peptide_latent_dim, c.hla_latent_dim) != (15, 180, 1280, 64, 32):
            raise ValueError('This release implements the Full 650M model described in the manuscript.')
        self.peptide_encoder = FusionEncoder(c.esm_dim, c.peptide_latent_dim)
        self.hla_encoder = FusionEncoder(c.esm_dim, c.hla_latent_dim)
        self.peptide_decoder = ResidueDecoder(c.peptide_latent_dim, c.peptide_length)
        self.hla_decoder = ResidueDecoder(c.hla_latent_dim, c.hla_length)
        self.cross_interaction = CrossInteraction(c.dropout)
        self.predictor = prediction_head()
        self.im_predictor = prediction_head()

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.exp(.5 * logvar) * torch.randn_like(mu)

    def encode(self, batch):
        mask = batch['mask']
        mp, lp, pf = self.peptide_encoder(batch['peptide_onehot'], batch['peptide_esm'], mask)
        mh, lh, hf = self.hla_encoder(batch['hla_onehot'], batch['hla_esm'])
        interaction = self.cross_interaction(pf, hf, mask)
        return mp, lp, mh, lh, interaction

    def head(self, task):
        if task not in ('el', 'im'):
            raise ValueError('task must be el or im')
        return self.predictor if task == 'el' else self.im_predictor

    def forward(self, batch, task='el'):
        mp, lp, mh, lh, interaction = self.encode(batch)
        zp, zh = self.sample(mp, lp), self.sample(mh, lh)
        return {'logits': self.head(task)(torch.cat((zp, zh, interaction), 1)).squeeze(-1),
                'peptide_reconstruction': self.peptide_decoder(zp) if task == 'el' else None,
                'hla_reconstruction': self.hla_decoder(zh) if task == 'el' else None,
                'mu_peptide': mp, 'logvar_peptide': lp, 'mu_hla': mh, 'logvar_hla': lh}

    @torch.no_grad()
    def posterior_probabilities(self, batch, task='el'):
        if self.training:
            raise RuntimeError('Posterior prediction requires model.eval().')
        mp, lp, mh, lh, interaction = self.encode(batch)
        probabilities = []
        for _ in range(20):
            zp, zh = self.sample(mp, lp), self.sample(mh, lh)
            logits = self.head(task)(torch.cat((zp, zh, interaction), 1)).squeeze(-1)
            probabilities.append(logits.float().sigmoid())
        return torch.stack(probabilities).mean(0)
