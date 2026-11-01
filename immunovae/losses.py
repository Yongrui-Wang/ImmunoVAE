"""Objectives as defined in the current manuscript and Supplementary Methods."""
import torch
from torch.nn import functional as F


def focal_loss(logits, labels):
    logits, labels = logits.float(), labels.float()
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    p_target = torch.exp(-bce)
    return (.7 * (1 - p_target).pow(1.3) * bce).mean()


def branch_kl(mu, logvar):
    mu, logvar = mu.float(), logvar.float()
    return .5 * (mu.square() + logvar.exp() - logvar - 1).sum(-1).mean()


def objective(output, batch, task):
    classification = focal_loss(output['logits'], batch['label'])
    if task == 'im':
        return {'total': classification, 'classification': classification}
    if task != 'el':
        raise ValueError('task must be el or im')
    valid = batch['mask'].bool()
    peptide = F.cross_entropy(output['peptide_reconstruction'][valid].float(),
                              batch['peptide_target'][valid])
    hla = F.cross_entropy(output['hla_reconstruction'].float().flatten(0, 1),
                          batch['hla_target'].flatten())
    kp = branch_kl(output['mu_peptide'], output['logvar_peptide'])
    kh = branch_kl(output['mu_hla'], output['logvar_hla'])
    total = classification + 1e-3 * (3 * peptide + .5 * hla + .2 * (.2 * kp + kh))
    return {'total': total, 'classification': classification, 'peptide_reconstruction': peptide,
            'hla_reconstruction': hla, 'peptide_kl': kp, 'hla_kl': kh}
