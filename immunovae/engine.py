"""EL/IM training with immutable per-epoch checkpoints and validation-AUROC selection."""
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Sampler

from .data import FeatureStore, PairDataset, check_development_split, read_pairs, resolve_table_path
from .losses import objective
from .model import ImmunoVAE
from .utils import (atomic_json, evaluation_rng, atomic_torch_save,
                    seed_everything, sha256, load_training_checkpoint, selected_checkpoint, checkpoint_rng_state)


@dataclass(frozen=True)
class TrainConfig:
    task: str
    seed: int = 42
    strategy: str = 'fine-tuned'
    epochs: int = 50

    def __post_init__(self):
        if self.task not in ('el', 'im') or self.seed not in (42, 2026, 2027):
            raise ValueError('Use task el/im and seed 42, 2026 or 2027.')
        if self.strategy not in ('fine-tuned', 'frozen', 'scratch') or self.epochs < 1:
            raise ValueError('Invalid strategy or epoch count.')
        if self.task == 'el' and self.strategy != 'fine-tuned':
            raise ValueError('EL pretraining does not use an IM transfer strategy.')

    @property
    def batch_size(self):
        return 256 if self.task == 'el' else 128


def protocol(config):
    el = config.task == 'el'
    return dict(asdict(config), batch_size=config.batch_size, optimizer='AdamW', weight_decay=1e-4,
                early_stopping=False, selection_metric='validation AUROC',
                checkpoint_policy='every_epoch_immutable_and_validation_AUROC_best',
                learning_rate={'warmup_start': 2e-5, 'peak': 2e-4} if el else
                              {'backbone': 3e-5, 'head': 1e-4},
                scheduler={'name': 'ReduceLROnPlateau', 'mode': 'max', 'factor': .5,
                           'patience': 3, 'cooldown': 1 if el else 0,
                           'min_lr': 1e-6 if el else 1e-7},
                inference={'draws': 20, 'seed': 91020, 'average': 'sigmoid probabilities'})


class TrainingBatches(Sampler):
    """Shuffle all rows; merge a singleton tail to keep BatchNorm valid."""
    def __init__(self, size, batch_size, generator, shard_ids=None):
        if size < 2:
            raise ValueError('Training requires at least two examples.')
        self.size, self.batch_size, self.generator = size, batch_size, generator
        grouped = {}
        for row, shard in enumerate(shard_ids if shard_ids is not None else [0] * size):
            grouped.setdefault(shard, []).append(row)
        self.shards = list(grouped.values())
        if sum(map(len, self.shards)) != size:
            raise ValueError('shard_id length does not match training rows.')

    def __iter__(self):
        batches = []
        for shard in self.shards:
            order = [shard[i] for i in torch.randperm(len(shard), generator=self.generator).tolist()]
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                if len(batch) == 1 and batches:
                    batches[-1].extend(batch)
                else:
                    batches.append(batch)
        if len(batches[0]) == 1 and len(batches) > 1:
            batches[1].extend(batches.pop(0))
        return iter(batches)

    def __len__(self):
        sizes = []
        for shard in self.shards:
            for start in range(0, len(shard), self.batch_size):
                size = min(self.batch_size, len(shard) - start)
                if size == 1 and sizes:
                    sizes[-1] += 1
                else:
                    sizes.append(size)
        return len(sizes) - int(len(sizes) > 1 and sizes[0] == 1)


def to_device(batch, device):
    return {key: tensor.to(device) for key, tensor in batch.items()}


def initialize_model(config, pretrained=None, device='cpu'):
    model = ImmunoVAE()
    if config.task == 'im' and config.strategy != 'scratch':
        if pretrained is None:
            raise ValueError('EL checkpoint required for fine-tuned or frozen IM transfer.')
        state = load_training_checkpoint(pretrained)
        if state['config']['task'] != 'el' or state['model_config'] != asdict(model.config):
            raise ValueError('Transfer requires a matching EL checkpoint from this release.')
        # The IM head is always newly initialized, never copied from the EL file.
        backbone = {key: value for key, value in state['model'].items() if not key.startswith('im_predictor.')}
        missing, unexpected = model.load_state_dict(backbone, strict=False)
        if unexpected or any(not key.startswith('im_predictor.') for key in missing):
            raise ValueError('Incompatible EL backbone state.')
    elif pretrained is not None:
        raise ValueError('A pretrained checkpoint is only accepted for IM transfer.')
    if config.task == 'im':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 73000)
            for module in model.im_predictor.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if config.task == 'el':
        modules = [model.peptide_encoder, model.hla_encoder, model.cross_interaction,
                   model.peptide_decoder, model.hla_decoder, model.predictor]
    else:
        modules = [model.im_predictor]
        if config.strategy != 'frozen':
            modules += [model.peptide_encoder, model.hla_encoder, model.cross_interaction]
    for module in modules:
        module.requires_grad_(True)
    return model.to(device)


def training_mode(model, config):
    model.train()
    if config.task == 'im' and config.strategy == 'frozen':
        model.eval()
        model.im_predictor.train()


def make_optimizer(model, config):
    if config.task == 'el':
        groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 2e-4}]
    elif config.strategy == 'frozen':
        groups = [{'params': model.im_predictor.parameters(), 'lr': 1e-4}]
    else:
        groups = [{'params': module.parameters(), 'lr': 3e-5} for module in
                  (model.peptide_encoder, model.hla_encoder, model.cross_interaction)]
        groups.append({'params': model.im_predictor.parameters(), 'lr': 1e-4})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


@torch.no_grad()
def predict(model, dataset, task, device='cpu', batch_size=128):
    scores = []
    modes = [(module, module.training) for module in model.modules()]
    try:
        with evaluation_rng():
            model.eval()
            generator = torch.Generator().manual_seed(91020)
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=generator):
                probabilities = model.posterior_probabilities(to_device(batch, device), task)
                scores.extend(probabilities.cpu().tolist())
    finally:
        # Restore a possibly mixed Frozen configuration without recursive train() calls.
        for module, training in modes:
            module.training = training
    return np.asarray(scores)


def validation_metrics(labels, scores):
    return {'AUROC': float(roc_auc_score(labels, scores)),
            'AP': float(average_precision_score(labels, scores))}


def validation_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def train(config, train_csv, valid_csv, features, output, pretrained=None, device='cpu'):
    """Only train and validation tables are accepted; no external-test argument exists."""
    train_csv, valid_csv = resolve_table_path(train_csv), resolve_table_path(valid_csv)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    train_frame, valid_frame = read_pairs(train_csv), read_pairs(valid_csv)
    check_development_split(train_frame, valid_frame)
    store = FeatureStore(features)
    training, validation = PairDataset(train_frame, store), PairDataset(valid_frame, store)
    provenance = {'train_csv_sha256': sha256(train_csv), 'valid_csv_sha256': sha256(valid_csv),
                  'feature_index_sha256': sha256(store.index_path),
                  'pretrained_sha256': sha256(pretrained) if pretrained else None,
                  'code_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                  'sampling': 'shuffled within supplied shards (or the single input table); no allele resampling; singleton tail merged',
                  'train_rows': len(training), 'valid_rows': len(validation),
                  'torch_version': str(torch.__version__),
                  'numpy_version': str(np.__version__), 'pandas_version': str(pd.__version__),
                  'device_type': torch.device(device).type}
    current_protocol = protocol(config)
    if pretrained is not None:
        selected_checkpoint(pretrained)
    model = initialize_model(config, pretrained, device)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=.5, patience=3,
        cooldown=1 if config.task == 'el' else 0, min_lr=1e-6 if config.task == 'el' else 1e-7)
    use_amp = torch.device(device).type == 'cuda'
    scaler = (torch.amp.GradScaler('cuda', enabled=use_amp)
              if hasattr(torch.amp, 'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=use_amp))
    generator = torch.Generator().manual_seed(config.seed + 44000)
    batches = TrainingBatches(len(training), config.batch_size, generator,
                              train_frame.shard_id.tolist() if 'shard_id' in train_frame else None)
    loader = DataLoader(training, batch_sampler=batches, generator=generator)
    best_auc, history, best = float('-inf'), [], None
    atomic_json(output / 'config.json', current_protocol)
    atomic_json(output / 'provenance.json', provenance)
    for epoch in range(1, config.epochs + 1):
        training_mode(model, config)
        totals, count = {}, 0
        if config.task == 'el' and epoch == 1:
            for group in optimizer.param_groups:
                group['lr'] = 2e-5 if len(loader) > 1 else 2e-4
        learning_rates = [group['lr'] for group in optimizer.param_groups]
        for step, batch in enumerate(loader):
            if config.task == 'el' and epoch == 1:
                warmup = (2e-5 + (2e-4 - 2e-5) * step / (len(loader) - 1)) if len(loader) > 1 else 2e-4
                for group in optimizer.param_groups:
                    group['lr'] = warmup
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with (torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()):
                result = model(batch, config.task)
            losses = objective(result, batch, config.task)
            if not all(torch.isfinite(loss).all() for loss in losses.values()):
                raise FloatingPointError(f'Non-finite objective at epoch {epoch}, step {step}.')
            scaler.scale(losses['total']).backward()
            scaler.step(optimizer)
            scaler.update()
            size = len(batch['label'])
            count += size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + value.detach().item() * size
        scores = predict(model, validation, config.task, device, config.batch_size)
        metrics = validation_metrics(valid_frame.label, scores)
        improved = metrics['AUROC'] > best_auc
        checkpoint = 'best.pt'
        if improved:
            best_auc = metrics['AUROC']
            best = {'checkpoint': checkpoint, 'epoch': epoch, 'validation': metrics,
                    'threshold': .5 if config.task == 'el' else validation_threshold(valid_frame.label, scores)}
        scheduler.step(metrics['AUROC'])
        history.append({'epoch': epoch, 'train': {k: v / count for k, v in totals.items()},
                        'validation': metrics, 'learning_rates_at_epoch_start': learning_rates,
                        'learning_rates_next_epoch': [g['lr'] for g in optimizer.param_groups]})
        payload = {'epoch': epoch, 'model': model.state_dict(),
                   'model_config': asdict(model.config), 'config': current_protocol,
                   'provenance': provenance, 'train_metrics': history[-1]['train'],
                   'validation_metrics': metrics, 'best': dict(best),
                   'optimizer_state': optimizer.state_dict(),
                   'scheduler_state': scheduler.state_dict(),
                   'scaler_state': scaler.state_dict(),
                   'rng_state': checkpoint_rng_state(),
                   'sampling_generator_state': generator.get_state()}
        atomic_torch_save(output / 'checkpoints' / f'epoch_{epoch:03d}.pt', payload)
        if improved:
            atomic_torch_save(output / 'best.pt', payload, overwrite=True)
        atomic_json(output / 'best.json', best)
        atomic_json(output / 'history.json', history)
        print(f'Epoch {epoch:03d}: validation AUROC={metrics["AUROC"]:.6f}, AP={metrics["AP"]:.6f}', flush=True)
    atomic_json(output / 'complete.json', {'epochs_completed': config.epochs,
                'selected_checkpoint': best['checkpoint'], 'selection_metric': 'validation AUROC',
                'selected_checkpoint_sha256': sha256(output / best['checkpoint'])})
    return best
