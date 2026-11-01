"""Download ESM-2 on demand and incrementally cache frozen residue embeddings."""
import json
from pathlib import Path

from filelock import FileLock
import pandas as pd
import torch

from .data import FeatureStore, read_pairs, resolve_table_path
from .utils import atomic_json, atomic_torch_save, sha256


def build_features(tables, output, device='cpu', batch_size=8):
    if batch_size < 1:
        raise ValueError('ESM batch size must be positive.')
    tables = [resolve_table_path(path) for path in tables]
    frames = [read_pairs(path, require_labels=False) for path in tables]
    sequences = {branch: sorted(set().union(*(set(f[column]) for f in frames)))
                 for branch, column in [('peptides', 'peptide'), ('hlas', 'hla_sequence')]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.cache.lock')):
        index_path = output / 'index.json'
        if index_path.exists():
            index = FeatureStore(output).index
        else:
            index = {'checkpoint': 'esm2_t33_650M_UR50D', 'layer': 33,
                     'dimension': 1280, 'dtype': 'float16',
                     'peptides': {}, 'hlas': {}, 'shards': {}, 'input_sha256': {}}
        missing = {branch: [s for s in seqs if s not in index[branch]]
                   for branch, seqs in sequences.items()}
        if not any(missing.values()):
            print(f'Reusing ESM-2 features: {output}', flush=True)
            return output
        print(f'Preparing ESM-2 features: {len(missing["peptides"])} peptides, '
              f'{len(missing["hlas"])} HLA sequences.', flush=True)
        import esm
        if not hasattr(esm, 'pretrained'):
            raise ImportError('Use fair-esm from environment.yml for ESM-2 feature extraction.')
        # The official loader downloads absent weights and reuses the torch hub cache.
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval().to(device).requires_grad_(False)
        converter = alphabet.get_batch_converter()
        try:
            for branch, maximum in [('peptides', 15), ('hlas', 180)]:
                seqs = missing[branch]
                existing = [int(p.stem.rsplit('_', 1)[1])
                            for p in output.glob(branch + '_[0-9]*.pt')]
                next_shard = max(existing, default=-1) + 1
                for start in range(0, len(seqs), 256):
                    group = seqs[start:start + 256]
                    storage = torch.zeros(len(group), maximum, 1280, dtype=torch.float16)
                    with torch.inference_mode():
                        for first in range(0, len(group), batch_size):
                            batch = group[first:first + batch_size]
                            _, _, tokens = converter([(str(i), s) for i, s in enumerate(batch)])
                            values = model(tokens.to(device), repr_layers=[33],
                                           return_contacts=False)['representations'][33]
                            for row, sequence in enumerate(batch):
                                storage[first + row, :len(sequence)] = values[row, 1:1 + len(sequence)].cpu().half()
                    name = f'{branch}_{next_shard:05d}.pt'
                    next_shard += 1
                    atomic_torch_save(output / name, storage)
                    index['shards'][name] = sha256(output / name)
                    for row, sequence in enumerate(group):
                        index[branch][sequence] = [name, row]
                    atomic_json(index_path, index)
                    print(f'{branch}: {min(start + len(group), len(seqs))}/{len(seqs)} cached', flush=True)
            index.setdefault('input_sha256', {}).update({str(p): sha256(p) for p in tables})
            atomic_json(index_path, index)
        finally:
            del model
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
    return output
