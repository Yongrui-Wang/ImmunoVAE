"""Explicit pair tables and lazily loaded, sequence-indexed ESM-2 caches."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import load_tensors, sha256

AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV'
PAD = 20


def resolve_table_path(path):
    """Use a CSV directly, or its gzip-compressed counterpart when supplied."""
    path = Path(path)
    if not path.exists() and path.suffix == '.csv':
        compressed = Path(str(path) + '.gz')
        if compressed.is_file():
            return compressed
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_pairs(path, require_labels=True):
    frame = pd.read_csv(resolve_table_path(path), dtype={'sample_id': str, 'peptide': str, 'hla_sequence': str})
    required = {'sample_id', 'peptide', 'hla_sequence'} | ({'label'} if require_labels else set())
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    if frame.empty or frame[list(required)].isnull().any().any():
        raise ValueError('Pair tables must be nonempty and have no missing input values.')
    if frame.sample_id.duplicated().any():
        raise ValueError('sample_id must be unique within each table.')
    for peptide, hla in zip(frame.peptide, frame.hla_sequence):
        if not 1 <= len(peptide) <= 15 or len(hla) != 180:
            raise ValueError('Expected peptides of 1–15 residues and exactly 180 HLA residues.')
        if not set(peptide + hla) <= set(AMINO_ACIDS):
            raise ValueError('Only the 20 canonical amino acids are supported; no silent replacement.')
    if 'label' in frame:
        labels = pd.to_numeric(frame.label, errors='raise')
        if labels.isnull().any() or not labels.isin([0, 1]).all():
            raise ValueError('Labels must be binary, without missing values.')
        frame['label'] = labels.astype(np.float32)
    if frame.duplicated(['peptide', 'hla_sequence']).any():
        raise ValueError('Duplicate peptide/HLA-sequence inputs must be resolved during preparation.')
    return frame.reset_index(drop=True)


def check_development_split(train, valid):
    train_keys = set(zip(train.peptide, train.hla_sequence))
    if train_keys.intersection(zip(valid.peptide, valid.hla_sequence)):
        raise ValueError('Exact input overlap between training and validation tables.')
    if set(train.label) != {0, 1} or set(valid.label) != {0, 1}:
        raise ValueError('Training and validation must each contain both classes.')


def residue_inputs(sequence, length):
    target = torch.full((length,), PAD, dtype=torch.long)
    target[:len(sequence)] = torch.tensor([AMINO_ACIDS.index(x) for x in sequence])
    onehot = torch.zeros(length, 21)
    onehot[torch.arange(len(sequence)), target[:len(sequence)]] = 1
    mask = torch.arange(length) < len(sequence)
    return onehot, target, mask


class FeatureStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.index_path = self.directory / 'index.json'
        self.index = json.loads(self.index_path.read_text())
        if self.index['checkpoint'] != 'esm2_t33_650M_UR50D' or self.index['layer'] != 33:
            raise ValueError('Expected frozen ESM-2 650M final-layer features.')
        if self.index['dimension'] != 1280:
            raise ValueError('Expected 1280-dimensional residue features.')
        self.loaded = OrderedDict()

    def feature(self, branch, sequence):
        filename, row = self.index[branch][sequence]
        path = (self.directory / filename).resolve()
        if self.directory not in path.parents:
            raise ValueError('Feature shard path escapes cache directory.')
        if filename not in self.loaded:
            expected = self.index['shards'][filename]
            if sha256(path) != expected:
                raise ValueError(f'Feature shard checksum mismatch: {filename}')
            self.loaded[filename] = load_tensors(path)
            if len(self.loaded) > 2:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(filename)
        if not isinstance(row, int) or not 0 <= row < len(self.loaded[filename]):
            raise ValueError('Invalid feature row index.')
        feature = self.loaded[filename][row]
        length = 15 if branch == 'peptides' else 180
        if feature.shape != (length, 1280) or not torch.isfinite(feature).all():
            raise ValueError('Invalid cached feature shape or non-finite value.')
        return feature.float()


class PairDataset(Dataset):
    def __init__(self, frame, features):
        self.frame = frame
        self.features = features
        for branch, column in [('peptides', 'peptide'), ('hlas', 'hla_sequence')]:
            absent = set(frame[column]) - self.features.index[branch].keys()
            if absent:
                raise ValueError(f'{len(absent)} missing {branch} sequences in feature cache.')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        po, pt, mask = residue_inputs(row.peptide, 15)
        ho, ht, _ = residue_inputs(row.hla_sequence, 180)
        item = {'peptide_onehot': po, 'peptide_target': pt,
                'peptide_esm': self.features.feature('peptides', row.peptide),
                'hla_onehot': ho, 'hla_target': ht,
                'hla_esm': self.features.feature('hlas', row.hla_sequence),
                'mask': mask}
        if 'label' in self.frame:
            item['label'] = torch.tensor(row.label, dtype=torch.float32)
        return item
