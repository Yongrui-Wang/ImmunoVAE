"""Explicit source-to-input conversion; no model scores are used in partitioning."""
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import read_pairs
from .utils import atomic_json, sha256


def allele_key(value):
    return re.sub('[^A-Z0-9]', '', str(value).upper())


def hla_mapping(path):
    table = pd.read_csv(path)
    allele_column = 'allele' if 'allele' in table else 'Alleles'
    sequence_column = next((key for key in ('hla_sequence', 'Sequence', 'Sequence276') if key in table), None)
    if sequence_column is None or allele_column not in table:
        raise ValueError('HLA map needs allele/Alleles and hla_sequence/Sequence/Sequence276.')
    result = {}
    for allele, sequence in zip(table[allele_column], table[sequence_column]):
        if pd.isna(allele) or pd.isna(sequence):
            raise ValueError('Missing HLA mapping entry.')
        sequence = str(sequence).strip().upper()
        if sequence_column == 'Sequence276':
            if len(sequence) != 276:
                raise ValueError('Sequence276 must contain exactly 276 residues.')
            sequence = sequence[1:181]
        if len(sequence) != 180:
            # Unused incomplete alleles are not eligible inputs to this model.
            continue
        key = allele_key(allele)
        if key in result and result[key] != sequence:
            raise ValueError('Conflicting HLA sequences for an allele alias.')
        result[key] = sequence
    return result


def canonical_table(path, mapping=None, name=None):
    frame = pd.read_csv(path)
    aliases = {'Peptide': 'peptide', 'pep': 'peptide', 'Alleles': 'allele',
               'mhc': 'allele', 'Label': 'label', 'tgt': 'label'}
    frame = frame.rename(columns={a: b for a, b in aliases.items() if b not in frame})
    if 'peptide' not in frame or 'label' not in frame:
        raise ValueError('Source tables need peptide and label columns (or documented aliases).')
    if frame.peptide.isnull().any():
        raise ValueError('Missing source peptide.')
    frame['peptide'] = frame.peptide.str.strip().str.upper()
    if 'hla_sequence' not in frame:
        if mapping is None or 'allele' not in frame:
            raise ValueError('Supply an HLA map or a 180-aa hla_sequence column.')
        frame['hla_sequence'] = frame.allele.map(allele_key).map(mapping)
    if frame.hla_sequence.isnull().any():
        raise ValueError('Unmapped or incomplete HLA sequence in a source table.')
    frame['hla_sequence'] = frame.hla_sequence.str.strip().str.upper()
    frame['source_row'] = range(len(frame))
    frame['source_table'] = name or Path(path).name
    if 'sample_id' not in frame:
        frame['sample_id'] = [f'{name or Path(path).name}:{i:08d}' for i in range(len(frame))]
    return frame


def source_manifest(paths, mapping):
    files = [Path(path) for path in paths] + ([Path(mapping)] if mapping else [])
    return {str(path): sha256(path) for path in files}


def deduplicate(frame):
    identity = ['peptide', 'hla_sequence']
    conflict = frame.groupby(identity).label.nunique().gt(1)
    if conflict.any():
        raise ValueError('Conflicting labels for identical inputs; resolve the source data explicitly.')
    return frame.drop_duplicates(identity, keep='first').copy()


def export(frame, path):
    frame.to_csv(path, index=False)
    read_pairs(path)


def prepare_el(train_csv, valid_csv, test_csv, external_csvs, output, mapping_csv=None):
    """Retain supplied EL partitions; precedence test > validation > training."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mapping = hla_mapping(mapping_csv) if mapping_csv else None
    blocked = set()
    for i, path in enumerate(external_csvs):
        blocked.update(canonical_table(path, mapping, f'external_{i}').peptide)
    frames = {}
    seen = set()
    excluded = []
    for split, path in [('test', test_csv), ('valid', valid_csv), ('train', train_csv)]:
        raw = canonical_table(path, mapping, split)
        cleaned = deduplicate(raw)
        mask = ~pd.Series(list(zip(cleaned.peptide, cleaned.hla_sequence)), index=cleaned.index).isin(seen)
        if split != 'test':
            mask &= ~cleaned.peptide.isin(blocked)
        keep = cleaned[mask].copy()
        excluded.append(raw[~raw.sample_id.isin(keep.sample_id)].assign(reason='duplicate_or_development_exclusion'))
        frames[split] = keep
        seen.update(zip(keep.peptide, keep.hla_sequence))
        export(keep, output / f'{split}.csv')
    pd.concat(excluded, ignore_index=True).to_csv(output / 'excluded_rows.csv', index=False)
    atomic_json(output / 'manifest.json', {'task': 'el', 'source_sha256': source_manifest(
        [train_csv, valid_csv, test_csv] + list(external_csvs), mapping_csv),
        'partition_rule': 'preserve supplied partitions; test > valid > train',
        'development_exclusion': 'all supplied external-cohort peptides',
        'counts': {name: len(frame) for name, frame in frames.items()}})


def prepare_im(train_csv, valid_csv, external_csvs, output, mapping_csv=None):
    """Nonrandom BigMHC development records; fixed stratified pair-level 80:20 split."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mapping = hla_mapping(mapping_csv) if mapping_csv else None
    frames = []
    for name, path in [('source_train', train_csv), ('source_valid', valid_csv)]:
        frame = canonical_table(path, mapping, name)
        if 'rand' not in frame:
            raise ValueError('BigMHC development input must include the rand source flag.')
        frames.append(frame[frame['rand'] == 0].copy())
    blocked = set()
    for i, path in enumerate(external_csvs):
        blocked.update(canonical_table(path, mapping, f'external_{i}').peptide)
    raw = pd.concat(frames, ignore_index=True)
    cleaned = raw[~raw.peptide.isin(blocked)].copy()
    identity = ['peptide', 'hla_sequence']
    conflicts = cleaned.groupby(identity).label.transform('nunique') > 1
    cleaned = cleaned[~conflicts].drop_duplicates(identity).copy()
    train, valid = train_test_split(cleaned, test_size=.2, stratify=cleaned.label, random_state=42)
    export(train.sort_index(), output / 'train.csv')
    export(valid.sort_index(), output / 'valid.csv')
    raw[~raw.sample_id.isin(cleaned.sample_id)].to_csv(output / 'excluded_rows.csv', index=False)
    atomic_json(output / 'manifest.json', {'task': 'im', 'source_sha256': source_manifest(
        [train_csv, valid_csv] + list(external_csvs), mapping_csv),
        'source_scope': 'BigMHC im_train + im_val only; rand=0; tgt labels',
        'partition_rule': 'deduplicate peptide/HLA-sequence; stratified 80:20; seed 42',
        'development_exclusion': 'all supplied external-cohort peptides',
        'counts': {'train': len(train), 'valid': len(valid)}})
