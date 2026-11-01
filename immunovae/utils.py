"""Reproducible RNG management and atomic model and feature-file writes."""
import contextlib
import hashlib
import json
import inspect
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_torch_save(path, payload, overwrite=False):
    """Publish a complete tensor file atomically; replacement must be explicit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp, path)
        else:
            os.link(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def checkpoint_rng_state():
    """Record RNG state using tensors and primitive types for restricted loading."""
    state = rng_state()
    numpy_state = state['numpy']
    state['numpy'] = [numpy_state[0], numpy_state[1].tolist(),
                      int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])]
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextlib.contextmanager
def evaluation_rng():
    state = rng_state()
    try:
        seed_everything(91020)
        yield
    finally:
        restore_rng(state)


def load_tensors(path):
    """Load locally generated tensor caches, using restricted loading when available."""
    options = {'weights_only': True} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(path, map_location='cpu', **options)


def load_training_checkpoint(path):
    """Load a model checkpoint containing tensors and primitive metadata."""
    return load_tensors(path)


def selected_checkpoint(path):
    """Require completed development and the final validation-selected checkpoint."""
    path = Path(path).resolve()
    if path.name != 'best.pt':
        raise ValueError('Expected the validation-selected best.pt file.')
    run = path.parent
    marker = run / 'complete.json'
    if not marker.exists():
        raise ValueError('Training must finish before EL transfer or external prediction.')
    complete = json.loads(marker.read_text())
    if (run / complete['selected_checkpoint']).resolve() != path:
        raise ValueError('Use the final validation-selected epoch recorded in complete.json.')
    if sha256(path) != complete['selected_checkpoint_sha256']:
        raise ValueError('Selected checkpoint checksum mismatch.')
    return path
