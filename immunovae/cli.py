"""One-command prediction/training with automatic frozen ESM-2 feature caching."""
import argparse
import json
from pathlib import Path

import torch

from .data import FeatureStore, PairDataset, check_development_split, read_pairs
from .engine import TrainConfig, predict, train, validation_metrics
from .features import build_features
from .model import ImmunoVAE, ModelConfig
from .prepare import prepare_el, prepare_im
from .utils import load_training_checkpoint, selected_checkpoint

PACKAGE = Path(__file__).resolve().parents[1]


def execution_device(value):
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if value == 'auto' else value
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu or --device auto.')
    return device


def feature_options(parser):
    parser.add_argument('--features', default='cache/esm2_650M',
                        help='Reusable ESM-2 feature directory; missing sequences are encoded automatically.')
    parser.add_argument('--esm-batch-size', type=int, default=8)
    parser.add_argument('--device', default='auto', help='auto, cpu, or a CUDA device such as cuda:0.')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='immunovae')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Prepare explicit EL or IM data partitions.')
    prepare.add_argument('--task', choices=['el', 'im'], required=True)
    prepare.add_argument('--train', required=True)
    prepare.add_argument('--valid', required=True)
    prepare.add_argument('--test')
    prepare.add_argument('--external', nargs='+', required=True)
    prepare.add_argument('--hla-map')
    prepare.add_argument('--output', required=True)
    cache = commands.add_parser('cache', help='Optionally prepare ESM-2 features separately.')
    cache.add_argument('--tables', nargs='+', required=True)
    cache.add_argument('--output', default='cache/esm2_650M')
    cache.add_argument('--device', default='auto')
    cache.add_argument('--batch-size', type=int, default=8)
    fit = commands.add_parser('train', help='Train EL or adapt IM with automatic ESM-2 caching.')
    fit.add_argument('--task', choices=['el', 'im'], required=True)
    fit.add_argument('--train', help='Defaults to the supplied task-specific training CSV.')
    fit.add_argument('--valid', help='Defaults to the supplied task-specific validation CSV.')
    fit.add_argument('--output', help='Defaults to runs/<task>_seed<seed>.')
    fit.add_argument('--seed', type=int, choices=[42, 2026, 2027], default=2026)
    fit.add_argument('--strategy', choices=['fine-tuned', 'frozen', 'scratch'], default='fine-tuned')
    fit.add_argument('--pretrained', help='EL checkpoint for IM transfer; auto-selected for the requested seed.')
    feature_options(fit)
    inference = commands.add_parser('predict', help='Predict with automatic ESM-2 caching.')
    inference.add_argument('--task', choices=['el', 'im'], help='Chooses the included checkpoint; default im.')
    inference.add_argument('--checkpoint', help='Optional custom validation-selected best.pt.')
    inference.add_argument('--input', required=True)
    inference.add_argument('--output', help='Defaults to outputs/<input-stem>_<task>_predictions.csv.')
    feature_options(inference)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if args.task == 'el':
            if not args.test:
                parser.error('EL preparation requires --test.')
            prepare_el(args.train, args.valid, args.test, args.external, args.output, args.hla_map)
        else:
            if args.test:
                parser.error('IM preparation does not accept --test.')
            prepare_im(args.train, args.valid, args.external, args.output, args.hla_map)
        return
    device = execution_device(args.device)
    print(f'Device: {device}', flush=True)
    if args.command == 'cache':
        build_features(args.tables, args.output, device, args.batch_size)
    elif args.command == 'train':
        config = TrainConfig(task=args.task, seed=args.seed, strategy=args.strategy)
        train_csv = Path(args.train) if args.train else PACKAGE / f'data/{args.task}/train.csv'
        valid_csv = Path(args.valid) if args.valid else PACKAGE / f'data/{args.task}/valid.csv'
        suffix = '' if args.strategy == 'fine-tuned' else '_' + args.strategy
        output = Path(args.output or f'runs/{args.task}{suffix}_seed{args.seed}')
        if output.exists():
            raise FileExistsError(f'Choose a new --output directory: {output}')
        pretrained = args.pretrained
        if args.task == 'im' and args.strategy != 'scratch':
            if not pretrained:
                candidates = [Path(f'runs/el_seed{args.seed}/best.pt'), PACKAGE / 'checkpoints/EL/best.pt']
                for candidate in candidates:
                    if candidate.exists():
                        saved = load_training_checkpoint(selected_checkpoint(candidate))
                        if saved['config'].get('seed', saved.get('seed')) == args.seed:
                            pretrained = str(candidate)
                            break
                if not pretrained:
                    parser.error(f'No EL checkpoint for seed {args.seed}; train EL first or provide --pretrained.')
            selected_checkpoint(pretrained)
        check_development_split(read_pairs(train_csv), read_pairs(valid_csv))
        build_features([train_csv, valid_csv], args.features, device, args.esm_batch_size)
        train(config, train_csv, valid_csv, args.features, output, pretrained, device)
    else:
        checkpoint = Path(args.checkpoint) if args.checkpoint else PACKAGE / f'checkpoints/{(args.task or "im").upper()}/best.pt'
        saved = load_training_checkpoint(selected_checkpoint(checkpoint))
        task = saved['config']['task']
        if args.task and args.task != task:
            parser.error('--task does not match the selected checkpoint.')
        if saved.get('epoch') != saved['best'].get('epoch'):
            raise ValueError('Use the validation-selected checkpoint listed in best.json.')
        output = Path(args.output or f'outputs/{Path(args.input).stem}_{task}_predictions.csv')
        if output.exists():
            raise FileExistsError(f'Choose a new --output path: {output}')
        frame = read_pairs(args.input, require_labels=False)
        build_features([args.input], args.features, device, args.esm_batch_size)
        dataset = PairDataset(frame, FeatureStore(args.features))
        model = ImmunoVAE(ModelConfig(**saved['model_config']))
        model.load_state_dict(saved['model'], strict=True)
        model.to(device).eval()
        scores = predict(model, dataset, task, device)
        result = frame.copy()
        result['probability'] = scores
        result['prediction'] = (scores >= saved['best']['threshold']).astype(int)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, mode='x')
        print(f'Saved {len(result)} predictions: {output}', flush=True)
        if 'label' in frame and frame.label.nunique() == 2:
            print(json.dumps(validation_metrics(frame.label, scores)))


if __name__ == '__main__':
    main()
