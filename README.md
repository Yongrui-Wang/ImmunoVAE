# ImmunoVAE

ImmunoVAE predicts HLA-I antigen presentation and peptide immunogenicity using frozen ESM-2 features, directional peptide–HLA attention and dual variational reconstruction. This package contains the model, study data, paired EL/IM checkpoints and predictions from three training seeds.

## 1. Install the environment

Clone this repository and enter its directory:

```bash
git clone https://github.com/Yongrui-Wang/ImmunoVAE.git
cd ImmunoVAE
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate immunovae
```

The environment includes Python 3.10, PyTorch 2.5.1 and **fair-esm 2.0.0** for ESM-2. No separate package-installation command is needed. Keep the terminal in this directory when running the examples below.

## 2. Predict with the included models

Immunogenicity prediction on TESLA requires one command:

```bash
python predict.py --task im --input data/external/TESLA.csv
```

The command selects `checkpoints/IM/best.pt`, downloads ESM-2 weights when needed, prepares missing sequence features, runs inference and writes `outputs/TESLA_im_predictions.csv`.

Other supplied test sets can be evaluated in the same way:

```bash
python predict.py --task im --input data/external/GBM.csv
python predict.py --task el --input data/el/test.csv
python predict.py --task el --input data/external/IEDB_Weekly.csv
```

The `el` and `im` tasks automatically select their respective included checkpoints. Both released checkpoints correspond to seed2026; the IM model was initialized from the paired EL model.

GPU execution is selected automatically when CUDA is available; otherwise the command uses the CPU. To select a device, output file or another trained checkpoint explicitly:

```bash
python predict.py --task im --input data/external/TESLA.csv --checkpoint runs/im_seed2026/best.pt --output outputs/TESLA_retrained.csv --device cuda:0
```

Use `--device cpu` for CPU execution. The output contains the input columns, a continuous `probability` and a binary `prediction` using the threshold stored in the selected checkpoint. If labels are present, AUROC and AP are printed. Existing output files are preserved; specify a new `--output` path for another evaluation.

## 3. Train an EL model

```bash
python train.py --task el
```

This command uses `data/el/train.csv` and `data/el/valid.csv`, automatically prepares ESM-2 features, trains seed2026 and saves the run in `runs/el_seed2026/`.

A different study seed or custom data partition can be supplied in the same command:

```bash
python train.py --task el --seed 42 --train data/el/train.csv --valid data/el/valid.csv --output runs/el_seed42
```

## 4. Transfer an EL model to IM

```bash
python train.py --task im
```

This command uses the fixed IM training and validation partitions. It first looks for `runs/el_seed2026/best.pt`; if that run is absent, it uses the included seed2026 EL checkpoint. The backbone is transferred and a newly initialized IM head is trained with full-model fine-tuning. Results are saved in `runs/im_seed2026/`.

For another seed, train the corresponding EL model first, then run:

```bash
python train.py --task im --seed 42
```

An EL model can also be selected explicitly:

```bash
python train.py --task im --seed 2026 --pretrained checkpoints/EL/best.pt --output runs/im_custom_seed2026
```

The other IM training strategies each use one command:

```bash
python train.py --task im --strategy frozen
python train.py --task im --strategy scratch
```

`frozen` trains the IM head while keeping the EL backbone fixed. `scratch` initializes the downstream model without EL pretraining. Training seeds are restricted to 42, 2026 and 2027; the default is 2026. Use a separate output directory for each run.

Both tasks train for 50 epochs without early stopping and select `best.pt` by the highest validation AUROC. Each completed epoch also saves an independent `checkpoints/epoch_001.pt`, `epoch_002.pt`, and so on. These files contain model, optimizer, scheduler and scaler states, configuration, metrics, RNG state and data provenance. `history.json` records training and validation metrics. External test sets are evaluated separately with `predict.py`.

## 5. ESM-2 features: automatic or separate

Training and prediction automatically use the frozen `esm2_t33_650M_UR50D` model and its final-layer, 1280-dimensional residue representations. Model-specific special tokens are excluded. Features are stored in `cache/esm2_650M/`, with a sequence index in `index.json`.

The official ESM loader downloads model weights on first use and subsequently reuses the local PyTorch cache. Sequence features are also reused across commands; only missing sequences are encoded. A fully populated feature cache permits inference without loading ESM-2 again.

To prepare features separately, run:

```bash
python extract_esm.py --tables data/external/GBM.csv data/external/TESLA.csv
```

A custom feature directory and ESM batch size can be selected:

```bash
python extract_esm.py --tables data/el/train.csv data/el/valid.csv --output cache/el --batch-size 4
```

Reuse that directory directly during training:

```bash
python train.py --task el --features cache/el
```

`--esm-batch-size` controls feature extraction within `train.py` and `predict.py`; `--batch-size` is the corresponding option in `extract_esm.py`. These options do not change the EL/IM training batch sizes.

## 6. Data and input format

The EL training partition is stored as `data/el/train.csv.gz` and read directly without manual decompression. Commands also accept `.csv.gz` input paths explicitly; the default `data/el/train.csv` path automatically uses the compressed file.

The supplied partitions contain:

| Task / dataset | Training pairs | Validation pairs | Test pairs |
|---|---:|---:|---:|
| EL primary | 1,705,801 | 89,721 | 6,342 |
| IM development | 5,179 | 1,295 | |
| GBM | | | 123 |
| TESLA | | | 399 |
| IEDB Weekly | | | 535 |

Custom input CSV files use the following columns:

| Column | Meaning |
|---|---|
| `sample_id` | Unique row identifier |
| `peptide` | Sequence of 1–15 canonical amino acids |
| `hla_sequence` | The corresponding 180-residue HLA sequence |
| `label` | Binary 0/1 label; required for training/validation, optional for inference |
| `allele` | Optional HLA allele name |
| `shard_id` | Optional preserved EL source-shard identifier for training |

The supplied input tables already have this format. Peptides are padded to 15 positions internally; padding is excluded from peptide pooling and reconstruction. HLA sequences contain 180 valid residues. Input validation checks lengths, residue identities, duplicate inputs and conflicting labels.

`data/all_hla_sequence.csv` supplies the allele mapping. `immunovae/prepare.py` provides source-data preparation; its `prepare` command is available through `python -m immunovae prepare --help`. The provided HLA representation corresponds to residues 2–181 when extracted from the 276-residue sequence field.

`predictions/ImmunoVAE_test_predictions.csv` contains only ImmunoVAE results: `task`, `dataset`, `peptide`, `allele`, `label`, `prediction_seed42`, `prediction_seed2026` and `prediction_seed2027`. It includes the primary EL, two low-frequency HLA, IEDB Weekly, GBM and TESLA test sets. These archived scores are separate from new predictions generated by the commands above.

## 7. Training settings

| Setting | EL | IM |
|---|---|---|
| Batch size | 256 | 128 |
| Learning rate | First-epoch warmup from 2e-5 to 2e-4 | Backbone 3e-5; prediction head 1e-4 |
| Optimizer | AdamW, weight decay 1e-4 | AdamW, weight decay 1e-4 |
| LR scheduler | Validation-AUROC ReduceLROnPlateau | Validation-AUROC ReduceLROnPlateau |
| Training objective | Focal classification plus reconstruction and KL terms | Focal classification |
| Inference | Mean sigmoid probability over 20 posterior draws | Same |

The focal loss uses a global coefficient of 0.7 for both classes and focusing parameter 1.3. With reconstruction terms `R` and branchwise KL terms:

```text
L_EL = L_focal + 1e-3 * (3 * R_peptide + 0.5 * R_HLA
                         + 0.2 * (0.2 * KL_peptide + KL_HLA))
L_IM = L_focal
```

Reconstruction averages valid peptide residues and all HLA positions, respectively. Each KL term sums latent dimensions and averages over the batch. ESM-2 remains frozen. Training draws one latent sample per branch; classification validation and inference average 20 posterior samples with inference seed 91020.

For all command-line options, use `python predict.py --help`, `python train.py --help` or `python extract_esm.py --help`.
