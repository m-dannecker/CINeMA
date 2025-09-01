# CINeMA

**C**onditional **I**mplicit **Ne**ural **M**ulti-Modal **A**tlas for a Spatio-Temporal Representation of the Perinatal Brain

CINeMA is a deep learning framework for building conditional implicit neural multi-modal atlases that provide spatio-temporal representations of the developing brain. The framework uses Implicit Neural Representations (INRs) to create continuous, smooth atlases that can be conditioned on various developmental parameters factors such as age, birth age, and other anatomy like lateral ventricular volume (to mode ventriculomegaly) and the corpus callosum.

## Features

- **Multi-modal Atlas Generation**: Support for multi modal learning, like T1w, T2w, and segmentation modalities
- **Conditional Atlas Building**: Generate atlases conditioned on temporal and other factors, like ventricular volume
- **Implicit Neural Representations**: Continuous, smooth atlas representations using SIREN networks
- **Flexible Configuration**: YAML-based configuration system for easy customization
- **Wandb Integration**: Built-in experiment tracking and logging with weights and biases
- **GPU Acceleration**: CUDA support for fast training and inference
- **Multiple Datasets**: Support for DHCP (Developing Human Connectome Project) datasets

## Precomputed Atlases
You can find pre-computed temporal atlases, modeling neurotypical fetal and neonatal brain development such as fetal brain development with ventriculomegaly (VM) and agenesis of the corpus callosum (ACC), in the [Zenodo repository](https://zenodo.org/records/17023473).

## 📋 Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🛠️ Installation

### Prerequisites

- Python 3.10
- CUDA-compatible GPU (recommended)
- Conda or Miniconda

### Option 1: Using Conda Environment (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd CINeMA
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate cinema
```

### Option 2: Using pip

1. Clone the repository:
```bash
git clone <repository-url>
cd CINeMA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

### Supported Datasets

CINeMA was tested on the dHCP (developing Humand Connectome Project) dataset, including neonatal and fetal data, and on an in-house dataset.
For any dataset to use, you need to specify the following files:
- **TSV File**: Contains metadata (including all properties you want to condition on, e.g. ventricular volume) and file paths for all subjects
- **Subject IDs YAML**: Lists subject IDs (same IDs as in the TSV file) for training/validation splits
- **Configuration Files**: Define dataset parameters, constraints, conditions, etc. 
The provided templates will help you to get started.

### Data Preprocessing
- Data is expected to be in nifti format. 
- All subjects must be roughly in the same orientation. 
- Fine registration is taken care of by the framework. No prior atlas is needed. 
- Conditionable properties like ventricular volume must be extracted from the data and stored in the TSV file beforehand.
- No resampling required! Data can be of different size and spacing due to resolution agnostic properties of INRs. 

### Config Setup
The config files are located in the `configs` folder. The provided templates are well commented and will help you to get started. Some important things to consider in config_data.yaml:
- *Conditions*, specify which properties are used for explicit conditioning of the atlas. If set true, they must also be specified in the `atlas_gen` section of config_atlas.yaml. Note, scan_age is usually set to false, as we never explicitly use scan age for atlas generation. Temporal atlas generation is done through latent regression.
- *Constraints*, specify the range of values for a condition. However, constraints do not need to be conditions. They can be used to cosntrain the sampling process of subjects. If a subject's value is not in this range, it will not be sampled at all. A distribution type can be specified to sample subjects accordingly. E.g. uniform_fillup for scan_age will sample subjects uniformly within the specified range if enough subjects for each age bin are available.

Some important things to consider in config_data.yaml:
config_atlas.yaml - Atlas generation parameters (*atlas_gen*):
- *temporal_values*, specify the temporal values (in weeks) for which to generate the atlas.
- *conditions*, specify which conditions are used for the atlas generation. They must be specified in config_data.yaml as well. Specify *values* for each condition if the condition is used for the atlas generation. Values can be normed between [-1, 1] by setting *normed_values* to true.
- *gaussian_span*, specifies the span of the gaussian distribution used to regress the latent code from the subjects in weeks.
- *cond_scale*, specifies the scale of the condition vector. Condition values are concatenated to the latent code which is drawn from a gaussian distribution with sigma=0.01. *cond_scale* scales the condition vector to a similar range. Higher *cond_scale* values leads to higher sensitivity of the model to the condition. 0.10 - 0.15 works well in practice.

## Usage

### Basic Training

To train a CINeMA atlas with default settings:

```bash
python run.py
```

### Custom Configuration

You can override configuration parameters via command line:

```bash
python run.py \
  --config_data dhcp_neo_mm \
  --seed 123 \
  --inr_decoder__latent_dim 256 3 3 3 \
  --inr_decoder__hidden_size 512
```

### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--config_data` | Dataset configuration name | `dhcp_neo_mm` |
| `--seed` | Random seed | `42` |
| `--inr_decoder__out_dim` | Output dimensions | `2 10` |
| `--inr_decoder__latent_dim` | Latent dimensions | `256 3 3 3` |
| `--inr_decoder__hidden_size` | Hidden layer size | `1024` |
| `--atlas_gen__cond_scale` | Condition scaling | `1.0` |

### Training with Different Datasets

For preterm neonates:
```bash
python run.py --config_data dhcp_neo_preterm_mm
```

For fetal data:
```bash
python run.py --config_data dhcp_fetal
```

## 📁 Output

After training, the following outputs are generated in the `output/` directory:

```
output/
└── {dataset}_{timestamp}_{job_id}/
    ├── config_atlas.yaml          # Saved atlas configuration
    ├── config_data.yaml           # Saved data configuration
    ├── model_epoch_{N}.pth        # Trained model checkpoints
    ├── {modality1}_ga={min_ga}-{max_ga}_cond={condition_number}_ep={epoch}.nii.gz
    ├── {modality2}_ga={min_ga}-{max_ga}_cond={condition_number}_ep={epoch}.nii.gz
    ├── ...
    ├── certainty_maps/            # Generated certainty maps for each segmentation label if activated in config_atlas.yaml
    │   ├── ....
    ├── train/           # Training subject reconstructions of of all modalities with metrics (if GT is available)
    ├── val/             # Validation subject reconstructions of of all modalities with metrics (if GT is available)
    ├── hist_scan_age_val/      # Histogram of scan age for validation subjects
    ├── hist_scan_age_train/    # Histogram of scan age for training subjects
    └── metrics/                   # Training metrics
```

### Atlas Generation

The framework generates atlases for each temporal value specified in the configuration. Atlases are saved as NIfTI files and can be loaded in standard neuroimaging software.

### Logging

If `logging: True` is set, training progress is logged to Weights & Biases. Make sure to specify your wandb entity in the config_atlas.yaml.


## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `inr_decoder__latent_dim` or `inr_decoder__hidden_size`
   - Decrease batch size in data loading (might lead to decreased atlas quality)
   - Decrease `n_samples` parameter for memory usage.
   - Generating a large number of atlases might lead to GPU and system memory issues. You can specify a smaller range of conditions/temporal steps to generate atlases or use coarser resolution for the atlases. 

2. **Configuration Errors**
   - Ensure all required fields are present in YAML files
   - Check that subject IDs exist in the TSV file
   - Verify file paths are correct
   - Verify that all conditions are specified in config_data.yaml and config_atlas.yaml *atlas_gen* section.

3. **Data Loading Issues**
   - Ensure NIfTI files are properly formatted (each file should be 3D)



## 📚 Citation

If you use CINeMA in your research, please cite:

```bibtex
@article{dannecker2025cinema,
  title={CINeMA: Conditional Implicit Neural Multi-Modal Atlas for a Spatio-Temporal Representation of the Perinatal Brain},
  author={Dannecker, Maik and Sideri-Lampretsa, Vasiliki and Starck, Sophie and Mihailov, Angeline and Milh, Mathieu and Girard, Nadine and Auzias, Guillaume and Rueckert, Daniel},
  journal={arXiv preprint arXiv:2506.09668},
  year={2025}
}
```


## 📄 License

This project is licensed under the terms specified in the LICENSE file.

For questions and support, please open an issue on the GitHub repository.
