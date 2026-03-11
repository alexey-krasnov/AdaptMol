# AdaptMol: Domain Adaptation for Molecular Image Recognition with Limited Supervision

![AdaptMol Overview](assets/mmdimage.png)

This repository contains the official implementation of our paper "AdaptMol: Domain Adaptation for Molecular Image Recognition with Limited Supervision".

## Citation

If you use our work in your research, please cite:
```bibtex
# Citation will be added upon publication
```

## Dataset

The dataset used in this work is available at: [train](https://huggingface.co/fffh1/AdaptMol/blob/main/train.tar.gz) [evaluation](https://huggingface.co/fffh1/AdaptMol/blob/main/evaluate.tar.gz)

## Pretrained Model

Download our pretrained model from: [model](https://huggingface.co/fffh1/AdaptMol/blob/main/pretrained.pth)

## Installation

Create the conda environment using the provided configuration file:
```bash
conda env create -f environment.yml
conda activate adaptmol
```

## Usage

### Inference

Run prediction on molecular images:
```bash
python predict.py --model_path checkpoints_path --image_path image_path
```

### Training

Training consists of four stages. Run them sequentially:

**Stage 1:**
```bash
bash scripts/stage1.sh
```

**Stage 2:** Generate predictions on USPTO dataset
```bash
bash scripts/predict_uspto.sh
```

**Stage 3:**
```bash
bash scripts/stage2.sh
```

**Stage 4:**
```bash
bash scripts/stage3.sh
```

## Paper Result Evaluation

To reproduce the results reported in our paper:
```bash
bash scripts/paper_evaluation.sh
```

## Acknowledgements

This work builds upon several excellent projects:

- **MolScribe**: We thank the authors for their work. Our code architecture is based on their implementation.
- **MolDepictor**: Our synthetic training data generation is based on their code with modifications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project includes components from third-party sources - see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) for details.