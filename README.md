# Exp2Struct

**Experimental Data → Protein Structure Prediction**

Machine Learning / Computational Biology project for predicting protein structural information from experimental molecular data.

## Project Overview

Exp2Struct исследует возможность восстановления пространственной информации о белках с использованием машинного обучения и экспериментальных молекулярных данных.

Основная идея:

```text
Protein sequence
        +
Experimental molecular data
        ↓
   Feature extraction
        ↓
    ML / DL model
        ↓
Distance / Contact prediction
        ↓
Structural reconstruction
        ↓
  3D protein structure
````

Проект ориентирован на исследование методов:

* classical Machine Learning;
* Deep Learning;
* Convolutional Neural Networks;
* Graph Neural Networks;
* Transformers;
* geometric protein structure reconstruction.

## Research Goal

Основная задача первой версии проекта:

> Предсказать взаимное пространственное расположение аминокислотных остатков белка на основе последовательности и экспериментальных данных.

В качестве промежуточного структурного представления будут использоваться:

* residue-residue contact maps;
* pairwise distance maps;
* geometric constraints.

После этого предсказанные ограничения будут использоваться для восстановления приблизительной 3D-структуры белка.

## Project Pipeline

```text
Experimental Data
        ↓
Data Validation
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
ML Model
        ↓
Pairwise Distance / Contact Map
        ↓
Geometric Reconstruction
        ↓
3D Structure
        ↓
Evaluation
```

## Planned Models

Проект будет развиваться поэтапно.

### Baseline

```text
Sequence + Experimental Features
                ↓
              MLP
                ↓
          Contact Map
```

### CNN

```text
Residue Features
      ↓
     CNN
      ↓
Pairwise Predictions
```

### GNN

```text
Protein
  ↓
Graph Representation
  ↓
Graph Neural Network
  ↓
Structural Predictions
```

### Transformer

```text
Protein Sequence
       +
Experimental Data
       ↓
Transformer
       ↓
Pairwise Residue Representation
       ↓
Distance Prediction
```

## Structure Reconstruction

После получения distance/contact maps проект будет использовать геометрические ограничения для восстановления координат атомов или аминокислотных остатков.

```text
Predicted Distance Matrix
          ↓
Geometric Optimization
          ↓
3D Coordinates
          ↓
Protein Structure
```

## Repository Structure

```text
Exp2Struct/
│
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── structure/
│   ├── training/
│   ├── inference/
│   └── visualization/
│
├── tests/
├── models/
├── results/
└── scripts/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Exp2Struct.git
cd Exp2Struct
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install the project:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

## Data

Raw experimental data will not be stored in the repository.

Expected project structure:

```text
data/
├── raw/
├── interim/
└── processed/
```

Dataset preparation and preprocessing will be implemented separately.

## Experiments

Experiments will be performed through Jupyter notebooks and reproducible Python scripts.

Planned experiments:

```text
01_data_exploration.ipynb
02_experimental_data_analysis.ipynb
03_feature_engineering.ipynb
04_baseline_model.ipynb
05_structure_reconstruction.ipynb
```

## Evaluation

Depending on the task, the project will use metrics for:

* contact prediction;
* pairwise distance prediction;
* structural similarity;
* reconstruction error.

Potential structural metrics:

```text
RMSD
TM-score
GDT
```

The exact evaluation protocol will be defined together with the selected dataset.

## Scientific Scope

The project is intended as an experimental Machine Learning research project at the intersection of:

```text
Data Science
Machine Learning
Computational Biology
Bioinformatics
Molecular Modeling
```

## Development Roadmap

```text
[ ] Project setup
[ ] Dataset selection
[ ] Data preprocessing
[ ] Exploratory analysis
[ ] Feature engineering
[ ] Baseline model
[ ] Contact prediction
[ ] Distance prediction
[ ] Structure reconstruction
[ ] CNN model
[ ] GNN model
[ ] Transformer model
[ ] Model comparison
[ ] Final benchmark
[ ] Visualization
[ ] Reproducible experiments
```

## Status

**Current version: `0.1.0`**

Project is under active development.

---

## License

MIT License

````
