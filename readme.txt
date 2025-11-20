# Collaborative CNN Team Project: Cats vs Dogs Classification

## Project Overview

This repository contains a collaborative machine learning project focused on binary image classification for cats and dogs. Two separate teams developed and trained Convolutional Neural Network (CNN) models on different datasets to explore cross-dataset generalization capabilities and domain adaptation challenges.

## Project Structure

```
├── models/           # Trained model files (.h5 format)
├── notebooks/        # Jupyter notebooks for training and testing
│   ├── train_v1.ipynb   # Training script for Model V1
│   ├── train_v2.ipynb   # Training script for Model V2
│   ├── test_v1.ipynb    # Testing script for Model V1
│   ├── test_v2.ipynb    # Testing script for Model V2
│   └── model_report.ipynb # Report generation notebook
├── results/          # Performance metrics and test results
├── data/             # Dataset directories (manual download may be required)
│   ├── user1/        # Dataset for Team 1 (Cat & Dog Dataset)
│   └── user2/        # Dataset for Team 2 (Dogs vs Cats Redux)
├── utils/            # Utility scripts
├── report.md         # Detailed analysis report
├── requirements.txt  # Python dependencies
└── readme.txt        # This file
```

## Team Contributions

### Team 1 (User 1)
- **Model**: Simple 2-layer CNN
- **Dataset**: "Cat & Dog Dataset (Tong Python)"
- **Architecture**: 3 Conv2D layers with Global Average Pooling
- **Performance**: 69.08% validation accuracy on own dataset

### Team 2 (User 2)
- **Model**: Improved CNN with Dropout regularization
- **Dataset**: "Dogs vs Cats Redux" (Kaggle competition dataset)
- **Architecture**: 3 Conv2D layers with 40% Dropout for regularization
- **Performance**: 71.62% validation accuracy on own dataset

## Key Findings

The project explored cross-dataset generalization by testing each team's model on the other team's dataset. Surprisingly, both models showed improved performance when tested on the alternative dataset:
- Model V1: 69.08% → 70.66% (Team 1 dataset → Team 2 dataset)
- Model V2: 71.62% → 71.21% (Team 2 dataset → Team 1 dataset)

This demonstrates positive transfer learning between the datasets and highlights the importance of robust model regularization.

## Requirements

All dependencies are listed in `requirements.txt`. Key libraries include:
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Usage Instructions

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Training Models**
   - Execute `notebooks/train_v1.ipynb` for Team 1 model
   - Execute `notebooks/train_v2.ipynb` for Team 2 model

3. **Testing and Evaluation**
   - Use `notebooks/test_v1.ipynb` and `notebooks/test_v2.ipynb`
   - Results are saved in the `results/` directory

4. **Viewing Results**
   - Detailed analysis in `report.md`
   - Performance metrics in `results/` directory

## Results Summary

| Model | Trained On | Tested On | Accuracy |
|-------|------------|-----------|----------|
| Model V1 | Team 1 Data | Team 1 Data | 69.08% |
| Model V1 | Team 1 Data | Team 2 Data | 70.66% |
| Model V2 | Team 2 Data | Team 2 Data | 71.62% |
| Model V2 | Team 2 Data | Team 1 Data | 71.21% |

## Detailed Analysis

For comprehensive analysis of the experiments, methodologies, and findings, please refer to `report.md`.

## Files in Results Directory

The `results/` directory contains:
- `metrics_v1.json` - Training/validation metrics for Model V1
- `metrics_v2.json` - Training/validation metrics for Model V2
- `test_v1_user2.json` - Model V1 tested on Team 2 data
- `test_v2_user1.json` - Model V2 tested on Team 1 data

## Notebooks Overview

- `train_v1.ipynb` - Implements the simple 2-layer CNN for Team 1
- `train_v2.ipynb` - Implements the improved CNN with dropout for Team 2
- `test_v1.ipynb` - Tests Team 1 model on Team 2 dataset
- `test_v2.ipynb` - Tests Team 2 model on Team 1 dataset
- `model_report.ipynb` - Generates the final analysis report

## Contributing

This was a collaborative effort between two development teams exploring the challenges and opportunities in distributed machine learning projects.

## License

This project is created for educational and research purposes.