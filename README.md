# Behavioural Neural Networks

This repository contains machine learning experiments for analyzing pigeon behavioral data using both traditional machine learning and deep learning approaches.

## Overview

The project implements and compares various machine learning models for classifying pigeon behavioral patterns from time series data. The data consists of 90-timestep sequences with 2 features per timestep, representing behavioral measurements from 6 different subjects across 7 different behavioral categories.

## Project Structure

```
behavioural-neural-networks/
├── Pigeon Samples Used for Experiments/    # Raw data files (sam*.csv)
├── traditional_ml_models.py                # Traditional ML models script
├── deep_learning_models.py                 # Deep learning models script
├── results/                                # Output directory for results
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Models Implemented

### Traditional Machine Learning Models (`traditional_ml_models.py`)
- **Random Forest**: Ensemble method with grid search optimization
- **Decision Tree**: Single tree classifier with hyperparameter tuning
- **Logistic Regression**: Linear classifier with regularization options
- **K-Nearest Neighbors**: Instance-based learning algorithm

### Deep Learning Models (`deep_learning_models.py`)
- **LSTM** (Long Short-Term Memory): Recurrent neural network for sequence modeling
- **GRU** (Gated Recurrent Unit): Alternative recurrent architecture
- **CNN** (Convolutional Neural Network): Custom architecture with the following layers:
  - Input: [90, 2] → Reshape to [90, 2, 1] for Conv2D
  - Conv2D: [82, 20] (Filters=20, Kernel=(9,2), Activation=Sigmoid)
  - MaxPooling: [41, 20] (Pool Size=(2,1))
  - Flatten: [820]
  - Dense: [7] (Units=7, Activation=Softmax)

## Data Format

The data consists of CSV files in the `Pigeon Samples Used for Experiments/` directory:
- File naming convention: `sam{subject}{session}{class}.csv`
  - `subject`: Subject number (1-6)
  - `session`: Session number (0-3)
  - `class`: Behavioral class (1-7)
- Each file contains a time series of behavioral measurements
- Data is padded/truncated to exactly 90 timesteps
- Features are standardized (for deep learning) or normalized (for traditional ML)

## Usage

### Prerequisites

Install required dependencies using pip:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

### Running Traditional ML Models

```bash
python traditional_ml_models.py
```

This script will:
1. Load and prepare the data
2. Perform grid search cross-validation for each model
3. Evaluate using leave-one-subject-out cross-validation
4. Save results to `results/traditional_ml_results.csv`

### Running Deep Learning Models

```bash
python deep_learning_models.py
```

This script will:
1. Load and prepare the data for sequence modeling
2. Train LSTM, GRU, and CNN models
3. Use leave-one-subject-out cross-validation
4. Run multiple trials per subject for statistical robustness
5. Save results to separate CSV files in `results/` directory

## Evaluation Methodology

Both scripts use **leave-one-subject-out cross-validation** to ensure proper evaluation:
- For each of the 6 subjects, use that subject's data as the test set
- Train on the remaining 5 subjects' data
- This prevents data leakage and provides realistic performance estimates

### Metrics Calculated

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro-averaged F1 score across all classes
- **Precision, Recall, Specificity**: Per-class performance metrics
- **Informedness**: Balanced measure of model performance

## Results

Results are saved in the `results/` directory:
- `traditional_ml_results.csv`: Comprehensive results for all traditional ML models
- `{MODEL}_{units}_{activation}_{learning_rate}.csv`: Individual results for each deep learning model

## Configuration

### Traditional ML Models
- Grid search parameters are defined within each model's evaluation function
- Cross-validation uses 5-fold CV for hyperparameter optimization
- Final evaluation uses leave-one-subject-out CV

### Deep Learning Models
- Learning rate: 0.01 (configurable in script)
- LSTM/GRU units: 20 (configurable)
- Activation function: sigmoid (configurable)
- Number of trials per subject: 10
- Early stopping: patience=50 epochs
- Batch size: 32

## Technical Notes

### Data Preprocessing
- **Traditional ML**: MinMaxScaler normalization, data flattened to feature vectors
- **Deep Learning**: StandardScaler standardization, maintains sequence structure
- **CNN**: Additional reshape to add channel dimension for Conv2D layers

### Model Architecture Details
- **LSTM/GRU**: Single recurrent layer followed by dense output layer
- **CNN**: Custom architecture designed for 1D time series with 2 channels
- All models output to 7 classes with softmax activation

## Citation

If you use this code in your research, please cite the original work and this repository (under review).

## License

This project is available under the MIT License.