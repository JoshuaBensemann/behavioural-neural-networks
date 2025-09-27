"""
Traditional Machine Learning Models for Behavioural Neural Networks Experiment

This script implements and evaluates several traditional machine learning models
using leave-one-subject-out cross-validation on pigeon behavioral data.

Models evaluated:
- Random Forest
- Decision Tree
- Logistic Regression
- K-Nearest Neighbors
"""

import os
from os import scandir
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

DATA = 'Pigeon Samples Used for Experiments/'
SERIES_LEN = 90

def load_and_prepare_data():
    """Load and prepare the pigeon behavioral data for traditional ML models."""
    print("Loading and preparing data...")
    
    files = [f"{DATA}{file.name}" for file in scandir(DATA) if 'sam' in file.name]
    
    # Collect all data for scaling
    all_data = []
    for file in files:
        temp = pd.read_csv(file, header=None)
        all_data.append(temp)
    
    all_data_combined = pd.concat(all_data, ignore_index=True)
    scaler = MinMaxScaler()
    scaler.fit(all_data_combined)
    
    # Process each file
    processed_data = []
    for file in files:
        x = pd.read_csv(file, header=None)
        x = scaler.transform(x)
        
        # Pad sequences to consistent length
        missing = SERIES_LEN - x.shape[0]
        if missing > 0:
            x = np.pad(x, ((0, missing), (0, 0)))
        elif missing < 0:
            x = x[:SERIES_LEN]
        
        # Extract label from filename (assumes format like 'sam101.csv' where 1 is the class)
        y = int(file[-5]) - 1  # Convert to 0-indexed
        subject = file[-7]  # Extract subject number
        
        processed_data.append((subject, x, y))
    
    # Create feature matrix
    subjects = [data[0] for data in processed_data]
    X_features = []
    y_labels = []
    
    for subject, x, y in processed_data:
        # Flatten the time series data
        flattened = x.flatten()
        X_features.append(flattened)
        y_labels.append(y)
    
    X = pd.DataFrame(X_features, columns=[f'feature_{i}' for i in range(len(X_features[0]))])
    X['Subject'] = subjects
    y = pd.DataFrame(y_labels, columns=['y'])
    
    print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]-1} features, {len(np.unique(y))} classes")
    return X, y

def evaluate_random_forest(X, y):
    """Evaluate Random Forest with grid search."""
    print("\n=== Random Forest ===")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Grid search on all data (excluding subject column)
    X_grid = X.drop(columns=['Subject'])
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro', verbose=2)
    
    print("Performing grid search...")
    grid_search.fit(X_grid, y.values.ravel())
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Leave-one-subject-out evaluation
    rf_accuracies = []
    rf_f1_scores = []
    
    print("Evaluating with leave-one-subject-out cross-validation...")
    for subject in range(1, 7):
        train_mask = X['Subject'] != str(subject)
        test_mask = X['Subject'] == str(subject)
        
        train_x = X[train_mask].drop(columns=['Subject'])
        train_y = y[train_mask].values.ravel()
        test_x = X[test_mask].drop(columns=['Subject'])
        test_y = y[test_mask].values.ravel()
        
        rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
        rf.fit(train_x, train_y)
        
        predictions = rf.predict(test_x)
        accuracy = accuracy_score(test_y, predictions)
        f1 = f1_score(test_y, predictions, average='macro')
        
        rf_accuracies.append(accuracy)
        rf_f1_scores.append(f1)
        
        print(f"Subject {subject} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    print(f"Average Accuracy: {np.mean(rf_accuracies):.4f} ± {np.std(rf_accuracies):.4f}")
    print(f"Average F1: {np.mean(rf_f1_scores):.4f} ± {np.std(rf_f1_scores):.4f}")
    
    return rf_accuracies, rf_f1_scores

def evaluate_decision_tree(X, y):
    """Evaluate Decision Tree with grid search."""
    print("\n=== Decision Tree ===")
    
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]
    }
    
    X_grid = X.drop(columns=['Subject'])
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro', verbose=2)
    
    print("Performing grid search...")
    grid_search.fit(X_grid, y.values.ravel())
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Leave-one-subject-out evaluation
    dt_accuracies = []
    dt_f1_scores = []
    
    print("Evaluating with leave-one-subject-out cross-validation...")
    for subject in range(1, 7):
        train_mask = X['Subject'] != str(subject)
        test_mask = X['Subject'] == str(subject)
        
        train_x = X[train_mask].drop(columns=['Subject'])
        train_y = y[train_mask].values.ravel()
        test_x = X[test_mask].drop(columns=['Subject'])
        test_y = y[test_mask].values.ravel()
        
        dt = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
        dt.fit(train_x, train_y)
        
        predictions = dt.predict(test_x)
        accuracy = accuracy_score(test_y, predictions)
        f1 = f1_score(test_y, predictions, average='macro')
        
        dt_accuracies.append(accuracy)
        dt_f1_scores.append(f1)
        
        print(f"Subject {subject} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    print(f"Average Accuracy: {np.mean(dt_accuracies):.4f} ± {np.std(dt_accuracies):.4f}")
    print(f"Average F1: {np.mean(dt_f1_scores):.4f} ± {np.std(dt_f1_scores):.4f}")
    
    return dt_accuracies, dt_f1_scores

def evaluate_logistic_regression(X, y):
    """Evaluate Logistic Regression with grid search."""
    print("\n=== Logistic Regression ===")
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l1', 'l2'],
        'max_iter': [1000, 2000]
    }
    
    X_grid = X.drop(columns=['Subject'])
    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro', verbose=2)
    
    print("Performing grid search...")
    grid_search.fit(X_grid, y.values.ravel())
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Leave-one-subject-out evaluation
    lr_accuracies = []
    lr_f1_scores = []
    
    print("Evaluating with leave-one-subject-out cross-validation...")
    for subject in range(1, 7):
        train_mask = X['Subject'] != str(subject)
        test_mask = X['Subject'] == str(subject)
        
        train_x = X[train_mask].drop(columns=['Subject'])
        train_y = y[train_mask].values.ravel()
        test_x = X[test_mask].drop(columns=['Subject'])
        test_y = y[test_mask].values.ravel()
        
        lr = LogisticRegression(**grid_search.best_params_, random_state=42)
        lr.fit(train_x, train_y)
        
        predictions = lr.predict(test_x)
        accuracy = accuracy_score(test_y, predictions)
        f1 = f1_score(test_y, predictions, average='macro')
        
        lr_accuracies.append(accuracy)
        lr_f1_scores.append(f1)
        
        print(f"Subject {subject} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    print(f"Average Accuracy: {np.mean(lr_accuracies):.4f} ± {np.std(lr_accuracies):.4f}")
    print(f"Average F1: {np.mean(lr_f1_scores):.4f} ± {np.std(lr_f1_scores):.4f}")
    
    return lr_accuracies, lr_f1_scores

def evaluate_knn(X, y):
    """Evaluate K-Nearest Neighbors with grid search."""
    print("\n=== K-Nearest Neighbors ===")
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    X_grid = X.drop(columns=['Subject'])
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro', verbose=2)
    
    print("Performing grid search...")
    grid_search.fit(X_grid, y.values.ravel())
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Leave-one-subject-out evaluation
    knn_accuracies = []
    knn_f1_scores = []
    
    print("Evaluating with leave-one-subject-out cross-validation...")
    for subject in range(1, 7):
        train_mask = X['Subject'] != str(subject)
        test_mask = X['Subject'] == str(subject)
        
        train_x = X[train_mask].drop(columns=['Subject'])
        train_y = y[train_mask].values.ravel()
        test_x = X[test_mask].drop(columns=['Subject'])
        test_y = y[test_mask].values.ravel()
        
        knn = KNeighborsClassifier(**grid_search.best_params_)
        knn.fit(train_x, train_y)
        
        predictions = knn.predict(test_x)
        accuracy = accuracy_score(test_y, predictions)
        f1 = f1_score(test_y, predictions, average='macro')
        
        knn_accuracies.append(accuracy)
        knn_f1_scores.append(f1)
        
        print(f"Subject {subject} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    print(f"Average Accuracy: {np.mean(knn_accuracies):.4f} ± {np.std(knn_accuracies):.4f}")
    print(f"Average F1: {np.mean(knn_f1_scores):.4f} ± {np.std(knn_f1_scores):.4f}")
    
    return knn_accuracies, knn_f1_scores

def save_results(results_dict, filename):
    """Save results to CSV file."""
    os.makedirs('results', exist_ok=True)
    
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'results/{filename}', index=False)
    print(f"Results saved to results/{filename}")

def main():
    """Main execution function."""
    print("Traditional ML Models Evaluation")
    print("=" * 50)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Evaluate models
    rf_acc, rf_f1 = evaluate_random_forest(X, y)
    dt_acc, dt_f1 = evaluate_decision_tree(X, y)
    lr_acc, lr_f1 = evaluate_logistic_regression(X, y)
    knn_acc, knn_f1 = evaluate_knn(X, y)
    
    # Compile results
    results = {
        'Subject': list(range(1, 7)),
        'RF_Accuracy': rf_acc,
        'RF_F1': rf_f1,
        'DT_Accuracy': dt_acc,
        'DT_F1': dt_f1,
        'LR_Accuracy': lr_acc,
        'LR_F1': lr_f1,
        'KNN_Accuracy': knn_acc,
        'KNN_F1': knn_f1
    }
    
    # Save results
    save_results(results, 'traditional_ml_results.csv')
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY RESULTS")
    print("=" * 50)
    print(f"Random Forest    - Avg Accuracy: {np.mean(rf_acc):.4f}, Avg F1: {np.mean(rf_f1):.4f}")
    print(f"Decision Tree    - Avg Accuracy: {np.mean(dt_acc):.4f}, Avg F1: {np.mean(dt_f1):.4f}")
    print(f"Logistic Reg     - Avg Accuracy: {np.mean(lr_acc):.4f}, Avg F1: {np.mean(lr_f1):.4f}")
    print(f"K-NN             - Avg Accuracy: {np.mean(knn_acc):.4f}, Avg F1: {np.mean(knn_f1):.4f}")

if __name__ == "__main__":
    main()