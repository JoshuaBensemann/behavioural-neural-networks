"""
Deep Learning Models for Behavioural Neural Networks Experiment

This script implements and evaluates deep learning models (LSTM, GRU, and CNN)
using leave-one-subject-out cross-validation on pigeon behavioral data.

Models evaluated:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- CNN (Convolutional Neural Network)
"""

import os
from os import scandir
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

DATA = 'Pigeon Samples Used for Experiments/'
SERIES_LEN = 90

def get_generators(files, val_bird=6, series_len=90):
    """Generate training and validation data for a specific validation bird."""
    print(f'Validation Bird {val_bird}')
    train_files = [file for file in files if int(file[-7]) != val_bird]
    val_files = [file for file in files if file not in train_files]

    # Fit scaler on training data only
    all_train_data = []
    for file in train_files:
        temp = pd.read_csv(file, header=None)
        all_train_data.append(temp)
    
    combined_data = pd.concat(all_train_data, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(combined_data)
    
    def prepare_data(file):
        x = pd.read_csv(file, header=None)
        x = scaler.transform(x)
        missing = series_len - x.shape[0]
        if missing > 0:
            x = np.pad(x, ((0, missing), (0, 0)))
        elif missing < 0:
            x = x[:series_len]
        
        # Ensure dtypes and shapes compatible with tf.data and keras
        x = x.astype(np.float32)
        y = np.array(int(file[-5]) - 1, dtype=np.int32)
        return x, y

    def train_gen():
        count = 0
        for file in train_files:
            x, y = prepare_data(file)
            if x.shape[0] == series_len:
                count += 1
                yield x, y
        print(f"  Training samples: {count}")
            
    def val_gen():
        count = 0
        for file in val_files:
            x, y = prepare_data(file)
            if x.shape[0] == series_len:
                count += 1
                yield x, y
        print(f"  Validation samples: {count}")
            
    return train_gen, val_gen

def get_labels_predictions(dataset, model):
    """Extract labels and predictions from dataset."""
    preds = []
    labels = []

    for x, y in dataset:
        batch_preds = model.predict(x, verbose=0)
        preds.append(np.argmax(batch_preds, axis=1))
        labels.append(y.numpy())
    
    preds = np.concatenate(preds)
    labels = np.concatenate(labels, axis=0)
    
    return labels, preds

def calculate_summary_stats(labels, preds, name):
    """Calculate comprehensive performance statistics."""
    cnf_matrix = confusion_matrix(labels, preds)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Avoid division by zero
    TPR = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) != 0)
    TNR = np.divide(TN, TN + FP, out=np.zeros_like(TN), where=(TN + FP) != 0)
    PPV = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0)
    FPR = np.divide(FP, FP + TN, out=np.zeros_like(FP), where=(FP + TN) != 0)
    
    Recall = TPR
    Informedness = TPR - FPR
    ACC = (TP + TN) / (TP + FP + FN + TN)
    
    f1 = np.divide(2 * PPV * Recall, PPV + Recall, out=np.zeros_like(PPV), where=(PPV + Recall) != 0)
    
    return {f"{name} Specificity": TNR, f"{name} Accuracy": ACC, f"{name} Recall": Recall,
            f"{name} Precision": PPV, f"{name} Informedness": Informedness, f"{name} F1": f1}

def get_printout_info(model, dataset, val_dataset):
    """Get comprehensive evaluation information."""
    eval_train = model.evaluate(dataset, verbose=0)
    eval_val = model.evaluate(val_dataset, verbose=0)
    
    train_labels, train_preds = get_labels_predictions(dataset, model)
    val_labels, val_preds = get_labels_predictions(val_dataset, model)
    
    train_summary = calculate_summary_stats(train_labels, train_preds, '')
    val_summary = calculate_summary_stats(val_labels, val_preds, 'Val')
    
    return (eval_train, eval_val, train_summary, val_summary)

def format_summary(summary_data):
    """Format summary data for saving."""
    eval_train, eval_val, train_summary, val_summary = summary_data
    
    formatted_data = {"Loss": eval_train[0], "Acc_Total": eval_train[1], 
                      "Val_Loss": eval_val[0], "Val_Acc_Total": eval_val[1]}
    
    for index, current_data in train_summary.items():
        for i in range(current_data.shape[0]):
            formatted_data[f'{index}_{i+1}'] = current_data[i]
            
    for index, current_data in val_summary.items():
        for i in range(current_data.shape[0]):
            formatted_data[f'{index}_{i+1}'] = current_data[i]            
        
    return formatted_data

def create_cnn_model(learning_rate):
    """
    Create CNN model with specified architecture:
    Input [90, 2] -> Conv2D [82, 20] -> MaxPooling [41, 20] -> Flatten [820] -> Dense [7]
    """
    model = keras.Sequential([
        keras.layers.InputLayer([90, 2, 1]),  # Need to add channel dimension for Conv2D
        keras.layers.Conv2D(filters=20, kernel_size=(9, 2), activation="sigmoid", 
                           input_shape=(90, 2, 1)),  # Output: [82, 1, 20]
        keras.layers.Reshape((82, 20)),  # Reshape to [82, 20] for MaxPooling1D
        keras.layers.MaxPooling1D(pool_size=2),  # Output: [41, 20]
        keras.layers.Flatten(),  # Output: [820]
        keras.layers.Dense(7, activation="softmax")  # Output: [7]
    ])
    return model

def run_trial(dataset, val_dataset, model_func, learning_rate, model_name):
    """Run a single trial with specified model."""
    print(f"  Running trial for {model_name}...")
    
    model = model_func(learning_rate)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['SparseCategoricalAccuracy'])

    callback = keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)

    # For CNN, we need to reshape the data to add channel dimension
    if model_name == "CNN":
        # Reshape datasets for CNN
        dataset_reshaped = dataset.map(lambda x, y: (tf.expand_dims(x, -1), y))
        val_dataset_reshaped = val_dataset.map(lambda x, y: (tf.expand_dims(x, -1), y))
        
        model.fit(x=dataset_reshaped, epochs=1000, 
                  validation_data=val_dataset_reshaped, 
                  callbacks=[callback], verbose=1)
        
        summary_data = get_printout_info(model, dataset_reshaped, val_dataset_reshaped)
    else:
        model.fit(x=dataset, epochs=1000, validation_data=val_dataset, 
                  callbacks=[callback], verbose=1)
        
        summary_data = get_printout_info(model, dataset, val_dataset)
    
    trial_summary = format_summary(summary_data)
    
    return trial_summary

def create_lstm_model(units, activation, learning_rate):
    """Create LSTM model."""
    model = keras.Sequential([
        keras.layers.InputLayer([90, 2]),
        keras.layers.LSTM(units, activation=activation),
        keras.layers.Dense(7, activation="softmax")
    ])
    return model

def create_gru_model(units, activation, learning_rate):
    """Create GRU model."""
    model = keras.Sequential([
        keras.layers.InputLayer([90, 2]),
        keras.layers.GRU(units, activation=activation),
        keras.layers.Dense(7, activation="softmax")
    ])
    return model

def main():
    """Main execution function."""
    print("Deep Learning Models Evaluation")
    print("=" * 50)
    
    files = [f"{DATA}{file.name}" for file in scandir(DATA) if 'sam' in file.name]
    
    learning_rates = [0.01]
    activations = ["sigmoid"]
    units = [20]
    n_trials = 10

    # Model configurations
    models = {
        "LSTM": lambda lr: create_lstm_model(units[0], activations[0], lr),
        "GRU": lambda lr: create_gru_model(units[0], activations[0], lr),
        "CNN": create_cnn_model
    }

    # Create results directory
    os.makedirs('results', exist_ok=True)

    for learning_rate in learning_rates:
        for name, model_func in models.items():
            print(f"\n=== Evaluating {name} ===")
            
            data_output = {}

            for bird in range(1, 7):
                print(f"\nProcessing Bird {bird}...")

                train_gen, val_gen = get_generators(files, val_bird=bird)

                # Run multiple trials per bird
                for trial in range(1, n_trials + 1):
                    print(f"  Trial {trial}/{n_trials}")

                    # Create datasets
                    dataset = tf.data.Dataset.from_generator(
                        train_gen,
                        output_signature=(
                            tf.TensorSpec(shape=(90, 2), dtype=tf.float32),
                            tf.TensorSpec(shape=(), dtype=tf.int32),
                        ),
                    )
                    dataset = dataset.batch(32)

                    val_dataset = tf.data.Dataset.from_generator(
                        val_gen,
                        output_signature=(
                            tf.TensorSpec(shape=(90, 2), dtype=tf.float32),
                            tf.TensorSpec(shape=(), dtype=tf.int32),
                        ),
                    )
                    val_dataset = val_dataset.batch(32)

                    # Run trial
                    trial_results = run_trial(dataset, val_dataset, model_func, 
                                            learning_rate, name)
                    
                    data_output[f'Val_Bird_{bird}_T{trial}'] = trial_results

            # Save results
            df = pd.DataFrame.from_dict(data_output, orient='index')
            filename = f'results/{name}_{units[0]}_{activations[0]}_{learning_rate}.csv'
            df.to_csv(filename)
            print(f"Results saved to {filename}")

    print("\n" + "=" * 50)
    print("Deep learning model evaluation completed!")
    print("Results saved in the 'results/' directory.")

if __name__ == "__main__":
    main()