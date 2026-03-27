import numpy as np
import pandas as pd
import gc
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_advanced_lstm(input_shape, units, dropout, lr, num_layers, activation_idx):
    """Builds a stacked LSTM based on GWO discovered parameters."""
    activations = ['relu', 'tanh', 'sigmoid']
    selected_act = activations[int(activation_idx)] 
    
    model = Sequential()
    for i in range(int(num_layers)):
        is_last_layer = (i == int(num_layers) - 1)
        if i == 0:
            model.add(LSTM(units=int(units), activation=selected_act, 
                           return_sequences=not is_last_layer, input_shape=input_shape))
        else:
            model.add(LSTM(units=int(units), activation=selected_act, 
                           return_sequences=not is_last_layer))
        model.add(Dropout(dropout))
    
    model.add(Dense(1))
    # Note: Compiling with MSE here. If you want to strictly optimize for MAPE, change loss='mse' to loss='mape'
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

def tune_lstm_with_gwo_advanced(X_train_2d, y_train, X_test_2d, y_test, n_wolves=5, iterations=20, seed=42):
    
    # --- 0. LOCK DOWN DETERMINISM ---
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # 1. Reshape FULL Data for LSTM (No downsampling)
    X_train_3d = np.reshape(X_train_2d, (X_train_2d.shape[0], 1, X_train_2d.shape[1]))
    X_test_3d = np.reshape(X_test_2d, (X_test_2d.shape[0], 1, X_test_2d.shape[1]))
    in_shape = (X_train_3d.shape[1], X_train_3d.shape[2])

    # 2. Search Space: [Units, Dropout, LR, Layers, Act_Idx]
    lb = np.array([16, 0.0, 0.0001, 1, 0])
    ub = np.array([256, 0.5, 0.01, 4, 2])
    dim = 5
    
    # Initialize wolves
    wolves = np.random.uniform(lb, ub, (n_wolves, dim))
    
    # SNAP TO GRID during initialization for discrete variables
    wolves[:, 3] = np.round(wolves[:, 3]) # Layers
    wolves[:, 4] = np.round(wolves[:, 4]) # Activation Index

    alpha_pos, alpha_score = np.zeros(dim), float('inf')
    beta_pos, beta_score = np.zeros(dim), float('inf')
    delta_pos, delta_score = np.zeros(dim), float('inf')

    print(f"Starting GWO Optimization on FULL {len(X_train_3d)} sequences...")

    for t in range(iterations):
        print(f"--- Iteration {t+1}/{iterations} ---")
        for i in range(n_wolves):
            # Clip bounds BEFORE evaluation
            wolves[i] = np.clip(wolves[i], lb, ub)
            p = wolves[i]
            
            # Brief training to evaluate fitness
            model = build_advanced_lstm(in_shape, p[0], p[1], p[2], p[3], p[4])
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
            
            # Train on the FULL dataset
            hist = model.fit(X_train_3d, y_train, validation_split=0.2,
                             epochs=10, batch_size= 64, verbose=0, callbacks=[early_stop])
            
            # Fitness evaluation
            fitness = min(hist.history['val_loss'])

            # Update Alpha, Beta, Delta
            if fitness < alpha_score:
                alpha_score, alpha_pos = fitness, wolves[i].copy()
            elif fitness < beta_score:
                beta_score, beta_pos = fitness, wolves[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, wolves[i].copy()

            # --- CRITICAL MEMORY LEAK FIX ---
            del model
            K.clear_session()
            gc.collect()

        # Checkpoint to disk after every iteration
        pd.DataFrame([alpha_pos], columns=['Units', 'Dropout', 'LR', 'Layers', 'Act_Idx']).to_csv('gwo_checkpoint.csv', index=False)

        # Update Wolf Positions (Standard GWO Logic + Discrete Snapping)
        a = 2 - t * (2 / iterations)
        for i in range(n_wolves):
            for j in range(dim):
                r1, r2 = np.random.random(), np.random.random()
                A1, C1 = 2*a*r1-a, 2*r2
                X1 = alpha_pos[j] - A1 * abs(C1 * alpha_pos[j] - wolves[i,j])
                
                r1, r2 = np.random.random(), np.random.random()
                A2, C2 = 2*a*r1-a, 2*r2
                X2 = beta_pos[j] - A2 * abs(C2 * beta_pos[j] - wolves[i,j])
                
                r1, r2 = np.random.random(), np.random.random()
                A3, C3 = 2*a*r1-a, 2*r2
                X3 = delta_pos[j] - A3 * abs(C3 * delta_pos[j] - wolves[i,j])
                
                new_pos = (X1 + X2 + X3) / 3

                # SNAP TO GRID during position update
                if j in [3, 4]:
                    wolves[i,j] = round(new_pos)
                else:
                    wolves[i,j] = new_pos

    # 3. Final Training and Predictions
    print(f"\nOptimization Complete. Best Parameters: Units={int(alpha_pos[0])}, Dropout={alpha_pos[1]:.2f}, LR={alpha_pos[2]:.4f}, Layers={int(alpha_pos[3])}, Act_Idx={int(alpha_pos[4])}")
    print("Training Final Model with Best Parameters...")
    
    best_model = build_advanced_lstm(in_shape, *alpha_pos)
    best_model.fit(X_train_3d, y_train, epochs=50, validation_split=0.2, 
                   callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)], 
                   verbose=1)
    
    y_train_pred = best_model.predict(X_train_3d)
    y_test_pred = best_model.predict(X_test_3d)
    
    return best_model, alpha_pos, y_train_pred, y_test_pred, X_train_3d, X_test_3d