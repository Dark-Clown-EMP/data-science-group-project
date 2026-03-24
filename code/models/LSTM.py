import math

import numpy as np
from tensorflow import keras #type:ignore
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout #type:ignore
import keras_tuner as kt
from sklearn.metrics import mean_squared_error



def tune_lstm_model(X_train_2d, y_train, X_test_2d, y_test, max_trials=10):
    """
    Takes 2D feature matrices, reshapes them for LSTM, and performs hyperparameter tuning.
    """
    print("Reshaping data for LSTM...")
    # 1. Reshape from (Samples, Features) to (Samples, Timesteps, Features)
    # Since lags were flattened into columns, Timesteps is 1.
    X_train_3d = np.reshape(X_train_2d, (X_train_2d.shape[0], 1, X_train_2d.shape[1]))
    X_test_3d = np.reshape(X_test_2d, (X_test_2d.shape[0], 1, X_test_2d.shape[1]))
    
    # 2. Define the model builder inside the function so it can dynamically read the 3D shape
    def build_model(hp):
        model = Sequential()
        
        # Tune LSTM units
        hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(LSTM(units=hp_units, activation='relu', input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])))
        
        # Tune Dropout
        hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1)
        model.add(Dropout(hp_dropout))
        
        model.add(Dense(1))
        
        # Tune Learning Rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='mse',
                      metrics=['mae'])
        return model

    # 3. Initialize KerasTuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='tuning_results',
        seed = 42,
        project_name='energy_demand_lstm'
    )
    
    # 4. Define Early Stopping to prevent wasting time on bad epochs
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    print("Starting hyperparameter search...")
    # 5. Execute the search
    tuner.search(
        X_train_3d, y_train,
        epochs=20,
        validation_data=(X_test_3d, y_test),
        callbacks=[early_stopping],
        batch_size=32,
        verbose=1
    )
    
    # 6. Extract the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n--- Tuning Complete: Optimal Hyperparameters ---")
    print(f"LSTM Units: {best_hps.get('units')}")
    print(f"Dropout Rate: {best_hps.get('dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    
    # Build the final model architecture using the best parameters
    best_model = tuner.hypermodel.build(best_hps)
    
    best_model.fit(
        X_train_3d, y_train,
        epochs=50, # You can increase this since early stopping protects you
        validation_split=0.2,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
    )

    print("\nGenerating predictions...")
    # 8. Predict on train and test sets
    y_train_pred = best_model.predict(X_train_3d)
    y_test_pred = best_model.predict(X_test_3d)

    # 9. Calculate RMSE
    train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n--- Model Performance ---")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    # Return the parameters, the model, and the reshaped data so you can train it next
    return best_hps, best_model, X_train_3d, X_test_3d, y_test_pred, y_train_pred

# --- How to call the function ---
# Assuming X_train, y_train, X_test, y_test are already created from your previous code:
# best_params, optimal_model, X_train_3d, X_test_3d = tune_lstm_model(X_train, y_train, X_test, y_test, max_trials=10)