import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout #type:ignore

# 1. Load your saved model
model = tf.keras.models.load_model('GWO_best_model.keras')

# 2. Extract Layer Information
lstm_layers = [layer for layer in model.layers if isinstance(layer, LSTM)]
dropout_layers = [layer for layer in model.layers if isinstance(layer, Dropout)]

num_layers = len(lstm_layers)
units = lstm_layers[0].units
# Keras stores activations as function objects, so we grab the __name__ attribute
activation = lstm_layers[0].activation.__name__ 
dropout_rate = dropout_layers[0].rate if dropout_layers else 0.0

# 3. Extract Optimizer Information
# We use .numpy() to convert the tensor value back to a standard Python float
learning_rate = model.optimizer.learning_rate.numpy()

# 4. Print the extracted blueprint
print("--- Extracted Model Hyperparameters ---")
print(f"LSTM Layers: {num_layers}")
print(f"Units per Layer: {units}")
print(f"Activation Function: {activation}")
print(f"Dropout Rate: {dropout_rate}")
print(f"Optimizer Learning Rate: {learning_rate:.6f}")