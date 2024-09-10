import tensorrt
import qutip as qt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from qutip import Qobj, wigner
from tqdm import tqdm
import time
import csv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.config.run_functions_eagerly(False)

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Utility functions
def get_vacuum_state_tf(dim):
    vacuum_state = qt.basis(dim, 0)
    return tf.convert_to_tensor(vacuum_state.full(), dtype=tf.complex64)

def annihilation(dim):
    diag_vals = tf.math.sqrt(tf.cast(tf.range(1, dim), dtype=tf.complex64))
    return tf.linalg.diag(diag_vals, k=1)

def number(dim):
    diag_vals = tf.cast(tf.range(0, dim), dtype=tf.complex64)
    return tf.linalg.diag(diag_vals)

def displacement_operator(dim, x, y=0):
    x2 = tf.identity(x)
    y2 = tf.identity(y)
    alpha = tf.complex(x2, y2)
    a = annihilation(dim)
    term1 = alpha * tf.linalg.adjoint(a)
    term2 = tf.math.conj(alpha) * a
    D = tf.linalg.expm(term1 - term2)
    return D

def displacement_encoding(dim, alpha_vec):
    alpha_vec = tf.cast(alpha_vec, dtype=tf.complex64)
    num = tf.shape(alpha_vec)[0]
    a = annihilation(dim)
    term1 = tf.linalg.adjoint(a)
    term2 = a
    term1_batch = tf.tile(tf.expand_dims(term1, 0), [num, 1, 1])
    term2_batch = tf.tile(tf.expand_dims(term2, 0), [num, 1, 1])
    alpha_vec = tf.reshape(alpha_vec, [-1, 1, 1])
    D = tf.linalg.expm(alpha_vec * term1_batch - tf.math.conj(alpha_vec) * term2_batch)
    return D

def rotation_operator(dim, theta):
    theta2 = tf.identity(theta)
    theta2 = tf.cast(theta2, dtype=tf.complex64)
    n = number(dim)
    R = tf.linalg.expm(-1j * theta2 * n)
    return R

def squeezing_operator(dim, r):
    r2 = tf.identity(r)
    r2 = tf.cast(r2, dtype=tf.complex64)
    a = annihilation(dim)
    term1 = r2 * tf.linalg.adjoint(a) * a
    term2 = tf.math.conj(r2) * a * tf.linalg.adjoint(a)
    S = tf.linalg.expm(0.5 * (term1 - term2))
    return S

def kerr_operator(dim, kappa):
    kappa2 = tf.identity(kappa)
    kappa2 = tf.cast(kappa2, dtype=tf.complex64)
    n = number(dim)
    K = tf.linalg.expm(1j * kappa2 * n * n)

    return K


# TensorFlow Custom Layer for Quantum Encoding
class QEncoder(tf.keras.layers.Layer):
    def __init__(self, dim, vacuum_state, **kwargs):
        super(QEncoder, self).__init__(**kwargs)
        self.dim = dim
        self.vacuum_state = tf.cast(vacuum_state, dtype=tf.complex64)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        batch_vacuum_state = tf.tile(tf.expand_dims(self.vacuum_state, axis=0), [batch_size, 1, 1])
        batch_displacement_operators = displacement_encoding(self.dim, inputs)
        displaced_states = tf.einsum('bij,bjk->bik', batch_displacement_operators, batch_vacuum_state)
        return displaced_states


# TensorFlow Custom Layer for Quantum Transformations
class QLayer(tf.keras.layers.Layer):
    def __init__(self, dim, stddev=0.1, **kwargs):
        super(QLayer, self).__init__(**kwargs)
        self.dim = dim
        self.stddev = stddev  # Standard deviation as a hyperparameter

    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.stddev, seed=42)
        self.theta_1 = self.add_weight("theta_1", shape=[1,], initializer=initializer, trainable=True)
        self.theta_2 = self.add_weight("theta_2", shape=[1,], initializer=initializer, trainable=True)
        self.r = self.add_weight("r", shape=[1,], initializer=initializer, trainable=True)
        self.bx = self.add_weight("bx", shape=[1,], initializer=initializer, trainable=True)
        self.bp = self.add_weight("bp", shape=[1,], initializer=initializer, trainable=True)
        self.kappa = self.add_weight("kappa", shape=[1,], initializer=initializer, trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        D_tensor = tf.expand_dims(displacement_operator(self.dim, self.bx, self.bp), 0)
        R_tensor_1 = tf.expand_dims(rotation_operator(self.dim, self.theta_1), 0)
        S_tensor = tf.expand_dims(squeezing_operator(self.dim, self.r), 0)
        R_tensor_2 = tf.expand_dims(rotation_operator(self.dim, self.theta_2), 0)
        K_tensor = tf.expand_dims(kerr_operator(self.dim, self.kappa), 0)

        D_tensor = tf.tile(D_tensor, [batch_size, 1, 1])
        R_tensor_1 = tf.tile(R_tensor_1, [batch_size, 1, 1])
        S_tensor = tf.tile(S_tensor, [batch_size, 1, 1])
        R_tensor_2 = tf.tile(R_tensor_2, [batch_size, 1, 1])
        K_tensor = tf.tile(K_tensor, [batch_size, 1, 1])

        transformed_state = tf.einsum('bij,bjk->bik', R_tensor_1, inputs)
        transformed_state = tf.einsum('bij,bjk->bik', S_tensor, transformed_state)
        transformed_state = tf.einsum('bij,bjk->bik', R_tensor_2, transformed_state)
        transformed_state = tf.einsum('bij,bjk->bik', D_tensor, transformed_state)
        activated_state = tf.einsum('bij,bjk->bik', K_tensor, transformed_state)
        
        self.final_state = activated_state
        return activated_state
    
    
# TensorFlow Custom Layer for Quantum Decoding
class QDecoder(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(QDecoder, self).__init__(**kwargs)
        self.dim = dim
        self.x_operator = self.build_x_operator()

    def build_x_operator(self):
        a = annihilation(self.dim)
        x_operator = (a + tf.linalg.adjoint(a)) / 2.0
        x_operator = tf.expand_dims(x_operator, axis=0)  # Add batch dimension
        return x_operator

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        batch_x_operator = tf.tile(self.x_operator, [batch_size, 1, 1])

        # Step 1: Compute \hat{O} | \psi \rangle for each state in the batch
        operator_applied_state = tf.einsum('bij,bjk->bik', batch_x_operator, inputs)

        # Take the complex conjugate of each state and adjust dimensions
        conj_inputs = tf.math.conj(inputs)  # Shape: (batch_size, dim, 1)
        conj_inputs_adj = tf.transpose(conj_inputs, perm=[0, 2, 1])  # Shape: (batch_size, 1, dim)

        # Compute the expectation value (inner product) for each state in the batch
        x_expectation = tf.einsum('bij,bjk->bi', conj_inputs_adj, operator_applied_state) 
        x_expectation = tf.squeeze(x_expectation, axis=-1)

        return tf.math.real(x_expectation)
    
# R2Score metric wrapper to handle shape issues
class R2ScoreWrapper(tf.keras.metrics.R2Score):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, 1])
        return super().update_state(y_true, y_pred, sample_weight)


# TensorFlow Custom Callback for Progress Bars
from IPython.display import clear_output

class TrainingProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch += 1  # epochs are zero-indexed in this method
        
        # Get training loss, validation loss, and learning rate
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        lr = self.model.optimizer.lr
        
        # If lr is a callable (LearningRateSchedule), get its current value
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)
        
        # Convert lr tensor to float
        lr = float(lr)

        print(f"Epoch: {epoch:5d} | LR: {lr:.10f} | Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}")

        # Every 5 epochs, clear the screen
        if epoch % 5 == 0:
            clear_output(wait=True)

# TensorFlow Custom Callback for Parameter Logging
class ParameterLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, fold, function_index, x_train, y_train, base_dir='Params'):
        super(ParameterLoggingCallback, self).__init__()
        self.fold = fold
        self.function_index = function_index
        self.x_train = x_train
        self.y_train = y_train
        self.base_dir = base_dir
        self.params_dir = os.path.join(base_dir, f'Function_{function_index}')
        self.filename = os.path.join(self.params_dir, f'parameters_fold_{fold}.csv')
        self.epoch = 0
        
        # Create directory if it doesn't exist
        os.makedirs(self.params_dir, exist_ok=True)
        
    def on_train_begin(self, logs=None):
        # Create the CSV file for parameters and write the header
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'theta_1', 'r', 'theta_2', 'bx', 'bp', 'kappa'])
        
        # Create a CSV file for training data
        train_data_file = os.path.join(self.params_dir, f'train_data_fold_{self.fold}.csv')
        with open(train_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for x, y in zip(self.x_train, self.y_train):
                # Assuming x is a single real number and y is the target (expected output)
                writer.writerow([x, y])
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        params = []
        for layer in self.model.layers:
            if isinstance(layer, QLayer):
                params.extend([
                    layer.theta_1.numpy()[0],
                    layer.r.numpy()[0],
                    layer.theta_2.numpy()[0],
                    layer.bx.numpy()[0],
                    layer.bp.numpy()[0],
                    layer.kappa.numpy()[0]
                ])
        
        # Append the parameters to the CSV file
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.epoch] + params)


# TensorFlow Custom Layer for Classical Phase Space Transformations
class CPLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CPLayer, self).__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
        
        # Initialize the weights for the rotations, squeezing, translations, and nonlinear activation
        self.theta1 = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.theta2 = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.r = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.kappa = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.b = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        
    def call(self, inputs):
        # Unpack x and p components
        x, p = inputs[..., 0], inputs[..., 1]
        
        # Apply the first rotation
        x_rot = x * tf.cos(self.theta1) - p * tf.sin(self.theta1)
        p_rot = x * tf.sin(self.theta1) + p * tf.cos(self.theta1)
        
        # Apply squeezing
        x_squeezed = tf.exp(-self.r) * x_rot
        p_squeezed = tf.exp(self.r) * p_rot
        
        # Apply the second rotation
        x_rot2 = x_squeezed * tf.cos(self.theta2) - p_squeezed * tf.sin(self.theta2)
        p_rot2 = x_squeezed * tf.sin(self.theta2) + p_squeezed * tf.cos(self.theta2)
        
        # Apply translation
        x_translated = x_rot2 + self.b
        p_translated = p_rot2
        
        # Calculate radius for the activation
        radius = tf.sqrt(x_translated**2 + p_translated**2)
        
        # Apply the nonlinear activation
        x_activated = x_translated * tf.cos(self.kappa * radius) - p_translated * tf.sin(self.kappa * radius)
        p_activated = x_translated * tf.sin(self.kappa * radius) + p_translated * tf.cos(self.kappa * radius)
        
        # Stack the transformed components together to return a 2D output
        outputs = tf.stack([x_activated, p_activated], axis=-1)
        return outputs
    
    
# TensorFlow Custom Layer for Extracting Positions
class ExtractXLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ExtractXLayer, self).__init__()

    def call(self, inputs):
        # Assuming the 'x' values are the first component in the (N, 2) input
        # Extracts and returns the 'x' component in shape (N, 1)
        return tf.expand_dims(inputs[..., 0], axis=-1)
    
        
# TensorFlow Custom Callback for Progress Bars
class TQDMProgressBar(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progress_bar = tqdm(total=self.epochs, desc=f"Epoch 1/{self.epochs}")
        self.start_time = time.time()  # Start time

    def on_epoch_end(self, epoch, logs=None):
        description = f"Epoch {epoch+1}/{self.epochs}"
        self.progress_bar.set_description(description)
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        self.progress_bar.close()
        self.end_time = time.time()  # End time
        self.total_time = self.end_time - self.start_time  # Total computation time
        print(f"Total training time: {self.total_time:.2f} seconds")


from sklearn.model_selection import KFold

def train_model(input_data, target_data, function_index, k_folds=5, learning_rate = 0.01, std=0.1, cutoff_dim=10, num_layers=2, epochs=200):
    fold_var = 1
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f'Training model for Function {function_index} with {num_layers} layers for {epochs} epochs...')
    
    for train_index, val_index in kf.split(input_data):
        # Split data into training and validation for the current fold
        x_train_fold, x_val_fold = input_data[train_index], input_data[val_index]
        y_train_fold, y_val_fold = target_data[train_index], target_data[val_index]

        # Create a new model for each fold
        vacuum_state = get_vacuum_state_tf(cutoff_dim)
        model = tf.keras.Sequential([QEncoder(dim=cutoff_dim, vacuum_state=vacuum_state, name='QuantumEncoding')])
        for i in range(num_layers):
            model.add(QLayer(dim=cutoff_dim, stddev=std, name=f'QuantumLayer_{i+1}'))
        model.add(QDecoder(dim=cutoff_dim, name='QuantumDecoding'))

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0)
        model.compile(optimizer=opt, loss='mse', metrics=[R2ScoreWrapper()])
        
        # Create the ParameterLoggingCallback
        param_logger = ParameterLoggingCallback(fold_var, function_index, x_train_fold, y_train_fold)
        
        # Train the model
        print(f'Training on fold {fold_var}...')
        history = model.fit(x_train_fold, y_train_fold, validation_data=(x_val_fold, y_val_fold), 
                            epochs=epochs, verbose=0, callbacks=[TrainingProgress(), param_logger])

        print(f'Training on fold {fold_var} complete.')
        fold_var += 1

    print('Training Complete.')
    model.summary()

    return history, model
        

# Function for training classical models
def train_classical_model(input_data, target_data, function_index, k_folds=5, learning_rate = 0.01, std=0.1, num_layers = 2, epochs=200):
    fold_var = 1
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f'Training model for Function {function_index} with {num_layers} layers for {epochs} epochs...')

    for train_index, val_index in kf.split(input_data):
        # Split data into training and validation for the current fold
        x_train_fold, x_val_fold = input_data[train_index], input_data[val_index]
        y_train_fold, y_val_fold = target_data[train_index], target_data[val_index]

        # Create a new model for each configuration
        layers = [CPLayer() for _ in range(num_layers)] + [ExtractXLayer()]
        model = tf.keras.Sequential(layers)

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0)
        model.compile(optimizer=opt, loss='mse', metrics=[R2ScoreWrapper()])
        
        print(f'Training model with {num_layers} layers for {epochs} epochs...')
        progress_bar = TQDMProgressBar()

        # Train the model
        print(f'Training on fold {fold_var}...')
        history = model.fit(x_train_fold, y_train_fold, validation_data=(x_val_fold, y_val_fold), 
                            epochs=epochs, verbose=0, callbacks=[progress_bar])

        print(f'Training on fold {fold_var} complete.')
        fold_var += 1

    print('Training Complete.')
    model.summary()

    return history, model