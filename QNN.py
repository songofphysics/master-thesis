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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


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
    adag = tf.linalg.adjoint(a)
    term1 = tf.math.conj(r2) * (a @ a)
    term2 = r2 * (adag @ adag)
    S = tf.linalg.expm(0.5 * (term1 - term2))
    return S

def kerr_operator(dim, kappa):
    kappa2 = tf.identity(kappa)
    kappa2 = tf.cast(kappa2, dtype=tf.complex64)
    n = number(dim)
    K = tf.linalg.expm(1j * kappa2 * (n @ n))

    return K

def cubic_phase_operator(dim, gamma):
    a = annihilation(dim)
    x = (a + tf.linalg.adjoint(a)) / 2.0
    gamma2 = tf.identity(gamma)
    gamma2 = tf.cast(gamma2, dtype=tf.complex64)
    V = tf.linalg.expm(1j * (gamma2/3) * (x @ x @ x))

    return V 


# TensorFlow Custom Layer for Quantum Encoding
class QEncoder(tf.keras.layers.Layer):
    def __init__(self, dim, vacuum_state, **kwargs):
        super(QEncoder, self).__init__(**kwargs)
        self.dim = dim
        self.vacuum_state = tf.cast(vacuum_state, dtype=tf.complex64)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        squeezed_vacuum_state = tf.matmul(squeezing_operator(self.dim, 100.0), self.vacuum_state)
        batch_squeezed_state = tf.tile(tf.expand_dims(squeezed_vacuum_state, axis=0), [batch_size, 1, 1])
        batch_displacement_operators = displacement_encoding(self.dim, inputs/2)
        displaced_states = tf.einsum('bij,bjk->bik', batch_displacement_operators, batch_squeezed_state)
        norm = tf.sqrt(tf.reduce_sum(tf.abs(displaced_states)**2, axis=1, keepdims=True))
        norm = tf.cast(norm, dtype=tf.complex64)
        return displaced_states/norm


# TensorFlow Custom Layer for Quantum Transformations
class QLayer(tf.keras.layers.Layer):
    def __init__(self, dim, stddev=0.05, tol=1e-9, activation='kerr', **kwargs):
        super(QLayer, self).__init__(**kwargs)
        self.dim = dim
        self.stddev = stddev
        self.tol = tol
        self.activation = activation.lower()
        if self.activation not in ['kerr', 'cubicphase']:
            raise ValueError("Activation must be either 'kerr' or 'cubicphase'")

    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.stddev, seed=42)
        self.theta_1 = self.add_weight("theta_1", shape=[1,], initializer=initializer, trainable=True)
        self.theta_2 = self.add_weight("theta_2", shape=[1,], initializer=initializer, trainable=True)
        self.r = self.add_weight("r", shape=[1,], initializer=initializer, trainable=True)
        self.bx = self.add_weight("bx", shape=[1,], initializer=initializer, trainable=True)
        self.bp = self.add_weight("bp", shape=[1,], initializer=initializer, trainable=True)
        
        if self.activation == 'kerr':
            self.kappa = self.add_weight("kappa", shape=[1,], initializer=initializer, trainable=True)
        else:  # cubicphase
            self.gamma = self.add_weight("gamma", shape=[1,], initializer=initializer, trainable=True)

    @tf.function
    def check_norm(self, states):
        norm = tf.sqrt(tf.reduce_sum(tf.abs(states)**2, axis=1, keepdims=True))
        return tf.reduce_all(tf.abs(norm - 1) < self.tol)

    @tf.function
    def apply_and_check(self, operator, state):
        new_state = tf.einsum('bij,bjk->bik', operator, state)
        is_valid = self.check_norm(new_state)
        return tf.cond(is_valid, lambda: new_state, lambda: state)

    @tf.function
    def quantum_operation(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Create operators
        R_tensor_1 = tf.tile(tf.expand_dims(rotation_operator(self.dim, self.theta_1), 0), [batch_size, 1, 1])
        S_tensor = tf.tile(tf.expand_dims(squeezing_operator(self.dim, self.r), 0), [batch_size, 1, 1])
        R_tensor_2 = tf.tile(tf.expand_dims(rotation_operator(self.dim, self.theta_2), 0), [batch_size, 1, 1])
        D_tensor = tf.tile(tf.expand_dims(displacement_operator(self.dim, self.bx, self.bp), 0), [batch_size, 1, 1])
        
        if self.activation == 'kerr':
            A_tensor = tf.tile(tf.expand_dims(kerr_operator(self.dim, self.kappa), 0), [batch_size, 1, 1])
        else:  # cubicphase
            A_tensor = tf.tile(tf.expand_dims(cubic_phase_operator(self.dim, self.gamma), 0), [batch_size, 1, 1])

        # Apply operations and check norm at each step
        state = self.apply_and_check(R_tensor_1, inputs)
        state = self.apply_and_check(S_tensor, state)
        state = self.apply_and_check(R_tensor_2, state)
        state = self.apply_and_check(D_tensor, state)
        state = self.apply_and_check(A_tensor, state)
        return state
    
    def call(self, inputs):
        return self.quantum_operation(inputs)
        
    
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

        print(f"Epoch: {epoch:5d} | LR: {lr:.7f} | Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}")

        # Every 5 epochs, clear the screen
        if epoch % 5 == 0:
            clear_output(wait=True)


# TensorFlow Custom Callback for Parameter Logging
class ParameterLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, fold, function_index, activation, base_dir='Params'):
        super(ParameterLoggingCallback, self).__init__()
        self.fold = fold
        self.function_index = function_index
        self.activation = activation
        self.base_dir = base_dir
        self.params_dir = os.path.join(base_dir, f'Function_{function_index}')
        self.filename = os.path.join(self.params_dir, f'parameters_fold_{fold}.csv')
        self.epoch = 0
        
        # Create directory if it doesn't exist
        os.makedirs(self.params_dir, exist_ok=True)
        
    def on_train_begin(self, logs=None):
        # Count the number of QLayers
        self.num_qlayers = sum(1 for layer in self.model.layers if isinstance(layer, QLayer))
        
        # Create the CSV file for parameters and write the header
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Epoch']
            for i in range(self.num_qlayers):
                if self.activation == 'kerr':
                    header.extend([f'Layer{i}_theta_1', f'Layer{i}_r', f'Layer{i}_theta_2', 
                               f'Layer{i}_bx', f'Layer{i}_bp', f'Layer{i}_kappa'])
                else:
                    header.extend([f'Layer{i}_theta_1', f'Layer{i}_r', f'Layer{i}_theta_2', 
                               f'Layer{i}_bx', f'Layer{i}_bp', f'Layer{i}_gamma'])
            writer.writerow(header)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        params = []
        for layer in self.model.layers:
            if isinstance(layer, QLayer):
                if self.activation == 'kerr':
                    layer_params = [
                        layer.theta_1.numpy()[0],
                        layer.r.numpy()[0],
                        layer.theta_2.numpy()[0],
                        layer.bx.numpy()[0],
                        layer.bp.numpy()[0],
                        layer.kappa.numpy()[0]
                    ]
                else:
                    layer_params = [
                        layer.theta_1.numpy()[0],
                        layer.r.numpy()[0],
                        layer.theta_2.numpy()[0],
                        layer.bx.numpy()[0],
                        layer.bp.numpy()[0],
                        layer.gamma.numpy()[0]
                    ]    
                params.extend(layer_params)
        
        # Append the parameters to the CSV file
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.epoch] + params)


# TensorFlow Custom Layer for Classical Phase Space Transformations
class CPLayer(tf.keras.layers.Layer):
    def __init__(self, stddev=0.05, activation='kerrlike'):
        super(CPLayer, self).__init__()
        self.stddev = stddev
        self.activation = activation
        
        if self.activation not in ['kerrlike', 'cubicphase']:
            raise ValueError("Activation must be either 'kerrlike' or 'cubicphase'")
        
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.stddev, seed=42)
        
        # Initialize the weights for the rotations, squeezing, translations, and nonlinear activation
        self.theta1 = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.theta2 = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.r = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.bx = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.by = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        
        # Initialize kappa or gamma based on the activation type
        if self.activation == 'kerrlike':
            self.kappa = self.add_weight(shape=(1,), initializer=initializer, trainable=True, name='kappa')
        else:  # cubicphase
            self.gamma = self.add_weight(shape=(1,), initializer=initializer, trainable=True, name='gamma')

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
        x_translated = x_rot2 + self.bx
        p_translated = p_rot2 + self.by
        
        # Apply the nonlinear activation based on the activation type
        if self.activation == 'kerrlike':
            radius = tf.sqrt(x_translated**2 + p_translated**2)
            x_activated = x_translated * tf.cos(self.kappa * radius) - p_translated * tf.sin(self.kappa * radius)
            p_activated = x_translated * tf.sin(self.kappa * radius) + p_translated * tf.cos(self.kappa * radius)
        else:  # cubicphase
            x_activated = x_translated
            p_activated = p_translated + self.gamma * x_translated**2
        
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


from sklearn.model_selection import KFold

# Function for training quantum models
def train_model(input_data, target_data, function_index, k_folds=5, learning_rate=0.01, std=0.05, cutoff_dim=10, num_layers=2, epochs=200, non_gaussian='kerr', rec=True):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f'Training model for Function {function_index} with {num_layers} layers for {epochs} epochs...')
    
    fold_histories = []
    models = []

    for fold, (train_index, val_index) in enumerate(kf.split(input_data), 1):
        print(f'Training on fold {fold}...')
        
        x_train_fold, x_val_fold = input_data[train_index], input_data[val_index]
        y_train_fold, y_val_fold = target_data[train_index], target_data[val_index]

        model = create_model(cutoff_dim, num_layers, non_gaussian, std)
        opt = tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0)
        model.compile(optimizer=opt, loss='mse', metrics=[R2ScoreWrapper()])
        
        callbacks = [TrainingProgress()]
        if rec:
            callbacks.append(ParameterLoggingCallback(fold, function_index, non_gaussian))
        
        history = model.fit(x_train_fold, y_train_fold, validation_data=(x_val_fold, y_val_fold), 
                            epochs=epochs, verbose=0, callbacks=callbacks)
        
        fold_histories.append(history.history)
        models.append(model)
        
        print(f'Fold {fold} complete.')

    # Calculate average cross-validated histories
    avg_history = {key: np.mean([h[key] for h in fold_histories], axis=0) for key in fold_histories[0].keys()}
    
    # Find the best model based on final validation loss
    best_model_index = np.argmin([h['val_loss'][-1] for h in fold_histories])
    best_model = models[best_model_index]

    print('Cross-validation complete.')
    print(f'Best model from fold {best_model_index + 1}')
    best_model.summary()

    return avg_history, best_model

# Helper function to initialise quantum models
def create_model(cutoff_dim, num_layers, non_gaussian, std):
    vacuum_state = get_vacuum_state_tf(cutoff_dim)
    model = tf.keras.Sequential([QEncoder(dim=cutoff_dim, vacuum_state=vacuum_state, name='QuantumEncoding')])
    for i in range(num_layers):
        model.add(QLayer(dim=cutoff_dim, activation=non_gaussian, stddev=std, tol=1e-3, name=f'QuantumLayer_{i+1}'))
    model.add(QDecoder(dim=cutoff_dim, name='QuantumDecoding'))
    return model
        

# Function for training classical models
def train_classical_model(input_data, target_data, function_index, k_folds=5, learning_rate=0.01, std=0.05, non_linearity='kerrlike', num_layers=2, epochs=200):
    
    input_data = np.hstack((input_data, np.zeros(input_data.shape)))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f'Training classical model for Function {function_index} with {num_layers} layers for {epochs} epochs...')
    
    fold_histories = []
    models = []

    for fold, (train_index, val_index) in enumerate(kf.split(input_data), 1):
        print(f'Training on fold {fold}...')
        
        x_train_fold, x_val_fold = input_data[train_index], input_data[val_index]
        y_train_fold, y_val_fold = target_data[train_index], target_data[val_index]

        # Create a new model for each fold
        layers = [CPLayer(stddev=std, activation=non_linearity) for _ in range(num_layers)] + [ExtractXLayer()]
        model = tf.keras.Sequential(layers)

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0)
        model.compile(optimizer=opt, loss='mse', metrics=[R2ScoreWrapper()])
        
        # Train the model
        history = model.fit(x_train_fold, y_train_fold, validation_data=(x_val_fold, y_val_fold), 
                            epochs=epochs, verbose=0, callbacks=[TrainingProgress()])

        fold_histories.append(history.history)
        models.append(model)
        
        print(f'Fold {fold} complete.')

    # Calculate average cross-validated histories
    avg_history = {key: np.mean([h[key] for h in fold_histories], axis=0) for key in fold_histories[0].keys()}
    
    # Find the best model based on final validation loss
    best_model_index = np.argmin([h['val_loss'][-1] for h in fold_histories])
    best_model = models[best_model_index]

    print('Cross-validation complete.')
    print(f'Best model from fold {best_model_index + 1}')
    best_model.summary()

    return avg_history, best_model