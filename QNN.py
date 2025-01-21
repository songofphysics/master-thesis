import tensorrt
import numpy as np
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

tf.config.run_functions_eagerly(True)

# Utility functions
@tf.function
def get_vacuum_state_tf(dim):
    vacuum_state = tf.zeros([dim, 1], dtype=tf.complex64)
    one = tf.constant([[1.0 + 0.0j]], dtype=tf.complex64)
    vacuum_state = tf.concat([one, vacuum_state[1:]], axis=0)
    return vacuum_state

@tf.function
def annihilation(dim):
    diag_vals = tf.sqrt(tf.cast(tf.range(1, dim), dtype=tf.complex64))
    return tf.linalg.diag(diag_vals, k=1)

@tf.function
def number(dim):
    return tf.linalg.diag(tf.cast(tf.range(dim), dtype=tf.complex64))

@tf.function
def displacement_operator(dim, x, y=0):
    alpha = tf.complex(x, y)
    a = annihilation(dim)
    a_dag = tf.linalg.adjoint(a)
    exponent = alpha * a_dag - tf.math.conj(alpha) * a
    return tf.linalg.expm(exponent)

@tf.function
def displacement_encoding(dim, alpha_vec):
    alpha_vec = tf.cast(alpha_vec, dtype=tf.complex64)
    num = tf.shape(alpha_vec)[0]
    a = annihilation(dim)
    a_dag = tf.linalg.adjoint(a)
    
    alpha_vec = tf.reshape(alpha_vec, [-1, 1, 1])
    exponent = alpha_vec * tf.expand_dims(a_dag, 0) - tf.math.conj(alpha_vec) * tf.expand_dims(a, 0)
    return tf.linalg.expm(exponent)

@tf.function
def rotation_operator(dim, theta):
    theta = tf.cast(theta, dtype=tf.complex64)
    n = number(dim)
    return tf.linalg.expm(-1j * theta * n)

@tf.function
def squeezing_operator(dim, r):
    r = tf.cast(r, dtype=tf.complex64)
    a = annihilation(dim)
    a_dag = tf.linalg.adjoint(a)
    exponent = 0.5 * (tf.math.conj(r) * (a @ a) - r * (a_dag @ a_dag))
    return tf.linalg.expm(exponent)

@tf.function
def kerr_operator(dim, kappa):
    kappa = tf.cast(kappa, dtype=tf.complex64)
    n = number(dim)
    return tf.linalg.expm(1j * kappa * (n @ n))

@tf.function
def cubic_phase_operator(dim, gamma):
    gamma = tf.cast(gamma, dtype=tf.complex64)
    a = annihilation(dim)
    x = (a + tf.linalg.adjoint(a)) / tf.cast(2.0, dtype=tf.complex64)
    return tf.linalg.expm(1j * (gamma/3) * (x @ x @ x))


# TensorFlow Custom Layer for Quantum Encoding
class QEncoder(tf.keras.layers.Layer):
    def __init__(self, dim, vacuum_state, r= 100.0, **kwargs):
        super(QEncoder, self).__init__(**kwargs)
        self.dim = dim
        self.r = r
        self.vacuum_state = tf.cast(vacuum_state, dtype=tf.complex64)

    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        squeezed_vacuum_state = tf.matmul(squeezing_operator(self.dim, self.r), self.vacuum_state)
        batch_squeezed_state = tf.tile(tf.expand_dims(squeezed_vacuum_state, axis=0), [batch_size, 1, 1])
        batch_displacement_operators = displacement_encoding(self.dim, inputs/2)
        displaced_states = tf.einsum('bij,bjk->bik', batch_displacement_operators, batch_squeezed_state)
        
        # Normalize the states
        norm = tf.sqrt(tf.reduce_sum(tf.abs(displaced_states)**2, axis=1, keepdims=True))
        norm = tf.cast(norm, dtype=tf.complex64)
        normalized_states = displaced_states / norm
        
        # Convert to density matrices
        density_matrices = tf.einsum('bij,bkj->bik', normalized_states, tf.math.conj(normalized_states))
        
        return density_matrices


# TensorFlow Custom Layer for Quantum Transformations
class QLayer(tf.keras.layers.Layer):
    def __init__(self, dim, stddev=0.05, activation='kerr', **kwargs):
        super(QLayer, self).__init__(**kwargs)
        self.dim = dim
        self.stddev = stddev
        self.activation = activation.lower()
        if self.activation not in ['kerr', 'cubicphase']:
            raise ValueError("Activation must be either 'kerr' or 'cubicphase'")

    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.stddev, seed=42)
        self.theta_1 = self.add_weight(name="theta_1", shape=[1,], initializer=initializer, trainable=True)
        self.theta_2 = self.add_weight(name="theta_2", shape=[1,], initializer=initializer, trainable=True)
        self.r = self.add_weight(name="r", shape=[1,], initializer=initializer, trainable=True)
        self.bx = self.add_weight(name="bx", shape=[1,], initializer=initializer, trainable=True)
        self.bp = self.add_weight(name="bp", shape=[1,], initializer=initializer, trainable=True)
        
        if self.activation == 'kerr':
            self.kappa = self.add_weight(name="kappa", shape=[1,], initializer=initializer, trainable=True)
        else:  # cubicphase
            self.gamma = self.add_weight(name="gamma", shape=[1,], initializer=initializer, trainable=True)

    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Create operators
        R_1 = rotation_operator(self.dim, self.theta_1)
        S = squeezing_operator(self.dim, self.r)
        R_2 = rotation_operator(self.dim, self.theta_2)
        D = displacement_operator(self.dim, self.bx, self.bp)
        
        if self.activation == 'kerr':
            A = kerr_operator(self.dim, self.kappa)
        else:  # cubicphase
            A = cubic_phase_operator(self.dim, self.gamma)

        # Combine all operators into a single unitary
        U = A@D@R_2@S@R_1
        
        # Expand U to match batch size
        U_batch = tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])
        
        # Apply the combined operation
        return tf.einsum('bij,bjk,blk->bil', U_batch, inputs, tf.math.conj(U_batch))
        

# Tensorflow Custom Layer for Loss Channel
class LossChannel(tf.keras.layers.Layer):    
    def __init__(self, dim, T=1.0, **kwargs):
        super(LossChannel, self).__init__(**kwargs)
        self.dim = dim
        self.T = T
        
    def build(self, input_shape):
        # Get annihilation operator from external function
        self.a = tf.cast(annihilation(self.dim), dtype=tf.complex64)
        
        # Channel parameters
        self.factor = tf.cast((1 - self.T) / self.T, dtype=tf.complex64)
        self.sqrt_T = tf.cast(tf.sqrt(self.T), dtype=tf.complex64)
        
        # Number operator and its transformation
        self.a_dag_a = tf.matmul(self.a, self.a, adjoint_a=True)
        self.sqrt_T_pow_a_dag_a = tf.linalg.expm(tf.math.log(self.sqrt_T) * self.a_dag_a)
        
        # Prepare Kraus operator components vectorized
        n_range = tf.cast(tf.range(self.dim), dtype=tf.complex64)
        factorial_term = tf.exp(tf.math.lgamma(tf.cast(n_range + 1, dtype=tf.float32)))
        power_term = tf.exp(n_range / 2 * tf.math.log(self.factor))
        self.E_n_diag = power_term / tf.sqrt(tf.cast(factorial_term, dtype=tf.complex64))
        
        # Compute powers of annihilation operator through matrix exponential
        n_expanded = tf.reshape(n_range, (self.dim, 1, 1))
        log_a = tf.where(
            tf.abs(self.a) > 0,
            tf.math.log(tf.cast(self.a, tf.complex64)),
            tf.zeros_like(self.a, dtype=tf.complex64)
        )
        self.a_powers = tf.linalg.expm(n_expanded * log_a)
        
        super(LossChannel, self).build(input_shape)
    
    @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.complex64)
        
        # Construct Kraus operators
        # E_n shape: [dim, dim, dim] where first dim indexes the n different operators
        E_n = tf.einsum('n,nij->nij', self.E_n_diag, self.a_powers)
        E_n = tf.matmul(E_n, self.sqrt_T_pow_a_dag_a)
        
        # First compute E_n E_n^† for each n
        # shape will be [dim, dim, dim]
        E_n_E_n_dag = tf.einsum('nij,nkj->nik', E_n, tf.math.conj(E_n))
        
        # Now apply the sum of these operators to the state
        # inputs shape: [batch_size, dim, dim]
        # output shape: [batch_size, dim, dim]
        output = tf.einsum('nik,bkj->bij', E_n_E_n_dag, inputs)
        
        return output


# TensorFlow Custom Layer for Quantum Decoding
class QDecoder(tf.keras.layers.Layer):
    def __init__(self, dim, sampling=False, num_shots=10000, **kwargs):
        super(QDecoder, self).__init__(**kwargs)
        self.dim = dim
        self.sampling = sampling
        self.num_shots = num_shots
        
    def build(self, input_shape):
        x_operator = self.build_x_operator()
        self.x_quad = self.add_weight(
            name='x_quad',
            shape=x_operator.shape,
            dtype=tf.complex64,
            initializer=lambda shape, dtype: x_operator,
            trainable=False
        )
        super().build(input_shape)
        
    def build_x_operator(self):
        a = annihilation(self.dim)
        x_operator = (a + tf.linalg.adjoint(a)) / 2.0
        return x_operator
        
    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        means = tf.math.real(tf.einsum('njk,jk->n', inputs, self.x_quad))

        if self.sampling:
            squares = tf.math.real(tf.einsum('njk,jk->n', 
                        inputs, self.x_quad@self.x_quad))
            variances = squares - tf.square(means)
            outcomes = tf.random.normal([batch_size], seed=42)
            outcomes = outcomes*tf.sqrt(variances/self.num_shots) + means
            
            return tf.expand_dims(outcomes, axis=-1)
        else:
            return tf.expand_dims(means, axis=-1)
    
    
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
        lr = self.model.optimizer.learning_rate
        
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
    def __init__(self, function_index, activation):
        super(ParameterLoggingCallback, self).__init__()
        self.function_index = function_index
        self.activation = activation
        self.filename = f'Params_{function_index}.csv'
        self.epoch = 0
        
    def on_train_begin(self, logs=None):
        # Count the number of compatible layers
        self.num_layers = sum(1 for layer in self.model.layers 
                            if isinstance(layer, (QLayer, CPLayer)))
        
        # Create the CSV file for parameters and write the header
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Epoch']
            for i in range(self.num_layers):
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
            if isinstance(layer, (QLayer, CPLayer)):
                if self.activation == 'kerr' or self.activation == 'kerrlike':
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
        self.theta_1 = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.theta_2 = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.r = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.bx = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.bp = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        
        # Initialize kappa or gamma based on the activation type
        if self.activation == 'kerrlike':
            self.kappa = self.add_weight(shape=(1,), initializer=initializer, trainable=True, name='kappa')
        else:  # cubicphase
            self.gamma = self.add_weight(shape=(1,), initializer=initializer, trainable=True, name='gamma')

    def call(self, inputs):
        # Unpack x and p components
        x, p = inputs[..., 0], inputs[..., 1]
        
        # Apply the first rotation
        x_rot = x * tf.cos(self.theta_1) - p * tf.sin(self.theta_1)
        p_rot = x * tf.sin(self.theta_1) + p * tf.cos(self.theta_1)
        
        # Apply squeezing
        x_squeezed = tf.exp(-self.r) * x_rot
        p_squeezed = tf.exp(self.r) * p_rot
        
        # Apply the second rotation
        x_rot2 = x_squeezed * tf.cos(self.theta_2) - p_squeezed * tf.sin(self.theta_2)
        p_rot2 = x_squeezed * tf.sin(self.theta_2) + p_squeezed * tf.cos(self.theta_2)
        
        # Apply translation
        x_translated = x_rot2 + self.bx
        p_translated = p_rot2 + self.bp
        
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
    
    
# TensorFlow Custom Layers for Extracting Positions
class ExtractXLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ExtractXLayer, self).__init__()

    def call(self, inputs):
        # Assuming the 'x' values are the first component in the (N, 2) input
        # Extracts and returns the 'x' component in shape (N, 1)
        return tf.expand_dims(inputs[..., 0], axis=-1)
    

# TensorFlow Custom Layer for Sampling Phase Space Points
class WignerSamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_samples=5000):
        super(WignerSamplingLayer, self).__init__()
        self.num_samples = int(num_samples)
        wigner_data = np.loadtxt('Julia_Sampler/wigner_samples.csv', 
                                delimiter=',', skiprows=1)[:, 1:3]
        self.wigner_samples = tf.convert_to_tensor(wigner_data, dtype=tf.float32)
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_samples": self.num_samples})
        return config
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        samples_needed = batch_size * self.num_samples
        return self.wigner_samples[:samples_needed]
    

# TensorFlow Custom Layers for Extracting Positions
class SampledXLayer(tf.keras.layers.Layer):
    def __init__(self, num_samples=5000):
        super(SampledXLayer, self).__init__()
        self.num_samples = int(num_samples)

    def call(self, inputs):
        x_values = inputs[..., 0]
        batch_size = tf.shape(x_values)[0]
        num_groups = batch_size // self.num_samples
        reshaped = tf.reshape(x_values, [num_groups, self.num_samples])
        averaged = tf.reduce_mean(reshaped, axis=1)
        return tf.expand_dims(averaged, axis=-1)


# Function for training quantum models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def train_model(input_data, target_data, function_index, 
                learning_rate=0.01, std=0.05, eta = 0.1,
                cutoff_dim=10, num_layers=6, epochs=100, r=2.0, 
                non_gaussian='kerr', rec=False, sample=False):
    
    x_train, x_test, y_train, y_test = train_test_split(
    input_data, target_data, test_size=0.2, random_state=42)
    
    print(f'Training model for Function {function_index} with {num_layers} layers for {epochs} epochs...')

    model = create_model(cutoff_dim, num_layers, non_gaussian, std, r=r, T=1-eta, sampling=sample)
    opt = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse', metrics=[R2ScoreWrapper()])
    
    callbacks = [TrainingProgress()]
    if rec:
        callbacks.append(ParameterLoggingCallback(function_index, non_gaussian))
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                        epochs=epochs, verbose=0, callbacks=callbacks)

    return history, model

def create_model(cutoff_dim, num_layers, non_gaussian, std, sampling=False, r=2.0, T = 0.9):
    vacuum_state = get_vacuum_state_tf(cutoff_dim)
    model = tf.keras.Sequential([QEncoder(dim=cutoff_dim, vacuum_state=vacuum_state, r=r, name='QuantumEncoding')])
    for i in range(num_layers):
        model.add(QLayer(dim=cutoff_dim, activation=non_gaussian, stddev=std, name=f'QuantumLayer_{i+1}'))
        if T != 1.0:
            model.add(LossChannel(dim=cutoff_dim, T=0.75, name=f'LossChannel_{i+1}'))
    model.add(QDecoder(dim=cutoff_dim, sampling=sampling, name='QuantumDecoding'))
    return model
        

# Function for training classical models
def train_classical_model(input_data, target_data, function_index, epochs=100,
                          learning_rate=0.01, std=0.05, num_layers=6,
                          non_linearity='kerrlike', rec=False):
    
    input_data = np.hstack((input_data, np.zeros(input_data.shape)))

    x_train, x_test, y_train, y_test = train_test_split(
    input_data, target_data, test_size=0.2, random_state=42)
    
    print(f'Training classical model for Function {function_index} with {num_layers} layers for {epochs} epochs...')

    # Create a new model for each fold
    layers = [CPLayer(stddev=std, activation=non_linearity) for _ in range(num_layers)] + [ExtractXLayer()]
    model = tf.keras.Sequential(layers)

    # Compile the model
    opt = Adam(learning_rate, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse', metrics=[R2ScoreWrapper()])
    
    callbacks = [TrainingProgress()]
    if rec:
        callbacks.append(ParameterLoggingCallback(function_index, non_linearity))

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                        epochs=epochs, verbose=0, callbacks=callbacks)

    return history, model


# Function for training classical models
def train_wigner_samples(input_data, target_data, function_index, epochs=100,
                        learning_rate=0.01, std=0.05, num_layers=6,
                        non_linearity='kerrlike', rec=False):
    
    num_samples = 5000
    
    layers = [WignerSamplingLayer(num_samples=num_samples)]
    layers.extend([CPLayer(stddev=std, activation=non_linearity) for _ in range(num_layers)])
    layers.append(SampledXLayer(num_samples=num_samples))
    
    model = tf.keras.Sequential(layers)
    opt = Adam(learning_rate, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse', metrics=[R2ScoreWrapper()])
    
    callbacks = [TrainingProgress()]
    if rec:
        callbacks.append(ParameterLoggingCallback(function_index, non_linearity))

    history = model.fit(input_data, target_data, validation_split=0.2,
                       epochs=epochs, verbose=0, callbacks=callbacks)
    
    return history, model


def cost(target_state, predicted_state):
    target_state = tf.squeeze(target_state)
    predicted_state = tf.squeeze(predicted_state)
    return tf.abs(tf.reduce_sum(tf.math.conj(predicted_state) * target_state) - 1)

def metric(target_state, predicted_state):
    target_state = tf.squeeze(target_state)
    predicted_state = tf.squeeze(predicted_state)
    return tf.abs(tf.reduce_sum(tf.math.conj(predicted_state)*target_state))**2


def learn_state(target_state, learning_rate=0.05, std=0.1, num_layers=10, epochs=1000, non_gaussian='kerr', rec=True):
    cutoff_dim = np.shape(target_state)[0]
    target_state = tf.reshape(target_state, [1,cutoff_dim,1])

    model = tf.keras.Sequential(QLayer(dim=cutoff_dim, activation=non_gaussian, stddev=std, tol=1e-3, name=f'QuantumLayer_{1}'))
    for i in range(2,num_layers+1):
        model.add(QLayer(dim=cutoff_dim, activation=non_gaussian, stddev=std, tol=1e-3, name=f'QuantumLayer_{i}'))

    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=opt, loss=cost, metrics=[metric])

    initial_state = tf.reshape(get_vacuum_state_tf(cutoff_dim), [1,cutoff_dim,1])
    history = model.fit(initial_state, target_state, epochs=epochs, verbose=1)

    return history, model