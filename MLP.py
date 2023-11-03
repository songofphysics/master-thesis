import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

tf.get_logger().setLevel('ERROR')

class CustomLayer1D(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomLayer1D, self).__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed = 42)
        self.theta1 = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.theta2 = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.r = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.kappa = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        self.b = self.add_weight(shape=(1,), initializer=initializer, trainable=True)
        
    def call(self, inputs):
        # Assuming inputs is a 1D point x0
        
        # 1. Initial rotation by theta1: x1 = x0 * cos(theta1)
        x1 = tf.multiply(inputs, tf.cos(self.theta1))
        
        # 2. Scaling by e^-r: x2 = e^-r * x1
        x2 = tf.multiply(tf.exp(-self.r), x1)
        
        # 3. Second rotation by theta2: x3 = x2 * cos(theta2)
        x3 = tf.multiply(x2, tf.cos(self.theta2))
        
        # Adding the bias term
        x3_biased = x3 + self.b
        
        # 4. Calculate the radius (distance from origin) at this point
        radius = tf.sqrt(tf.square(x3_biased))
        
        # 5. Calculate angular velocity, which is proportional to the radius
        angular_velocity = self.kappa * radius
        
        # 6. Rotate by the angular velocity
        activation = x3_biased * tf.cos(angular_velocity)
        
        return activation
    
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

# Function for training models with different configurations
def train_classical_models(configs, x_data, y_data_noisy):
    trained_models = []
    histories = []
    for num_layers, epochs in configs:
        # Create a new model for each configuration
        layers = [CustomLayer1D() for _ in range(num_layers)]
        model = tf.keras.Sequential(layers)

        # Compile and train the model
        model.compile(optimizer='adam', loss='mse')
        print(f'Training model with {num_layers} layers for {epochs} epochs...')
        progress_bar = TQDMProgressBar()
        history = model.fit(x_data, y_data_noisy, validation_split=0.25, epochs=epochs, verbose=0, callbacks=[progress_bar])

        # Store the trained model and its history
        histories.append(history)
        trained_models.append(model)

    return trained_models, histories

# Function for plotting results and loss histories
def plot_classical_results(models, histories, configs, x_data, y_data, y_data_noisy):
    # Set up the figure for the fits
    fig, axes = plt.subplots(2, len(models)//2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot the fits
    for i, model in enumerate(models):
        y_pred = model.predict(x_data)
        axes[i].scatter(x_data, y_data_noisy, s=5, label='Noisy Data')
        axes[i].plot(x_data, y_pred, label='Fitted Curve', color='r')
        axes[i].plot(x_data, y_data, label='True Curve', color='g')
        axes[i].legend()
        axes[i].set_title(f'Fitted Sine Curve - Layers: {configs[i][0]}, Epochs: {configs[i][1]}')
    
    plt.tight_layout()
    plt.savefig('fits_classical.png')
    
    # Set up the figure for the loss histories
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Evaluate models and collect losses
    losses = [history.history['loss'] for history in histories]

    # Plot loss for varying epochs with num_layers = 6
    for i, loss in enumerate(losses[:len(losses)//2]):
        axes[0].plot(loss, label=f'Epochs: {configs[i][1]}')
    axes[0].set_title(f'Loss for Varying Epochs (Layers = {configs[0][0]})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    
    # Plot loss for varying num_layers for epochs = 500
    for i, loss in enumerate(losses[len(losses)//2:]):
        axes[1].plot(loss, label=f'Layers: {configs[i+len(losses)//2][0]}')
    axes[1].set_title(f'Loss for Varying Layers (Epochs = {configs[-1][1]})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('mses_classical.png')

    # Plot loss and validation loss for the best model
    config_index = configs.index((8, 500))
    history = histories[config_index]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(8, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f'Loss and Validation Loss vs Epochs for {configs[config_index][0]} layers')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mse_with_val_classical.png')

# Generate noisy sine curve data
np.random.seed(0)
n_points = 100
x_data = np.linspace(0, 2*np.pi, n_points)
y_data = np.sin(x_data) 
y_data_noisy = y_data + np.random.normal(0, 0.1, n_points)

# Reshape data to fit the model's input shape
x_data_reshaped = x_data.reshape(-1, 1)
y_data_reshaped = y_data_noisy.reshape(-1, 1)

# Evaluate the trained model
configs = [
    (6, 100),
    (6, 250),
    (6, 500),
    (7, 500),
    (8, 500),
    (10, 500)
]

# Train the models with different configurations
trained_models, histories = train_classical_models(configs, x_data_reshaped, y_data_reshaped)

# Plot the results and loss histories
plot_classical_results(trained_models, histories, configs, x_data, y_data, y_data_noisy)