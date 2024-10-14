import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define the Generator
def build_generator(latent_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, input_dim=latent_dim),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(512),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(784, activation='tanh')
    ])
    return model

# Define the Discriminator
def build_discriminator(img_shape):
    model = keras.Sequential([
        keras.layers.Dense(512, input_dim=np.prod(img_shape)),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Define the GAN
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            disc_loss = self.loss_fn(tf.ones_like(real_output), real_output) + \
                        self.loss_fn(tf.zeros_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"d_loss": disc_loss, "g_loss": gen_loss}

# Hyperparameters
latent_dim = 100
img_shape = (28, 28, 1)
epochs = 50000
batch_size = 64

# Build and compile the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Build and compile the generator
generator = build_generator(latent_dim)

# Build and compile the GAN
gan = GAN(discriminator, generator, latent_dim)
gan.compile(d_optimizer=keras.optimizers.Adam(0.0002, 0.5),
            g_optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss_fn=keras.losses.BinaryCrossentropy())

# Load and preprocess the data
(X_train, _), (_, _) = keras.datasets.mnist.load_data()
X_train = X_train / 127.5 - 1.0
X_train = np.expand_dims(X_train, axis=3)

# Train the GAN
gan.fit(X_train, epochs=epochs, batch_size=batch_size)

# After training, you can use the generator to reconstruct images
noise = tf.random.normal(shape=(1, latent_dim))
generated_image = generator(noise, training=False)
