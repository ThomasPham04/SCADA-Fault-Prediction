"""
AutoDecoderModel — models.architectures.autodecoder
A skeleton for the AutoDecoder architecture for fault detection.
"""

import os
import sys

# Optional: suppress TensorFlow logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Ensure base directory is in sys.path if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from config import ...

class AutoEncoder:
    """
    Args:
        input_shape: Tuple (window_size, n_features).
        latent_dim:  Dimensionality of the latent representation.
    """

    def __init__(
        self,
        input_shape: int,
        latent_dim: int = 4,
    ) -> None:
        self.input_shape = input_shape
        self.latent_dim = latent_dim

    def build(self) -> Model:
        """
        Construct and compile the Model.
        
        Returns:
            Compiled Keras Model.
        """
        inputs = Input(shape=(self.input_shape,), name="Input_Layer")
        
        #Encoder
        x = Dense(44, activation="relu", name="Encoder_Layer_1")(inputs)
        x = Dense(25, activation="relu", name="Encoder_Layer_2")(x)
        encoded = Dense(self.latent_dim, activation="relu", name="Encoder_Layer_3")(x)

        #Decoder 
        decoder_layer_1 = Dense(25, activation="relu", name="Decoder_Layer_1")
        decoder_layer_2 = Dense(44, activation="relu", name="Decoder_Layer_2")
        output_layer = Dense(self.input_shape, activation="sigmoid", name="Output_Layer")

        x = decoder_layer_1(encoded)
        x = decoder_layer_2(x)
        decoded = output_layer(x)
        
        autoencoder = Model(inputs, decoded, name="Autoencoder")
        encoder = Model(inputs, encoded, name="Encoder") 
        
        # Build the standalone Decoder model
        encoded_input = Input(shape=(self.latent_dim,), name="Decoder_Input")
        dec_x = decoder_layer_1(encoded_input)
        dec_x = decoder_layer_2(dec_x)
        dec_out = output_layer(dec_x)
        decoder = Model(encoded_input, dec_out, name="Decoder")
        
        # Use Mean Squared Error loss for autoencoders
        autoencoder.compile(optimizer = 'adam', loss = 'mse', metrics=['mae'])
        autoencoder.summary()
        
        return autoencoder, encoder, decoder

# ---------------------------------------------------------------------------
# Backward-compatible module-level alias (optional)
# ---------------------------------------------------------------------------

def build_autodecoder_model(input_shape: int, latent_dim: int = 16) -> tuple:
    """Alias — wraps AutoDecoderModel(...).build()."""
    return AutoEncoder(input_shape=input_shape, latent_dim=latent_dim).build()
