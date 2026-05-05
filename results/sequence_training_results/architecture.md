# Model Architecture: Standard LSTM-AE

This experiment used the default high-complexity LSTM Autoencoder.

- **Model Name:** `lstm_ae`
- **Encoder:**
  - LSTM Layer 1: 96 units
  - Dropout: 0.2
  - LSTM Layer 2: 48 units
- **Bottleneck:** `RepeatVector` (Latent dimension = 48)
- **Decoder:**
  - LSTM Layer 1: 48 units
  - Dropout: 0.2
  - LSTM Layer 2: 96 units
- **Output:** TimeDistributed Dense layer
