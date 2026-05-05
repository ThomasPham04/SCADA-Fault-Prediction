# Model Architecture: Lite Experiment (24 Units)

This experiment used a simplified 2-layer LSTM Autoencoder for fast iteration.

- **Model Name:** `small_lstm_ae`
- **Encoder:**
  - LSTM Layer: 24 units
- **Bottleneck:** `RepeatVector` (Latent dimension = 24)
- **Decoder:**
  - LSTM Layer: 24 units
- **Output:** TimeDistributed Dense layer
