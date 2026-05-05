# Model Architecture: 32 -> 8 Bottleneck

This experiment used a 4-layer architecture with a significant compression bottleneck of 8 units.

- **Model Name:** `small_lstm_ae`
- **Encoder:**
  - LSTM Layer 1: 32 units
  - LSTM Layer 2: 8 units
- **Bottleneck:** `RepeatVector` (Latent dimension = 8)
- **Decoder:**
  - LSTM Layer 1: 8 units
  - LSTM Layer 2: 32 units
- **Output:** TimeDistributed Dense layer
