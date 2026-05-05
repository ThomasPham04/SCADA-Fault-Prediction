# Model Architecture: Global Deep 4-Unit Bottleneck

This experiment used a 6-layer deep architecture with an extreme compression bottleneck of 4 units, trained on the **pooled global dataset** (all assets).

- **Model Name:** `small_lstm_ae`
- **Scope:** Global (Pooled)
- **Encoder:**
  - LSTM Layer 1: 44 units
  - LSTM Layer 2: 25 units
  - LSTM Layer 3: 4 units
- **Bottleneck:** `RepeatVector` (Latent dimension = 4)
- **Decoder:**
  - LSTM Layer 1: 4 units
  - LSTM Layer 2: 25 units
  - LSTM Layer 3: 44 units
- **Output:** TimeDistributed Dense layer
