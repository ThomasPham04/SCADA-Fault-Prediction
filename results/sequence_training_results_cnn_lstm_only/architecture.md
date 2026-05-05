# Model Architecture: CNN-LSTM Classifier

This experiment used a hybrid architecture combining Convolutional layers for spatial feature extraction and LSTM for temporal dependencies.

- **Model Name:** `cnn_lstm`
- **Layers:**
  - Conv1D: 64 filters (size 5, relu)
  - MaxPooling1D (pool 2)
  - Dropout
  - Conv1D: 64 filters (size 3, relu)
  - MaxPooling1D (pool 2)
  - LSTM: 64 units
  - Dropout
  - Dense: 32 units (relu)
  - Output: Dense (1, sigmoid)
