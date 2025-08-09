import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ReduceLROnPlateau

# === 1. Load & preprocess data ===
df = pd.read_csv('eeg_raw_dataset.csv')
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])
eeg_data = df.drop(columns=['label']).values.astype(np.float32)
eeg_data = eeg_data.reshape((200, 128, 8))  # [samples, timesteps, features]

# === 2. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    eeg_data, labels, test_size=0.1, random_state=42, stratify=labels
)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)

# === 3. Build LSTM model ===
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(128, 8)),
#     tf.keras.layers.LSTM(64),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 8)),  # Use the shape of the reshaped data
    tf.keras.layers.LSTM(64, return_sequences=True), # Add return_sequences=True for stacking LSTMs or adding Conv1D before LSTM
    tf.keras.layers.Dropout(0.2), # Add dropout after LSTM
    tf.keras.layers.LSTM(32), # Add another LSTM layer
    tf.keras.layers.Dropout(0.2), # Add dropout after LSTM
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Add dropout after Dense layer
    tf.keras.layers.Dense(2, activation='softmax') # Output layer with 2 units for binary classification
])


# Define the learning rate reduction callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor the validation loss
    factor=0.5,          # Reduce learning rate by half
    patience=5,          # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=0.001       # Lower bound on the learning rate
)

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 4. Train model ===
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=16,
    callbacks=[reduce_lr] # Add the callback here
)

# === 5. Visualize accuracy & loss ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()