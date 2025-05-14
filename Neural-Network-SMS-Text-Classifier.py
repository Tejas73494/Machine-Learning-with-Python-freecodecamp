import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers

# Download and load the data
train_data_url = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
test_data_url = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"

!wget {train_data_url} -O train-data.tsv
!wget {test_data_url} -O valid-data.tsv

train_path = "train-data.tsv"
test_path = "valid-data.tsv"

train_df = pd.read_csv(train_path, sep='\t', header=None, names=["Label", "Message"]).dropna()
test_df = pd.read_csv(test_path, sep='\t', header=None, names=["Label", "Message"]).dropna()

# Encode labels
train_df['Label'] = pd.factorize(train_df['Label'])[0]
test_df['Label'] = pd.factorize(test_df['Label'])[0]

train_labels = train_df['Label'].values
test_labels = test_df['Label'].values

# Prepare datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_df["Message"].values, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_df["Message"].values, test_labels))

BUFFER_SIZE = 100
BATCH_SIZE = 32
VOCAB_SIZE = 1000
MAX_SEQUENCE_LENGTH = 100

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Vectorize the text data
vectorizer = TextVectorization(output_mode='int', max_tokens=VOCAB_SIZE, output_sequence_length=MAX_SEQUENCE_LENGTH)
vectorizer.adapt(train_dataset.map(lambda text, label: text))

# Build the model
model = tf.keras.Sequential([
    vectorizer,
    layers.Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=64, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    validation_steps=30,
    epochs=15,
)

# Prediction function
def predict_sms(text):
    # Convert the text to a tensor (batch of size 1)
    text_tensor = tf.convert_to_tensor([text])
    pred_prob = model.predict(text_tensor)[0][0]
    pred_label = "spam" if pred_prob >= 0.5 else "ham"
    return [pred_prob, pred_label]


# Test the prediction function
def evaluate_predictions():
    test_cases = [
        ("how are you doing today", "ham"),
        ("sale today! to stop texts call 98912460 4", "spam"),
        ("i dont want to go. can we try it a different day? available sat", "ham"),
        ("our new mobile video service is live. just install on your phone to start watching.", "spam"),
        ("you have won £1000 cash! call to claim your prize.", "spam"),
        ("i'll bring it tomorrow. don't forget the milk.", "ham"),
        ("wow, is your arm alright. that happened to me one time too", "ham")
    ]

    success = True
    for msg, expected in test_cases:
        prediction = predict_sms(msg)
        if prediction[1] != expected:
            success = False

    if success:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

evaluate_predictions()
