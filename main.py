# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. Load dataset
df = pd.read_csv("Tamil_All_Names.csv")

# Ensure column names are correct
df.columns = df.columns.str.strip()  # remove spaces
print("Columns:", df.columns)

# Expect dataset to have "Name" and "Gender"
df = df.dropna(subset=["Name", "Gender"])  # remove missing rows
names = df["Name"].astype(str).values
labels = df["Gender"].astype(str).values

# 2. Encode labels (Male/Female -> 0/1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# 3. Tokenize names
tokenizer = Tokenizer(char_level=True)  # use characters for Tamil names
tokenizer.fit_on_texts(names)
X = tokenizer.texts_to_sequences(names)

# Pad sequences
max_len = max(len(x) for x in X)
X = pad_sequences(X, maxlen=max_len, padding="post")

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 6. Train model
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32, callbacks=[early_stop])

# 7. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# 8. Save model
model.save("gender_name_model.keras")

# Save tokenizer and label encoder
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model, tokenizer, and label encoder saved successfully!")
