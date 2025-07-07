import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import StandardScaler , LabelEncoder

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import SelectKBest
from tensorflow.python.keras.utils.version_utils import callbacks

df = pd.read_csv("data/breast cancer.csv")

# print(df.head())
# print(df.info())
scaler = StandardScaler()

le = LabelEncoder()

y = df["diagnosis"]
X = df.drop(columns=["diagnosis"] , axis=1)

y = le.fit_transform(y)

X = scaler.fit_transform(X)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

model = Sequential()
model.add(Dense(64 , activation="relu" , input_dim=31))
model.add(Dense(32 , activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1 , activation="sigmoid"))

model.compile(loss="binary_crossentropy" , optimizer="Adam" , metrics=["accuracy"])

callback = EarlyStopping(
    monitor="val_loss",
    verbose = 1,
    patience=5,
    restore_best_weights="True"
)

model.fit(
    X_train , y_train,
    validation_data=(X_test,y_test),
    epochs = 100,
    batch_size=32,
    callbacks=[callback],
    verbose = 1

)

loss, acc = model.evaluate(X_test, y_test)
print(f"Validation Accuracy: {acc:.4f}")