import os
import pickle
import numpy as np
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense

MODEL_PATH = "iris_rnn.keras"
SCALER_PATH = "scaler.pkl"

CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]


def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # One timestep, four features (One-to-One style demonstration)
    X = X.reshape((X.shape[0], 1, 4))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Sequential([
        SimpleRNN(32, input_shape=(1, 4)),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    with st.spinner("Training Model..."):
        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=8,
            validation_split=0.1,
            verbose=1
        )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\nClassification Report")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, preds))

    model.save(MODEL_PATH)

    st.success(f"Model trained successfully!\nAccuracy: {accuracy*100:.2f}%")


def predict_flower(sl, sw, pl, pw):
    model = load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    sample = np.array([[sl, sw, pl, pw]])
    sample = scaler.transform(sample)
    sample = sample.reshape((1, 1, 4))

    pred = model.predict(sample, verbose=0)[0]

    idx = np.argmax(pred)

    return CLASS_NAMES[idx], pred[idx], pred


def prediction_page():
    st.subheader("Predict Iris Species")

    c1, c2 = st.columns(2)

    with c1:
        sl = st.number_input("Sepal Length", value=5.1)
        pl = st.number_input("Petal Length", value=1.4)

    with c2:
        sw = st.number_input("Sepal Width", value=3.5)
        pw = st.number_input("Petal Width", value=0.2)

    if st.button("Predict"):

        flower, confidence, probs = predict_flower(
            sl, sw, pl, pw
        )

        st.success(f"Predicted Species: **{flower}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

        st.write("### Probabilities")

        for i, name in enumerate(CLASS_NAMES):
            st.write(f"{name}: {probs[i]*100:.2f}%")


def main():

    st.set_page_config(
        page_title="Iris One-to-One RNN",
        page_icon="🌸"
    )

    st.title("🌸 Iris Classification using One-to-One SimpleRNN")

    menu = st.sidebar.selectbox(
        "Menu",
        [
            "Train Model",
            "Predict",
            "About"
        ]
    )

    if menu == "Train Model":

        if st.button("Start Training"):
            train_model()

    elif menu == "Predict":

        if not os.path.exists(MODEL_PATH):
            st.warning("Please train the model first.")
        else:
            prediction_page()

    else:

        st.markdown("""
## About

This application demonstrates an Iris flower classifier using a SimpleRNN.

### Dataset Features
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Output Classes
- Setosa
- Versicolor
- Virginica

### Model
- SimpleRNN(32)
- Dense(16, ReLU)
- Dense(3, Softmax)

Input Shape:
(samples, 1, 4)
""")


if __name__ == "__main__":
    main()
