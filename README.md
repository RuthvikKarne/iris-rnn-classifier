# Iris Flower Classification using SimpleRNN (One-to-One)

A Streamlit web application that demonstrates how to train and deploy a Recurrent Neural Network (RNN) using TensorFlow/Keras to classify Iris flowers. The model is structured to process the 4 features of the Iris dataset as a single sequence of data (one-to-one architecture).

## Features

- **Interactive Training**: Train a SimpleRNN model directly from the UI with real-time progress.
- **Performance Metrics**: View the classification report and confusion matrix right after training.
- **Real-time Prediction**: Enter Sepal and Petal dimensions to instantly predict the species (Setosa, Versicolor, or Virginica) along with confidence probabilities.
- **Model Persistence**: Automatically saves and loads the trained model (`iris_rnn.keras`) and scaler (`scaler.pkl`).

## Neural Network Architecture

The network treats the 4 input features (Sepal Length, Sepal Width, Petal Length, Petal Width) as a sequence of length 1 with 4 features per step:
- **Input Shape**: `(samples, 1, 4)`
- **Layer 1**: `SimpleRNN(32)`
- **Layer 2**: `Dense(16, activation='relu')`
- **Output Layer**: `Dense(3, activation='softmax')`

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd iris-rnn-classifier
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your browser (usually `http://localhost:8501`).
2. Navigate to the **Train Model** menu from the sidebar and click **Start Training**.
3. Once training is complete, navigate to the **Predict** menu.
4. Adjust the flower dimensions using the input fields and click **Predict** to see the model's classification and confidence scores.

## Technologies Used

- [Streamlit](https://streamlit.io/) for the frontend interface.
- [TensorFlow / Keras](https://www.tensorflow.org/) for building and training the RNN.
- [Scikit-Learn](https://scikit-learn.org/) for dataset loading, preprocessing, and metrics.
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) for data manipulation.
