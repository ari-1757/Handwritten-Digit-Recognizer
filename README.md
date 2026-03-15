# Handwritten-Digit-Recognizer
Overview

This project is a handwritten digit recognition web application that allows users to draw a digit (0–9) on a canvas and get a prediction from a trained machine learning model.

The model is trained using a Support Vector Machine (SVM) on the MNIST dataset.
The user interface is built with Streamlit, allowing real-time interaction through a web browser.

Features

Draw digits directly in the browser

Real-time prediction using a trained SVM model

Confidence score visualization for each digit (0–9)

Canvas reset functionality

Image preprocessing pipeline similar to MNIST

Clean two-column Streamlit interface

Project Structure
digit_recoganization/
│
├── digit_gui_predictor.py      # Streamlit web application
├── digit_svm_model.pkl         # Trained SVM model
│
├── train_model.py              # Model training script
│
├── train-images.idx3-ubyte     # MNIST training images
├── train-labels.idx1-ubyte     # MNIST training labels
├── t10k-images.idx3-ubyte      # MNIST test images
└── t10k-labels.idx1-ubyte      # MNIST test labels
Installation

Install the required dependencies:

pip install streamlit streamlit-drawable-canvas numpy opencv-python joblib scikit-learn
Running the Application

Start the Streamlit web application:

streamlit run digit_gui_predictor.py

Then open the browser at:

http://localhost:8501

Draw a digit in the canvas and the model will predict the number.

Model Training

The model is trained using the MNIST dataset with the following pipeline:

Load MNIST image and label files

Convert images to 784 feature vectors (28×28)

Train a Support Vector Machine classifier

Save the trained model using joblib

Example training code:

from sklearn import svm
from sklearn.metrics import accuracy_score

model = svm.SVC(kernel='rbf')

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

The trained model is saved as:

digit_svm_model.pkl
Image Preprocessing Pipeline

To ensure compatibility with the MNIST dataset, the following preprocessing steps are applied:

Convert drawing to grayscale

Invert colors (white digit on black background)

Apply Gaussian blur for noise removal

Threshold the image

Detect digit bounding box

Resize while preserving aspect ratio

Center digit using center-of-mass

Pad image to 28×28

Normalize pixel values

This preprocessing significantly improves prediction accuracy.

Example Output

After drawing a digit, the application displays:

Predicted digit

Processed 28×28 image

Confidence scores for digits 0–9

Technologies Used

Python

Streamlit

OpenCV

NumPy

Scikit-learn

Joblib

Future Improvements

Possible enhancements include:

Replace SVM with a Convolutional Neural Network (CNN)

Add real-time prediction while drawing

Deploy the application online

Improve UI/UX

Add probability bar charts

License

This project is for educational and research purposes.
