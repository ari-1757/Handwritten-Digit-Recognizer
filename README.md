# Handwritten Digit Recognizer

A Machine Learning web application that recognizes handwritten digits (0–9) drawn by the user.  
The model is trained using **Support Vector Machine (SVM)** on the **MNIST dataset** and deployed using **Streamlit**.

---

## Features

- Draw digits in a browser canvas
- Real-time digit prediction
- Confidence score visualization
- Clear canvas button
- MNIST-style preprocessing pipeline
- Interactive web interface

---

## Project Structure

```
digit_recoganization/
│
├── digit_gui_predictor.py
├── digit_svm_model.pkl
│
├── train_model.py
│
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

---

## Installation

Install required libraries:

```bash
pip install streamlit streamlit-drawable-canvas numpy opencv-python joblib scikit-learn
```

---

## Run the Application

Start the Streamlit server:

```bash
streamlit run digit_gui_predictor.py
```

Then open your browser:

```
http://localhost:8501
```

Draw a digit in the canvas and the model will predict the number.

---

## Model Training

Example training code:

```python
from sklearn import svm
from sklearn.metrics import accuracy_score

model = svm.SVC(kernel='rbf')

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
```

The trained model is saved as:

```
digit_svm_model.pkl
```

---

## Preprocessing Pipeline

The input image is processed before prediction:

1. Convert to grayscale
2. Invert colors
3. Apply Gaussian blur
4. Threshold the image
5. Detect digit bounding box
6. Resize while preserving aspect ratio
7. Center digit using center of mass
8. Pad image to **28×28**
9. Normalize pixel values

This makes the input similar to the **MNIST dataset** format.

---

## Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Scikit-learn
- Joblib

---

## Future Improvements

- Upgrade model to CNN for higher accuracy
- Deploy the app online
- Add real-time prediction while drawing
- Improve UI design

---

## License

Educational project for Machine Learning practice.
