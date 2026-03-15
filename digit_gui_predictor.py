import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import joblib

# Load trained model
model = joblib.load("digit_svm_model.pkl")

st.set_page_config(page_title="Digit Recognizer", layout="wide")

st.title("Handwritten Digit Recognizer")

st.sidebar.header("Model Info")
st.sidebar.write("Model: SVM")
st.sidebar.write("Dataset: MNIST")

# Canvas reset state
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

col1, col2 = st.columns(2)

# DRAWING AREA
with col1:

    st.subheader("Draw Digit")

    if st.button("Clear Canvas"):
        st.session_state.canvas_key += 1

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )


# HIGH-ACCURACY PREPROCESSING
def preprocess(img):

    img = np.array(img)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert colors (MNIST style)
    img = 255 - img

    # Noise removal
    img = cv2.GaussianBlur(img,(5,5),0)

    # Threshold
    _, img = cv2.threshold(img,30,255,cv2.THRESH_BINARY)

    coords = cv2.findNonZero(img)

    if coords is None:
        return None, None

    x,y,w,h = cv2.boundingRect(coords)

    digit = img[y:y+h, x:x+w]

    # Preserve aspect ratio
    if h > w:
        new_h = 20
        new_w = int(w * (20/h))
    else:
        new_w = 20
        new_h = int(h * (20/w))

    digit = cv2.resize(digit,(new_w,new_h))

    # Create 28x28 canvas
    canvas = np.zeros((28,28),dtype=np.uint8)

    x_offset = (28-new_w)//2
    y_offset = (28-new_h)//2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    # Center using center of mass
    M = cv2.moments(canvas)

    if M["m00"] != 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])

        shift_x = 14 - cx
        shift_y = 14 - cy

        matrix = np.float32([[1,0,shift_x],[0,1,shift_y]])

        canvas = cv2.warpAffine(canvas, matrix, (28,28))

    preview = canvas.copy()

    # Normalize
    canvas = canvas / 255.0

    canvas = canvas.reshape(1,784)

    return canvas, preview


# PREDICTION AREA
with col2:

    st.subheader("Prediction")

    if canvas_result.image_data is not None:

        img = canvas_result.image_data.astype("uint8")

        processed, preview = preprocess(img)

        if processed is not None:

            prediction = model.predict(processed)

            st.success(f"Predicted Digit: {prediction[0]}")

            st.write("Model Input (28x28)")
            st.image(preview,width=150)

            # Confidence scores
            scores = model.decision_function(processed)

            st.write("Confidence Scores")

            min_score = min(scores[0])
            max_score = max(scores[0])

            for i, score in enumerate(scores[0]):

                normalized = (score-min_score)/(max_score-min_score+1e-5)

                st.progress(float(normalized))
                st.write(f"Digit {i}: {score:.2f}")

        else:
            st.write("Draw a digit first.")