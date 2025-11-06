import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import base64
import io
import os

# -------------------------
# Config
# -------------------------
MODEL_FILENAME = "model.pkl"  # put your trained sklearn model here
BACKGROUND_IMAGE = "liver_bg.png"  # place a liver drawing image in the same folder

st.set_page_config(page_title="Liver Disease Detection", layout="centered")

# -------------------------
# Utility functions
# -------------------------

def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def image_to_base64(img_path):
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None


def set_background(img_path):
    """Set a full-page background image using CSS and base64 encoding."""
    b64 = image_to_base64(img_path)
    if not b64:
        return
    css = f"""
    <style>
    .stApp {{
      background-image: url('data:image/png;base64,{b64}');
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
      filter: none;
    }}
    /* optional: make main container slightly translucent so text is readable */
    .stApp > header {{visibility: hidden;}}
    .css-1d391kg {{background: rgba(255,255,255,0.82); border-radius: 10px; padding: 1rem;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -------------------------
# Main UI
# -------------------------

def main():
    # Background
    set_background(BACKGROUND_IMAGE)

    # Header / first impression
    st.markdown("""
    <div style='text-align:center; padding:10px;'>
      <h1 style='margin:0; font-size:42px;'>Liver health is important</h1>
      <p style='margin:0; font-size:18px; color:#333;'>Check your liver health — quick, private, and based on clinical features.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")

    # Layout: two columns, left for inputs, right for info + model upload
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Enter patient parameters")
        # Feature inputs as listed in user's image
        age = st.slider("Age of the patient", 1, 120, 45)
        gender = st.radio("Gender of the patient", options=["Male", "Female"])  # encode to numeric later

        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=1.0, step=0.1, format="%.2f")
        direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.1, step=0.1, format="%.2f")
        alkphos = st.number_input("Alkphos (Alkaline Phosphatase)", min_value=0.0, value=200.0, step=1.0, format="%.1f")
        sgpt = st.number_input("SGPT (Alanine Aminotransferase)", min_value=0.0, value=20.0, step=1.0, format="%.1f")
        sgot = st.number_input("SGOT (Aspartate Aminotransferase)", min_value=0.0, value=20.0, step=1.0, format="%.1f")
        total_proteins = st.number_input("Total Proteins", min_value=0.0, value=7.0, step=0.1, format="%.2f")
        alb = st.number_input("ALB (Albumin)", min_value=0.0, value=3.5, step=0.1, format="%.2f")
        ag_ratio = st.number_input("A/G Ratio (Albumin/Globulin)", min_value=0.0, value=1.0, step=0.1, format="%.2f")

        st.markdown("\n")
        predict_btn = st.button("Predict")

    with col2:
        st.header("Model & Notes")
        st.info("This app loads a pre-trained model named 'model.pkl' in the app folder. Uploading a different model is supported.")

        uploaded_model = st.file_uploader("(Optional) Upload model.pkl", type=["pkl"])
        if uploaded_model is not None:
            # save to disk for reuse
            model_path = os.path.join(os.getcwd(), MODEL_FILENAME)
            with open(model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.success("Model uploaded and saved as model.pkl")

        st.markdown("**Feature mapping (index -> name)**")
        st.write({
            0: "Age of the patient",
            1: "Gender of the patient (Male/Female)",
            2: "Total Bilirubin",
            3: "Direct Bilirubin",
            4: "Alkphos Alkaline Phosphatase",
            5: "Sgpt Alamine Aminotransferase",
            6: "Sgot Aspartate Aminotransferase",
            7: "Total Proteins",
            8: "ALB Albumin",
            9: "A/G Ratio Albumin and Globulin Ratio"
        })

    # Load model (from file system)
    model = load_model(MODEL_FILENAME)
    if model is None:
        st.warning("No model found in the app folder. Put your trained model file named 'model.pkl' in the folder, or upload one above.")

    # Prediction action
    if predict_btn:
        if model is None:
            st.error("Cannot predict because no model is loaded.")
        else:
            # Preprocess input into model's expected feature order
            # Encoding gender: common datasets use 1=male, 0=female or vice versa. We'll default to Male=1 Female=0
            gender_enc = 1 if gender == "Male" else 0

            X = np.array([[
                float(age),
                float(gender_enc),
                float(total_bilirubin),
                float(direct_bilirubin),
                float(alkphos),
                float(sgpt),
                float(sgot),
                float(total_proteins),
                float(alb),
                float(ag_ratio)
            ]])

            # If the model expects a DataFrame with named columns, try to convert
            try:
                # Attempt predict_proba
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    # If binary classification, proba[:,1] is probability of positive (disease)
                    if proba.shape[1] == 2:
                        p_disease = float(proba[:, 1][0])
                    else:
                        p_disease = float(np.max(proba))
                else:
                    # fallback to predict
                    pred = model.predict(X)[0]
                    # assume label 1 means disease
                    p_disease = 1.0 if pred == 1 else 0.0
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

            # Display results
            colA, colB = st.columns([2, 3])
            with colA:
                st.subheader("Result")
                threshold = 0.5
                if p_disease >= threshold:
                    st.markdown(f"<h2 style='color:#b00020'>Disease likely ({p_disease*100:.1f}%)</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color:#0a8f3d'>No disease likely ({(1-p_disease)*100:.1f}% healthy signal)</h2>", unsafe_allow_html=True)

            with colB:
                st.subheader("Probability")
                st.progress(min(max(p_disease, 0.0), 1.0))

            st.write("\n")
            st.markdown("**Model note:** Predictions are only as good as the model and input features. This is a demo — consult a clinician for medical advice.")

    # Footer with helpful tips
    st.write("---")
    st.markdown("**Tips while using this app**\n\n- Ensure your model expects the same feature order shown above.\n- Common label encoding for `Gender`: Male=1, Female=0 (adjust your preprocessing if different).\n- If your model was trained on scaled features, the app should apply the same scaler before prediction (include scaler in the pickle or adapt the code).\n")


if __name__ == "__main__":
    main()


# -------------------------
# Developer notes (keep in the folder with the script):
# 1. Place your trained model in the same directory and name it 'model.pkl'. The pickle should contain
#    either a scikit-learn estimator (with predict/predict_proba) or a pipeline which includes preprocessing.
# 2. Place a liver background image named 'liver_bg.png' in the same directory (or edit BACKGROUND_IMAGE).
# 3. Run with: streamlit run streamlit_liver_app.py
# 4. If your model expects different feature order / scaling, update the X construction or wrap with a Pipeline.
# -------------------------
