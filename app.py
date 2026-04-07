import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
# 🔥 Page config
st.set_page_config(page_title="AI Skin Cancer Detector", layout="centered")

# 🔥 PREMIUM CSS
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00E5FF;
}
.subtitle {
    text-align: center;
    color: #d1d1d1;
    margin-bottom: 25px;
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 0px 20px rgba(0,255,255,0.2);
}
.result-box {
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 🔥 HEADER
st.markdown('<div class="title">🧠 AI Skin Cancer Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning Based Medical Assistant</div>', unsafe_allow_html=True)

# 🔥 SIDEBAR
st.sidebar.title("📌 About")
st.sidebar.info("""
This app uses a deep learning CNN model to detect skin cancer risk.

⚠️ Not a medical diagnosis.
""")
st.sidebar.metric("Model Accuracy", "≈ 88%")
st.markdown("---")
st.caption("Built by Subash | AI Skin Cancer Detector")


print("Current Dir:", os.getcwd())

model = load_model("final_model.keras", compile=False)


# 🔥 MAIN CARD
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg","png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

        # preprocess
        img = cv2.resize(img, (224,224))
        img = img / 255.0
        img = img.reshape(1,224,224,3)

        # 🔥 Loading animation
        with st.spinner("🔍 Analyzing image..."):
            prediction = model.predict(img)[0][0]

        st.markdown("---")

        # 🔥 Confidence bar
        st.progress(float(prediction))

        # 🔥 RESULT DISPLAY
        if prediction > 0.4:
            st.markdown(
                f'<div class="result-box" style="color:red;">🚨 Malignant (Cancer)</div>',
                unsafe_allow_html=True
            )
            st.error(f"Confidence: {prediction:.2f}")
        else:
            st.markdown(
                f'<div class="result-box" style="color:lime;">✅ Benign (Safe)</div>',
                unsafe_allow_html=True
            )
            st.success(f"Confidence: {1 - prediction:.2f}")

        # 🔥 Disclaimer
        st.warning("⚠️ This is an AI prediction. Consult a doctor for medical advice.")

    st.markdown('</div>', unsafe_allow_html=True)
