"""
PixelMind: Deep Learning Image Recognition System
===================================================
A production-ready Streamlit web application that serves a trained
CIFAR-10 Convolutional Neural Network (CNN) for real-time image
classification.

Run locally with:
    streamlit run app.py
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
# Streamlit powers the web UI. TensorFlow loads/runs the trained CNN.
# NumPy handles array math, Pillow handles image decoding/resizing, and
# Matplotlib renders the probability bar chart.
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt


# =============================================================================
# 2. PAGE CONFIGURATION
# =============================================================================
# st.set_page_config must be the first Streamlit command executed in the
# script. It controls the browser tab title/icon and overall page layout.
st.set_page_config(
    page_title="PixelMind | CIFAR-10 Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 3. CONSTANTS
# =============================================================================
# Path to the trained model file. Keep this file in the same directory as
# app.py (this is also what the deployment instructions assume).
MODEL_PATH = "cifar_img_classifier.keras"

# The CNN was trained on 32x32 RGB images, so every uploaded image must be
# resized to this exact shape before being fed to the model.
IMG_SIZE = (32, 32)

# CIFAR-10 class labels, in the exact order the model's output layer was
# trained on (index 0 -> "Airplane", index 1 -> "Automobile", etc.).
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]


# =============================================================================
# 4. MODEL LOADING (CACHED)
# =============================================================================
# @st.cache_resource ensures the (potentially large) Keras model is loaded
# into memory only ONCE per server process, instead of on every single user
# interaction/script rerun. This is the recommended pattern for ML models,
# DB connections, and other expensive, non-serializable resources.
@st.cache_resource(show_spinner="Loading the PixelMind CNN model...")
def load_model():
    """Load and return the trained Keras model, or None if loading fails."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except (OSError, ValueError) as exc:
        # OSError -> file not found / unreadable. ValueError -> corrupt or
        # incompatible model file. Both are surfaced to the user instead of
        # crashing the app.
        st.error(
            f"❌ Could not load the model from '{MODEL_PATH}'.\n\n"
            f"Details: {exc}"
        )
        return None


# =============================================================================
# 5. IMAGE PREPROCESSING
# =============================================================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into the exact tensor shape/format the model
    expects: (1, 32, 32, 3) float32 array with pixel values in [0, 1].
    """
    # Step 1: Force 3-channel RGB. This silently handles RGBA, grayscale
    # ("L"), or palette ("P") images that would otherwise break the model's
    # fixed 3-channel input shape.
    image = image.convert("RGB")

    # Step 2: Resize to the model's expected spatial dimensions.
    image = image.resize(IMG_SIZE)

    # Step 3: Convert to a NumPy array and normalize pixel values from the
    # standard [0, 255] range down to [0, 1], matching how the training
    # data was scaled.
    img_array = np.array(image).astype("float32") / 255.0

    # Step 4: Add a batch dimension -> shape becomes (1, 32, 32, 3), since
    # Keras models expect a batch axis even for a single prediction.
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =============================================================================
# 6. PREDICTION
# =============================================================================
def predict(model, img_array: np.ndarray):
    """
    Run inference and return (predicted_label, confidence_pct, all_probs).
    """
    # verbose=0 suppresses TensorFlow's progress bar in the Streamlit logs.
    probabilities = model.predict(img_array, verbose=0)[0]

    top_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[top_index]
    confidence_pct = float(probabilities[top_index]) * 100.0

    return predicted_label, confidence_pct, probabilities


# =============================================================================
# 7. PROBABILITY BAR CHART
# =============================================================================
def plot_probabilities(probabilities: np.ndarray):
    """Build a Matplotlib bar chart of the per-class prediction probabilities."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Highlight the predicted (highest-probability) class in a different
    # color so it stands out at a glance.
    top_index = int(np.argmax(probabilities))
    bar_colors = ["#4C72B0"] * len(CLASS_NAMES)
    bar_colors[top_index] = "#DD8452"

    ax.bar(CLASS_NAMES, probabilities * 100, color=bar_colors)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Prediction Probability by Class")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    return fig


# =============================================================================
# 8. SIDEBAR (PROJECT DESCRIPTION)
# =============================================================================
def render_sidebar():
    """Render the static project-description sidebar."""
    with st.sidebar:
        st.title("🧠 PixelMind")
        st.caption("Deep Learning Image Recognition System")

        st.markdown("### About")
        st.write(
            "PixelMind is a Convolutional Neural Network (CNN) trained on "
            "the CIFAR-10 dataset. Upload any photo and the model will "
            "classify it into one of 10 everyday object categories."
        )

        st.markdown("### How it works")
        st.write(
            "1. Upload a JPG, JPEG, or PNG image\n"
            "2. The image is resized to 32×32 and normalized\n"
            "3. The CNN predicts class probabilities\n"
            "4. The top prediction and full distribution are displayed"
        )

        st.markdown("### Recognized classes")
        st.write(", ".join(CLASS_NAMES))

        st.markdown("---")
        st.caption("Built with TensorFlow/Keras, Streamlit, and Matplotlib.")


# =============================================================================
# 9. MAIN APPLICATION
# =============================================================================
def main():
    render_sidebar()

    # --- Header -------------------------------------------------------------
    st.title("PixelMind: Deep Learning Image Recognition System")
    st.write(
        "Upload an image below and the CNN will classify it into one of "
        "10 CIFAR-10 categories, along with a confidence score and full "
        "probability breakdown."
    )
    st.divider()

    # --- Load model (cached) -------------------------------------------------
    model = load_model()
    if model is None:
        # Stop execution here so the rest of the UI (which depends on the
        # model) never tries to run against a None object.
        st.stop()

    # --- File upload widget ---------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        help="For best results, use a clear, well-lit photo of a single object.",
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to get started.")
        return

    # --- Validate and open the uploaded file -----------------------------
    try:
        image = Image.open(uploaded_file)
        image.load()  # Forces Pillow to fully decode the file now, so any
        # corruption is caught here instead of later mid-prediction.
    except UnidentifiedImageError:
        st.error(
            "⚠️ The uploaded file could not be read as an image. "
            "Please upload a valid JPG, JPEG, or PNG file."
        )
        return
    except Exception as exc:  # noqa: BLE001 - surfacing any unexpected I/O error
        st.error(f"⚠️ An unexpected error occurred while reading the file: {exc}")
        return

    # --- Layout: image on the left, results on the right ---------------------
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # --- Preprocess + predict (with error handling) --------------------------
    try:
        processed = preprocess_image(image)
        predicted_label, confidence_pct, probabilities = predict(model, processed)
    except Exception as exc:  # noqa: BLE001
        st.error(f"⚠️ Prediction failed: {exc}")
        return

    with col2:
        st.subheader("Prediction Result")
        st.metric(label="Predicted Class", value=predicted_label)
        st.write(f"**Confidence:** {confidence_pct:.2f}%")
        st.progress(min(int(round(confidence_pct)), 100))

    st.divider()

    # --- Full probability distribution ----------------------------------------
    st.subheader("Probability Distribution Across All Classes")
    fig = plot_probabilities(probabilities)
    st.pyplot(fig)

    with st.expander("View raw probability values"):
        for class_name, prob in sorted(
            zip(CLASS_NAMES, probabilities), key=lambda x: x[1], reverse=True
        ):
            st.write(f"**{class_name}:** {prob * 100:.2f}%")


# =============================================================================
# 10. FOOTER (CREDITS)
# =============================================================================
def render_footer():
    """
    Render a simple credit footer at the bottom of every page.
    Edit the placeholder text below with your own name/links.
    """
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; opacity: 0.7; font-size: 0.85rem;">
            Developed by <b>M.A.T</b> &nbsp;·&nbsp; © 2026 PixelMind<br>
            Built with TensorFlow, Keras &amp; Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
if __name__ == "__main__":
    main()
    render_footer()