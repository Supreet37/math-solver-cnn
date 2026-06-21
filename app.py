"""
Handwritten Math Expression Solver — Streamlit App
Deploy target: Streamlit Community Cloud (share.streamlit.io)

Pipeline (matches the training notebook exactly):
  Image -> Preprocess (grayscale/threshold) -> Segment (connected components)
        -> CNN classify each symbol crop -> Parse tokens -> Evaluate -> Result
"""

import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

# streamlit-drawable-canvas gives us an actual drawing canvas in the browser
from streamlit_drawable_canvas import st_canvas

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Math Solver",
    page_icon="🧮",
    layout="wide",
)

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               '+', '-', 'x', '/', '.', '=']
MODEL_PATH = "handwritten_math_solver.keras"      # used if present (full model load)
WEIGHTS_PATH = "model.weights.h5"                  # fallback (weights-only load)


def build_cnn(num_classes=16, input_shape=(28, 28, 1)):
    """Must exactly match the architecture in the training notebook,
    so that weights-only loading lines up layer-for-layer."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='MathSymbolCNN')


# ──────────────────────────────────────────────────────────────────────────
# PIPELINE CLASSES — ported directly from the notebook
# ──────────────────────────────────────────────────────────────────────────
class MathExpressionSolver:
    OPERATOR_MAP = {'+': '+', '-': '-', 'x': '*', '/': '/', '.': '.', '=': '='}

    def parse(self, symbols: list) -> str:
        tokens = []
        i = 0
        while i < len(symbols):
            sym = symbols[i]
            if sym.isdigit() or sym == '.':
                num = sym
                while i + 1 < len(symbols) and (symbols[i + 1].isdigit() or symbols[i + 1] == '.'):
                    i += 1
                    num += symbols[i]
                tokens.append(num)
            elif sym in self.OPERATOR_MAP:
                mapped = self.OPERATOR_MAP[sym]
                if mapped == '=' and tokens and tokens[-1] == '=':
                    i += 1
                    continue
                tokens.append(mapped)
            i += 1
        return ' '.join(tokens)

    def evaluate(self, expression: str):
        try:
            allowed = set('0123456789 +-*/.()=')
            if not all(c in allowed for c in expression):
                return None, "Invalid characters in expression"

            if '=' in expression:
                lhs = expression.split('=')[0].strip()
                if not lhs:
                    return None, "Empty expression before '='"
                result = eval(lhs)
                return result, None

            result = eval(expression)
            return result, None
        except ZeroDivisionError:
            return None, "Division by zero"
        except Exception as e:
            return None, str(e)

    def solve(self, symbols: list):
        expr = self.parse(symbols)
        result, error = self.evaluate(expr)
        return expr, result, error


class SymbolSegmenter:
    """Segments a math expression image into individual symbol crops
    using connected component analysis."""

    def preprocess(self, img_array: np.ndarray) -> np.ndarray:
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()

        if gray.mean() > 127:
            gray = 255 - gray

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def segment(self, binary: np.ndarray, padding=4):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)

        boxes = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            if area < 30:
                continue
            boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: b[0])
        return boxes

    def extract_crops(self, binary: np.ndarray, boxes, size=28, padding=4):
        crops = []
        H, W = binary.shape
        for (x, y, w, h) in boxes:
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(W, x + w + padding)
            y2 = min(H, y + h + padding)
            crop = binary[y1:y2, x1:x2]
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            crops.append(crop.astype('float32') / 255.0)
        return crops


class HandwrittenMathSolver:
    """End-to-end pipeline: image -> answer."""

    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.segmenter = SymbolSegmenter()
        self.solver = MathExpressionSolver()

    def predict_symbols(self, crops):
        if not crops:
            return [], []
        batch = np.array(crops)[..., np.newaxis]
        probs = self.model.predict(batch, verbose=0)
        idxs = np.argmax(probs, axis=1)
        return [self.class_names[i] for i in idxs], probs

    def solve_image(self, img_array: np.ndarray):
        binary = self.segmenter.preprocess(img_array)
        boxes = self.segmenter.segment(binary)
        if not boxes:
            return [], "", None, None, binary, [], []

        crops = self.segmenter.extract_crops(binary, boxes)
        symbols, probs = self.predict_symbols(crops)
        expression, result, error = self.solver.solve(symbols)

        return symbols, expression, result, error, binary, boxes, list(zip(crops, probs))


# ──────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached so it only loads once per session)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Try 1: load the full saved model (architecture + weights)
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e_full:
        full_error = str(e_full)

    # Try 2: rebuild architecture in code, then load weights only.
    # This sidesteps Keras version mismatches in saved-config deserialization.
    try:
        import os
        if os.path.exists(WEIGHTS_PATH):
            model = build_cnn(num_classes=len(CLASS_NAMES))
            model.load_weights(WEIGHTS_PATH)
            return model, None
        else:
            return None, (
                f"Full model load failed:\n{full_error}\n\n"
                f"Also tried weights-only fallback ('{WEIGHTS_PATH}') but that file wasn't found either."
            )
    except Exception as e_weights:
        return None, (
            f"Full model load failed:\n{full_error}\n\n"
            f"Weights-only fallback also failed:\n{e_weights}"
        )


# ──────────────────────────────────────────────────────────────────────────
# VISUALIZATION (matplotlib figure -> shown in Streamlit)
# ──────────────────────────────────────────────────────────────────────────
def build_visualization(orig, binary, boxes, crops_probs, symbols):
    n = max(len(boxes), 1)
    fig = plt.figure(figsize=(max(10, n * 1.8), 6))
    gs = fig.add_gridspec(2, n + 2, hspace=0.5, wspace=0.3)

    ax_orig = fig.add_subplot(gs[0, :2])
    ax_orig.imshow(orig, cmap='gray' if orig.ndim == 2 else None)
    ax_orig.set_title('Input', fontweight='bold', fontsize=10)
    ax_orig.axis('off')
    for (x, y, w, h) in boxes:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax_orig.add_patch(rect)

    ax_bin = fig.add_subplot(gs[1, :2])
    ax_bin.imshow(binary, cmap='gray')
    ax_bin.set_title('Preprocessed', fontweight='bold', fontsize=10)
    ax_bin.axis('off')

    for i, ((crop, prob), sym) in enumerate(zip(crops_probs, symbols)):
        ax = fig.add_subplot(gs[0, i + 2])
        ax.imshow(crop, cmap='gray')
        conf = float(np.max(prob)) * 100
        ax.set_title(f"'{sym}'\n{conf:.0f}%", fontsize=9,
                     color='green' if conf > 80 else 'orange')
        ax.axis('off')

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────────────────────────────────
def run_pipeline_and_show(img_array, solver):
    symbols, expression, result, error, binary, boxes, crops_probs = solver.solve_image(img_array)

    if not boxes:
        st.warning("No symbols detected. Try a clearer image with darker, thicker strokes.")
        return

    fig = build_visualization(img_array, binary, boxes, crops_probs, symbols)
    st.pyplot(fig)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Detected symbols:**")
        st.code(" ".join(symbols))
        st.markdown("**Parsed expression:**")
        st.code(expression if expression else "—")
    with col2:
        if result is not None:
            st.success(f"### ✅ Answer: {expression} = {result:.6g}" if isinstance(result, float)
                        else f"### ✅ Answer: {expression} = {result}")
        else:
            st.error(f"### ❌ Could not evaluate\n{error}")


# ──────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────
def main():
    st.title("🧮 Handwritten Math Expression Solver")
    st.caption("Draw a math expression or upload a photo — the CNN model reads it and solves it.")

    model, load_error = load_model()

    if model is None:
        st.error(
            f"**Model failed to load.**\n\n"
            f"Error details:\n```\n{load_error}\n```"
        )
        st.info(
            "📌 This app will load the model in one of two ways:\n\n"
            f"**Option A** — `{MODEL_PATH}` (full saved model)\n"
            f"**Option B** — `{WEIGHTS_PATH}` (weights-only, more version-tolerant)\n\n"
            "If you're hitting a Keras deserialization error with Option A "
            "(common when Colab's Keras version differs from your local one), "
            "go back to Colab and add this cell after training:\n"
            "```python\nmodel.save_weights('model.weights.h5')\nfiles.download('model.weights.h5')\n```\n"
            "Then place `model.weights.h5` next to `app.py` and re-run. "
            "This rebuilds the architecture in code and only loads the numeric weights, "
            "which avoids version-mismatch errors entirely."
        )
        return

    solver = HandwrittenMathSolver(model, CLASS_NAMES)

    tab1, tab2 = st.tabs(["✏️ Draw", "📷 Upload Image"])

    # ── TAB 1: Draw on canvas ──────────────────────────────────────────
    with tab1:
        st.write("Draw your expression below (e.g. `12+34`, `9x8`, `100/4`):")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=10,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=200,
            width=700,
            drawing_mode="freedraw",
            key="canvas",
        )

        col_a, col_b = st.columns([1, 5])
        with col_a:
            solve_clicked = st.button("Solve", type="primary", key="solve_draw")

        if solve_clicked:
            if canvas_result.image_data is None or canvas_result.image_data[:, :, :3].sum() == 0:
                st.warning("Please draw something first.")
            else:
                img_array = canvas_result.image_data[:, :, :3].astype(np.uint8)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                with st.spinner("Solving..."):
                    run_pipeline_and_show(gray, solver)

    # ── TAB 2: Upload image ────────────────────────────────────────────
    with tab2:
        uploaded_file = st.file_uploader("Upload a photo of a handwritten expression",
                                          type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("L")
            img_array = np.array(image)
            st.image(image, caption="Uploaded image", width=400)
            if st.button("Solve", type="primary", key="solve_upload"):
                with st.spinner("Solving..."):
                    run_pipeline_and_show(img_array, solver)

    st.markdown("---")
    with st.expander("ℹ️ How it works"):
        st.markdown(
            "1. **Preprocess** — convert to grayscale, threshold (Otsu), clean noise.\n"
            "2. **Segment** — connected component analysis finds each symbol's bounding box.\n"
            "3. **Classify** — a CNN (trained on MNIST digits + handwritten operator symbols) "
            "predicts each cropped symbol.\n"
            "4. **Parse & Evaluate** — symbols are joined into a valid expression and evaluated.\n\n"
            f"Classes recognized: `{', '.join(CLASS_NAMES)}`"
        )


if __name__ == "__main__":
    main()