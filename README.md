# 🧮 Handwritten Math Solver

A web app that reads a handwritten math expression — drawn on a canvas or uploaded as a photo — and solves it automatically. It uses a **Convolutional Neural Network (CNN)**, a type of deep learning model especially good at recognizing patterns in images, to identify each handwritten digit and operator before computing the answer.

🔗 **Live app:** [handwritten-simple-math-solver.streamlit.app](https://handwritten-simple-math-solver.streamlit.app/)

---

## 🧠 How It Works

1. **Preprocessing** — the input image (drawing or photo) is converted to grayscale and thresholded so the strokes stand out clearly from the background.
2. **Segmentation** — OpenCV's connected component analysis finds each individual symbol in the expression and draws a bounding box around it, left to right.
3. **Classification (CNN)** — each cropped symbol is fed into the CNN, which predicts what it is: a digit (0–9) or an operator (+, −, ×, ÷, ., =). The CNN works by sliding small filters over the image to detect edges and shapes, then combining those into higher-level features — much like how the human eye recognizes handwriting strokes before recognizing the full character.
4. **Parsing & Evaluation** — the predicted symbols are joined into a valid expression and evaluated to produce the final answer.

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — builds and runs the CNN that classifies each handwritten symbol
- **OpenCV** — image preprocessing and symbol segmentation (connected components)
- **NumPy** — array/image data handling
- **Streamlit** — the web app framework powering the UI
- **streamlit-drawable-canvas** — lets users draw expressions directly in the browser
- **Pillow (PIL)** — image loading for uploaded photos
- **Matplotlib** — visualizes detected symbols, bounding boxes, and confidence scores

**Training data:** MNIST (handwritten digits 0–9) + a Kaggle handwritten math symbols dataset (+, −, ×, ÷, ., =)

---

## 🐍 Python Version

**Python 3.10** is recommended for best compatibility with the pinned TensorFlow/Keras versions in `requirements.txt`.

```bash
py -3.10 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

On Streamlit Community Cloud, set this under **App settings → Python version → 3.10** if not auto-detected, to match the environment the model was trained/exported with.

---

## 📁 Project Structure

```
math_solver_app/
├── app.py
├── requirements.txt
├── handwritten_math_solver.keras   # full saved model
├── model.weights.h5                # weights-only fallback (more version-tolerant)
└── README.md
```

The app tries loading `handwritten_math_solver.keras` first; if that fails due to a Keras version mismatch, it automatically falls back to rebuilding the CNN architecture in code and loading `model.weights.h5`.

---

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Draw a sum or upload a photo to test.

---

## 🚀 Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (model files included — small enough, no Git LFS needed).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub.
3. **New app** → select your repo, branch `main`, file `app.py` → **Deploy**.

---

## 🔧 Troubleshooting

- **Model load error (Keras deserialization)**: usually a Keras version mismatch between the training environment (Colab) and the one running the app. The `model.weights.h5` fallback handles this automatically.
- **Canvas not drawing**: confirm `streamlit-drawable-canvas` installed correctly.
- **Low accuracy on photos**: works best with dark, thick strokes on a plain light background.
- **Slow first prediction**: TensorFlow initializes on first run — normal, later predictions are fast.