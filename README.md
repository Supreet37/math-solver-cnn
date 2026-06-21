# 🧮 Handwritten Math Solver — Streamlit App

This is a ready-to-deploy Streamlit web app for your trained CNN model.
It lets users either **draw** an expression on a canvas or **upload a photo**,
then runs your exact notebook pipeline: preprocess → segment → classify → parse → solve.

---

## ⚠️ Step 0 — You need the trained model file first

This app does **not** train anything — it only *runs* your already-trained model.

1. Open your notebook in **Google Colab**.
2. Run every cell, including **Step 4 (Train)** and **Step 11 (Save the Model)**.
3. This downloads `handwritten_math_solver.keras` to your computer.
4. Put that file in this same folder, next to `app.py`.

Folder should look like:
```
math_solver_app/
├── app.py
├── requirements.txt
├── handwritten_math_solver.keras   ← you add this
└── README.md
```

---

## 🖥️ Run it locally (test before deploying)

```bash
cd math_solver_app
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`. Try drawing a sum or uploading a photo.

---

## 🚀 Deploy for free — Streamlit Community Cloud

This gives you a real public link like `https://your-app-name.streamlit.app`.

### 1. Push this folder to GitHub
```bash
cd math_solver_app
git init
git add .
git commit -m "Handwritten math solver app"
git branch -M main
git remote add origin https://github.com/ompreet-s/math-solver-app.git
git push -u origin main
```
> Make sure `handwritten_math_solver.keras` is included in the push — check it's not
> in a `.gitignore`. If it's too large for GitHub's normal limit (it usually won't be —
> this model is small, typically a few MB), use [Git LFS](https://git-lfs.github.com/).

### 2. Deploy
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo `math-solver-app`, branch `main`, file `app.py`
5. Click **Deploy**

That's it — within a couple of minutes you'll get a public URL you can share with your instructor.

---

## 🔧 Notes / Troubleshooting

- **Model not found error**: means `handwritten_math_solver.keras` isn't in the repo/folder. Re-check step 0.
- **Canvas not drawing**: make sure `streamlit-drawable-canvas` installed correctly (`pip show streamlit-drawable-canvas`).
- **Low accuracy on photos**: works best with dark, thick strokes on a plain light background — same kind of input your notebook's Step 10 test expects.
- **Slow first prediction**: TensorFlow takes a few seconds to initialize on first run — totally normal, later predictions are fast.
- **requirements.txt versions**: pinned to known-stable versions. If Streamlit Cloud's Python version conflicts, drop the `==x.x.x` pins and let it resolve freely.
