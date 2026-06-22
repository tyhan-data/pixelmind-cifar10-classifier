# Deploying PixelMind to Streamlit Community Cloud

This guide walks through deploying the `PixelMind` app to
[Streamlit Community Cloud](https://streamlit.io/cloud) (free tier).

## 1. Project files

Your project folder should contain exactly these three files together:

```
pixelmind_app/
├── app.py
├── requirements.txt
└── cifar_img_classifier.keras
```

The model file **must** sit in the same folder as `app.py`, since the app
loads it with the relative path `"cifar_img_classifier.keras"`. The model
is ~20 MB, which is well under GitHub's 100 MB file-size limit, so no
Git LFS is required.

## 2. Push the project to GitHub

```bash
cd pixelmind_app
git init
git add app.py requirements.txt cifar_img_classifier.keras
git commit -m "Initial commit: PixelMind CIFAR-10 classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If you don't already have a GitHub account/repo, create one at
[github.com/new](https://github.com/new) first (public or private — both
work with Streamlit Community Cloud).

## 3. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.
2. Click **"New app"**.
3. Select:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. (Optional) Click **"Advanced settings"** to pin a specific Python
   version (3.11 or 3.12 is recommended for TensorFlow compatibility).
5. Click **"Deploy"**.

Streamlit Cloud will install everything listed in `requirements.txt` and
launch the app. The first build typically takes 2–5 minutes because
TensorFlow is a large dependency — subsequent restarts are much faster
thanks to caching.

## 4. Verify the deployment

Once the build finishes, you'll get a public URL like:

```
https://<your-app-name>.streamlit.app
```

Open it and confirm:
- The sidebar renders the project description.
- Uploading a JPG/PNG produces a prediction, confidence score, and bar
  chart.
- Uploading a non-image file (e.g., a `.txt` renamed to `.jpg`) shows a
  graceful error instead of crashing the app.

## 5. Updating the app later

Any time you `git push` new commits to the connected branch, Streamlit
Community Cloud automatically rebuilds and redeploys the app — no manual
redeploy step needed.

## 6. Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | `requirements.txt` missing or misnamed | Confirm the file is named exactly `requirements.txt` at the repo root |
| App build times out | TensorFlow install is large | Re-run the deploy (Streamlit Cloud caches dependencies after the first successful build) |
| `Could not load the model` error in-app | Model file missing from repo or wrong filename | Confirm `cifar_img_classifier.keras` is committed and matches `MODEL_PATH` in `app.py` exactly |
| App sleeps after inactivity | Normal free-tier behavior | Visiting the URL wakes it back up within ~30 seconds |

## 7. (Optional) Resource limits

Streamlit Community Cloud's free tier provides 1 GB of RAM per app. This
CNN model is small enough to run comfortably within that limit; no GPU is
required for inference.
