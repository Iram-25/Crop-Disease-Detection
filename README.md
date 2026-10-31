# Tomato Leaf Disease Detection using CNN

A Convolutional Neural Network (CNN) built with TensorFlow/Keras that classifies tomato leaf images into disease categories, served through a Streamlit web app.

## Project Overview
- **Dataset:** Tomato leaf images ([Kaggle: kaustubhb999/tomatoleaf](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)), split into `train` and `val` folders.
- **Input:** 128×128 RGB image
- **Model:** 3 × (Conv2D + MaxPooling) → Flatten → Dense(128) → Dropout(0.5) → Softmax
- **Augmentation:** random horizontal flip, rotation (0.1), zoom (0.1)
- **Training:** Adam optimizer, sparse categorical cross-entropy, early stopping (patience 10, restores best weights)
- **Evaluation:** validation accuracy, loss/accuracy curves, confusion matrix

## Repository Structure
```
├── app.py                    # Streamlit app
├── tomato-disease.ipynb      # Kaggle training notebook
├── requirements.txt
├── class_names.json          # label order from training (download from Kaggle output)
├── tomato_disease_cnn.keras  # trained model (download from Kaggle output)
├── .gitignore
└── README.md
```

## Files needed before running
The notebook writes these to `/kaggle/working/`. Download them from the notebook's **Output** tab and place them beside `app.py`:

- `tomato_disease_cnn.keras` — the trained model (use the `.keras` file itself, not the `.zip`)
- `class_names.json` — class order saved by the notebook

The model and labels must come from the same training run.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```
Open the URL shown in the terminal and upload a JPG, JPEG, PNG or WebP leaf photo (up to 15 MB). The model includes pixel rescaling internally, so the app passes raw 0–255 pixel values.

## Retrain
Open the notebook on Kaggle, attach the dataset, and run all cells. It trains the model, plots results, and saves the model and `class_names.json`.

## Deploy (Streamlit Community Cloud)
Push `app.py`, `requirements.txt`, `README.md`, `tomato_disease_cnn.keras` and `class_names.json` to GitHub, then select `app.py` as the main file. Use the `MODEL_PATH` and `LABELS_PATH` environment variables if the files live elsewhere.

## Limitations
- Trained on tomato leaves only; predictions on other plants are not meaningful.
- Best with a clear, well-lit photo of a single leaf; field photos with cluttered backgrounds may reduce accuracy.
- Outputs are model predictions and should be verified by a qualified agricultural adviser before treatment decisions.

## Tech Stack
Python · TensorFlow/Keras · NumPy · Pandas · Matplotlib · Seaborn · Streamlit · Pillow
