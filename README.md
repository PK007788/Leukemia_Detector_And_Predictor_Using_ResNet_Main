# Leukemia_Detector_And_Predictor_Using_ResNet_Main

This project implements a deep learning–based system for detecting leukemic cells in peripheral blood smear images using a fine-tuned ResNet-50 Convolutional Neural Network (CNN).
The system performs segmentation, classification, annotation, and prediction confidence analysis, and is deployed using a Flask backend and Streamlit frontend.

1. Overview

Acute Lymphoblastic Leukemia (ALL) diagnosis involves identifying abnormal blast cells in a blood smear. Manual examination is time-consuming and subjective, so automated methods support faster screening.

This project builds an automated pipeline that:

Accepts a blood smear image (JPG, PNG, BMP).

Segments out individual cells using classical image processing.

Classifies each cell as:

Leukemia

Normal

Draws bounding boxes and produces an annotated output image.

Displays prediction probabilities for each detected cell.

The model achieves strong performance on the C-NMC 2019 dataset with a cleaned and balanced subset.

2. How Classification Works
2.1 Cell Segmentation

Cells are located using:

Grayscale conversion

Gaussian blurring

Otsu thresholding

Contour detection

Bounding-box extraction

Each detected cell is cropped and normalized before being passed to the CNN.

2.2 Deep Learning Model (ResNet-50)

The model uses ResNet-50, a deep residual neural network pre-trained on ImageNet, then fine-tuned on labeled leukemia cell images.

The network learns discriminative patterns such as:

Nuclear shape: Leukemic blasts often have a larger, irregular nucleus.

Chromatin texture: Leukemic cells show finer, more diffused chromatin.

Nucleus-to-cytoplasm ratio: Leukemic cells typically have a higher ratio.

Cytoplasmic staining: Variations in color intensity correlate with malignancy.

Cell boundary morphology: Blast cells usually have smoother contours.

The model does not “see” medical meaning directly, but learns statistical image features representing these characteristics through convolutional filters.

2.3 Final Classification

For each cropped cell patch, the model outputs:

probability_leukemia
probability_normal


The larger probability determines the final predicted label.

3. System Architecture
Backend (Flask)

Loads the trained .keras model.

Performs segmentation, preprocessing, prediction.

Generates annotated images with bounding boxes.

Returns structured JSON responses to the frontend.

Logs predictions (optional) into CSV for analysis.

Frontend (Streamlit)

Upload interface supporting up to 5 images.

Displays segmented crops and model predictions.

Shows annotated output images with bounding boxes.

Presents class probabilities and prediction summaries.

4. Folder Structure
Leukemia_Detector_And_Predictor_Using_ResNet_Main/
│
├── LDwebapp/
│   ├── backend/
│   │   ├── app.py
│   │   ├── leukemia_model.keras (not stored on GitHub)
│   │   ├── utils.py
│   │   └── requirements.txt
│   │
│   └── frontend/
│       ├── streamlit_app.py
│       ├── sample_input.jpg
│       └── requirements.txt
│
├── leukemia/                # Training notebooks, dataset setup
├── .gitattributes           # Git LFS settings
├── README.md
└── ...


Model files are intentionally excluded or kept in Git LFS due to GitHub size limitations.

5. How to Run Locally
Backend
cd LDwebapp/backend
pip install -r requirements.txt
python app.py




Frontend

In a new terminal:

cd LDwebapp/frontend
pip install -r requirements.txt
streamlit run streamlit_app.py



6. Dataset

This work is based on the C-NMC 2019 dataset for ALL cell classification.
Images were cleaned, relabeled where necessary, and split into train/validation/test sets.

7. Future Improvements

Use a learned segmentation model (U-Net / Mask R-CNN).

Implement multi-class classification (e.g., mature lymphocyte vs. blast).

Deploy as a standalone web application with cloud storage.

Integrate Grad-CAM to show what regions influence predictions.

8. Disclaimer

This tool is intended for educational and research purposes only.
It is not a clinical diagnostic device and should not replace professional medical evaluation.
