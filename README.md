# Leukemia_Detector_And_Predictor_Using_ResNet_Main

## Automated Blood Cell Classification Using Deep Learning (ResNet–50)

A lightweight, end-to-end system for detecting and classifying leukemic cells from microscopic blood smear images.
This project includes:

-Deep learning model (ResNet-50 backbone)

-Automated cell segmentation using OpenCV

-Flask backend (model inference + preprocessing + segmentation)

-Streamlit frontend (user interface + visualization)

-Logging of predictions and annotated outputs

## 1. Project Overview

Early detection of leukemia can significantly improve treatment outcomes.
This system classifies blood cells into:

-Leukemic

-Normal

## The workflow includes:

-Image Upload (JPG/PNG/BMP)

-Automated Nucleus Segmentation

-Bounding-Box Extraction

-ResNet-based Classification

-Probability Scores + Annotated Image Output

-Prediction Logging for Auditability

## 2. How the Model Performs Classification

The model doesn’t look at the entire image at once. Instead, it:

--Segments the nucleus or dominant WBC region
--Using thresholding + contour detection.

--Extracts the region of interest (ROI)
--This focuses the model on the actual cell, not the background.

--Resizes and normalizes the ROI
  (224×224, scaled between 0–1)

--Uses a fine-tuned ResNet-50 to classify the cell.

### What ResNet Learns in Leukemia vs Normal Cells

#### The model focuses on:

-Feature	Leukemic Cells	Normal Cells
-Nucleus Shape	Often irregular, larger, more variable	More uniform and rounded
-Chromatin Texture	Dense, clumped	Smooth and consistent
-Color/Staining Pattern	Darker nucleus, uneven staining	Evenly stained
-Cell Boundary	May appear distorted	Clear, defined boundary

### ResNet automatically extracts multi-level features from these patterns across 50 layers.

## 3. Performance Metrics
#### Classification Report (Test Set)
#### Class	Precision	Recall	F1-Score	Support
#### Leukemia	0.83	0.90	0.87	1092
#### Normal	0.74	0.61	0.67	509
#### Accuracy			0.81	1601
## 4. Interpreting These Results

#### The model is highly sensitive to leukemia cells (recall 0.90).
#### Meaning: If a cell is leukemic, there is a 90% chance the model will catch it.

#### Normal cells have lower recall (0.61), meaning:
#### Some normal cells are misclassified as leukemic.

### This is common when:

#### The dataset is imbalanced (more leukemia cells than normal)

#### Normal cell features vary mildly across labs and staining methods

#### Still, 81% overall accuracy indicates the model generalizes reasonably well.

## 5. Important Note on Dataset & Hardware Limitations

### This project was built under:

#### Dataset Limitations.

#### Limited number of normal cell samples.

#### Some samples had inconsistent staining / imaging quality.

#### No professionally annotated bounding boxes.

#### All segmentation was hand-designed (not a learned segmentation model)

#### These factors naturally limit the maximum achievable accuracy.

### Device Limitations:

#### The model was trained on limited hardware, restricting:

#### Total training time.

#### Batch size.

#### Ability to use more complex architectures.

#### Ability to perform extensive hyperparameter tuning.

#### Ability to train with large augmentations.

## Future Improvements

-With:

--Larger balanced datasets

--Better quality microscopic images

--Modern architectures (EfficientNet-V2, ViT, ConvNeXt)

### --GPU access for deeper fine-tuning

--Learned segmentation models (U-Net, Mask R-CNN)

--Accuracy can be pushed well above 90%.

### This project lays the foundation — the system is fully functional and can grow as better data becomes available.

## 6. System Architecture
### User → Streamlit UI → Flask Backend → Segmentation → ResNet Model → Prediction → UI Visualization

## Frontend (Streamlit)

--Upload up to 5 images

## Displays:

--Annotated images

--Cropped cells

--Prediction labels

--Probability scores

--Backend (Flask)

--Accepts images

--Performs segmentation + preprocessing

--Runs inference

## Saves:

--Annotated outputs

--Cropped cells

## CSV logs

### 7. How to Run
#### Backend
--cd backend
--pip install -r requirements.txt
--python app.py

#### Frontend
--cd frontend
--pip install -r requirements.txt
--streamlit run streamlit_app.py

```8. Folder Structure
project/
│
├── LDwebapp/
│   ├── backend/
│   │   ├── app.py
│   │   ├── leukemia_model.keras
│   │   ├── utils.py
│   │   ├── requirements.txt
│   │   └── static/saved/
│   │
│   └── frontend/
│       ├── streamlit_app.py
│       ├── sample_input.jpg
│       └── requirements.txt
│
└── README.md```

# 9. Disclaimer

# This tool is created strictly for educational and research purposes.
# It is NOT a clinical diagnostic tool and should not be used as a medical decision-making system.
