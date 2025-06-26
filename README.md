# Fair Melanoma Detection

This project implements a deep learning pipeline for detecting melanoma in dermoscopic images, with a strong emphasis on achieving *fair and robust performance across diverse skin tones*. Developed for the LUMEN Data Science 2024/25 competition.

## Core Idea & Pipeline

Our solution uses a *two-stage approach*:
1. **Segmentation:** A U-Net model (ResNet34 backbone) first localizes the skin lesion within the image. This helps the subsequent classifier focus on the most relevant area.
2. **Classification:** An EfficientNet-B4 based model then classifies the segmented lesion as malignant (melanoma) or benign.

A custom *skin tone classification* module (based on background image brightness) is integrated to enable fairness evaluation and targeted data balancing.

![Main pipeline](assets/pipeline_main.jpg)

## Key Features & Techniques

*   **Fairness Focus:**
    *   Utilized ISIC (2016-2020) and Fitzpatrick17k datasets
    *   Fitzpatrick17k data (127 images of darkest skin tones) was incorporated to improve representation
    *   Targeted upsampling for underrepresented skin tones (especially the darkest category) and malignant cases
    *   GroupClassBalancedSampler used in training for equitable exposure to (skin_tone, target) groups
    *   Performance and fairness metrics (accuracy, recall, precision, F1, selection rate) monitored per skin tone using fairlearn.MetricFrame
*   **Robust Preprocessing:**
    *   Duplicate image removal
    *   Lesion cropping via the segmentation model
    *   Hair removal using inpainting
    *   Standardized 512x512 image resizing (padding for lower-res Fitzpatrick images)
    *   Preprocessing results are cached for efficiency
*   **Advanced Classification Model Training:**
    *   EfficientNet-B4 (pretrained, e.g., Noisy Student via TIMM)
    *   Focal Loss for class imbalance
    *   AdamW optimizer & OneCycleLR scheduler
    *   Progressive layer unfreezing
    *   Mixup augmentation
    *   Automatic Mixed Precision (AMP)
    *   Dynamic threshold optimization on validation data

## Datasets & Balancing

*   **Sources:** ISIC 2016, 2017, 2019, 2020, and 127 images from Fitzpatrick17k (all categorized as darkest skin tone)
*   **Initial State:** Significant class imbalance (more benign) and underrepresentation of dark skin tones
    *   Example *Train (before balancing)*: 35,864 samples | 22.10% Malignant
*   **After Balancing (Training Set):** Through upsampling of malignant cases and dark skin tone samples, and downsampling of some benign groups:
    *   Example *Train (after balancing)*: 46,178 samples | 37.86% Malignant
    *   The proportion of darker skin tones in the training set was significantly increased
*   **Validation Set:** Maintained a distribution similar to the original data to reflect real-world scenarios
    *   Example 'Validation': 8,974 samples | 22.52% Malignant

![Skin tone distribution](assets/skin_tone_distribution.jpg)

## Performance Highlights (Validation Set)

*   **Overall Accuracy: ~93.8%**
*   **Weighted F1 Score: ~0.94**
*   **Malignant Recall (Sensitivity): 89%**
*   **Malignant F1 Score: 87%**
*   **Optimal Decision Threshold: 0.4242**

**Fairness:**
The model showed relatively consistent accuracy, recall, and precision across skin tones (Max-Min disparities ~5-8%). The selection rate disparity was larger (~35%), primarily reflecting the differing true prevalence of malignancy in the validation set's skin tone groups. The darkest skin tone category (4) achieved the highest recall (94.6%) and precision (91.4%).

![Results](assets/results.jpg)


## Getting Started

### 1. Environment Setup

*   **Hardware:**
    *   CPU: AMD Ryzen Threadripper 3990X (or similar multi-core processor)
    *   GPU: 2 x NVIDIA GeForce RTX 3090 (24GB VRAM each) or similar CUDA-enabled GPU(s)
    *   RAM: 256 GB
*   **Software:**
    *   OS: Ubuntu 22.04.4 LTS / Windows 10/11
    *   NVIDIA Driver: Version 550.120 (or newer compatible version)
    *   CUDA: Version 12.4 (or newer compatible version)
    *   Python: v3.11.7
*   **Instructions:**

    *For Linux/macOS:*
    ```bash
    # Navigate to your project directory
    # Create a Python virtual environment
    python3 -m venv venv

    # Activate the environment
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

    *For Windows:*
    ```bash
    # Navigate to your project directory in PowerShell or Command Prompt
    # Create a Python virtual environment
    python -m venv venv

    # Activate the environment
    .\venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```
    
### 2. Data Preparation

To reproduce our results, you will need to download and organize the datasets as follows. All data should be placed within a data/ directory at the root of the project.

**Expected Directory Structure (inside data/):**
```
data/
├── 2016_train/                     # ISIC 2016 Images
├── 2017_train/                     # ISIC 2017 Images
├── 2019_train/                     # ISIC 2019 Images
├── 2020_train/                     # ISIC 2020 Images (JPEG format)
├── masks/                          # ISIC 2017 & 2018 Segmentation Masks
├── 2016_Metadata.csv
├── 2017_Metadata.csv
├── 2018_Metadata.csv               # Note: ISIC 2018 images are not used due to duplicates
├── 2019_Metadata.csv
├── 2020_Metadata.csv
├── Fitzpatrick_Metadata.csv
└── duplicate_images.txt
```

### Download Instructions:

1.  **ISIC Dermoscopic Images:**
    *   [*2016 Images:*](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip) Extract the images into: **data/2016_train/**
    *   [*2017 Images:*](https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip) Extract the images into: **data/2017_train/**
    *   [*2019 Images:*](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip) Extract the images into: **data/2019_train/**
    *   [*2020 Images (JPEG):*](https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip) Extract the images into: **data/2020_train/**
**Note:** ISIC 2018 images are not downloaded as they are largely duplicates found in other years and are removed during our deduplication process

2.  **Metadata Files:**
    *   [*ISIC 2016 Metadata:*](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv) Download and save as **data/2016_Metadata.csv**
    *   [*ISIC 2017 Metadata:*](https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv) Download and save as **data/2017_Metadata.csv**
    *   [*ISIC 2018 Metadata:*](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip) Download, extract the CSV, and save as **data/2018_Metadata.csv**
    *   [*ISIC 2019 Metadata:*](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv) Download and save as **data/2019_Metadata.csv**
    *   [*ISIC 2020 Metadata:*](https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv) Download and save as **data/2020_Metadata.csv**
    *   [*Fitzpatrick17k Metadata & Images:*](https://github.com/mattgroh/fitzpatrick17k/blob/main/fitzpatrick17k.csv) Save it as **data/Fitzpatrick_Metadata.csv**. The script is designed to download the images referenced in this CSV automatically during preprocessing if they are not found locally

3.  **Segmentation Masks (ISIC 2017 & 2018):**
    *   [*2017 Masks:*](https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip) Extract the masks into **data/masks/**
    *   [*2018 Masks:*](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip) Extract the masks into **data/masks/**
**Note:** There might be ~1800 images with duplicate names when extracting masks. If prompted, choose to "skip these files" as they are duplicates.

4.  **Duplicate Images List:**
    *   Download the [06 - all_train_duplicates_deleted_(all but newest).txt](https://github.com/mmu-dermatology-research/isic_duplicate_removal_strategy/blob/main/file_lists/06%20-%20all_train_duplicates_deleted_(all%20but%20newest).txt) file
    *   Save this file as **data/duplicate_images.txt**. This file is used to remove duplicate entries during the metadata cleaning process.

**Important:** After downloading, ensure your **data/** directory matches the "Expected Directory Structure" shown above. Paths and filenames are critical for the scripts to run correctly. Any deviations might require adjustments in **config/config.py**.


### 3. Training the Model
To run the full pipeline (segmentation model training + classification model training):
```bash
python train.py
```

Checkpoints are saved in **checkpoints/**. Training parameters can be adjusted in **config/config.py**.

### 4. Making Predictions
To generate predictions on a new set of images:
```bash
python predict.py <INPUT_IMAGE_FOLDER> <OUTPUT_CSV_FILE>
```

Example:
```bash
python predict.py data/my_test_images/ my_predictions.csv
```

## Project Structure
A modular structure is used:
*   **checkpoints/**: Trained model weights
*   **classification_model/** & **segmentation_model/**: Code for respective models (datasets, architectures, training logic, etc.)
*   **common/**: Shared utilities, constants, visualization
*   **config/**: Configuration files
*   **data/**: Raw and preprocessed data
*   **pipeline/**: Preprocessing steps (skin tone, lesion cropping, hair removal)
*   **utils/**: Helper functions (data sampling, thresholding)
*   root: **train.py**, **predict.py**, **requirements.txt**

# Other collaborators

Special thanks to everyone who contributed during the competition:

- [Lovro Barić](https://github.com/LoVrO312)
- [Lorena Đaković](https://github.com/lorka1)
- [Lara Slišković](https://github.com/lsliskov)
