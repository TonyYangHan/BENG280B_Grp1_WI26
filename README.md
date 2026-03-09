# 🧠 BENG 280B Group 1: Few-Shot Stroke Lesion Segmentation
**Group members: Yang Han, Tinger Shi, Jason Chiu, Terence (Xinyuan) Luo** <br>
Welcome to the visualization and evaluation suite for the BENG 280B Winter 2026 project: **"Few-shot segmentation of 2D DWI images with Self-Supervised Learning (SSL)"**. 

This repo houses the finalized core model architecture that uses a self-supervised pretrained encoder (ViT-Base) to extract ischemic stroke lesion features in DWI scans and a trained convolutional decoder that segments the lesions. It also contains a comprehensive set of 2D and 3D visualization scripts. These tools were specifically developed to assess the spatial accuracy and noise robustness of our model.

---

## 📂 File Structure & Functionality

### 1. Core Model
* **`model.py`**: Contains the core architecture, including the pre-trained ViT-Base encoder and our custom Convolutional Decoder.
* **`data.py`**: Handles the loading and preprocessing of the ISLES 2022 multi-modal DWI/ADC dataset.
* **`train.py`**: The main file that trains the self-supervised segmentation framework.
* **`utils.py`**: Contains helper functions
* **`loss.py`**: Implements the loss functions used in model training
* **`data.py`**: Handles the loading and preprocessing of the ISLES 2022 multi-modal DWI/ADC dataset.
* **`cross_val.py`**: Conducts 5-fold cross validation for the model




### 2. Unet Baseline & Generalization Folders
* 🏗️ **`Unet/` (Baseline Model)**
  * **Purpose**: Provides the standard U-Net architecture for stroke lesion segmentation.
  * **Function**: Used to establish baseline performance metrics and generate comparison results against our proposed SSL (ViT-Base) approach.
* 🌍 **`generalization/` (PI-CAI Dataset)**
  * **Purpose**: Tests the generalization capability of our SSL model on a completely different medical imaging domain.
  * **Function**: Generates code and adapts a new decoder designed specifically to train and evaluate on the PI-CAI (Prostate Cancer) dataset.

### 3. Visualization Folder 
These scripts generate the high-quality figures and interactive 3D plots featured in our final presentation slides.

* 📊 **`vis_patient_full.py` (2D Clean Data)**
  * **Purpose**: Generates slice-by-slice 2D visual comparisons for target patients.
  * **Output**: A 1x4 subplot displaying [Original DWI, Ground Truth, AI Prediction, Probability Heatmap]. 
  * **Presentation Mapping**: Corresponds to the clean data evaluation slices.

* 🌪️ **`vis_noise_comparison_v2.py` (2D Robustness Test)**
  * **Purpose**: Evaluates the model's resilience to synthetic perturbations.
  * **Method**: Applies Gaussian blur (kernel=5, sigma=1.5) and Gaussian noise (~N(0, 0.2)) to the input slices.
  * **Output**: A perfectly aligned 2x3 matrix comparing Clean vs. Noisy predictions alongside a Difference/Error map.
  * **Presentation Mapping**: Slide 9 & Slide 57 (2D noisy vs. clean prediction).

* 🧊 **`vis_3d_mesh_v2.py` (3D Spatial Reconstruction)**
  * **Purpose**: Reconstructs 2D slice predictions into a volumetric 3D representation.
  * **Features**: Uses `skimage.measure.marching_cubes` and `plotly` to render interactive HTML files. Employs a specific color-coding strategy (Solid Blue Pred vs. Ghost Gold GT) to clearly show overlapping boundaries.
  * **Presentation Mapping**: Slide 11 & Slide 88 (3D visualization clean data).

* 🛡️ **`vis_3d_noise_compare.py` (3D Robustness Reconstruction)**
  * **Purpose**: Visualizes how noise impacts the 3D spatial coherence of the lesion prediction.
  * **Output**: An interactive side-by-side 3D split-screen comparing the reconstructed volume before and after noise injection.
  * **Presentation Mapping**: Slide 12 & Slide 85 (3D visualization clean vs. noisy data).

* 🧮 **`vis_quant_metrics.py`**
  * **Purpose**: Calculates quantitative metrics (Dice Score, IoU) across the dataset to back up our visual findings.

---

## 🛠️ How to Run

### Dependencies
Ensure you have the required libraries installed in your conda environment:
```bash
pip install torch numpy matplotlib plotly scikit-image opencv-python torchvision
```

### Execution
**Important Note:** Before running any visualization script, please ensure you update the file paths at the top of each `.py` file to match your local environment:
* `DATA_ROOT`: Directory containing the preprocessed `.npz` files.
* `CKPT_PATH`: Path to the trained model weights (`.pth` file).

Then, simply run the desired script via python:

```bash
# Example: Generate interactive 3D Meshes
python vis_3d_mesh_v2.py

# Example: Generate 2D Noise Robustness matrices
python vis_noise_comparison_v2.py
```

Output files (images and `.html` files) will be automatically saved in their respective generated folders (e.g., `vis_3d_mesh_results_labeled`, `vis_noise_robustness2`).

---
*Developed for BENG 280B (Winter 2026) at UC San Diego.*
