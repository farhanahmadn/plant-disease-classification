# 🍃 Plant Disease Classification: EfficientNetB0 vs ResNet50

A comparative Deep Learning study evaluating **EfficientNetB0** and **ResNet50** architectures for plant disease classification. This project implements Transfer Learning, Fine-Tuning, and uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visually interpret the models' decision-making processes.

## 🚀 Tech Stack
* **Framework:** PyTorch & Torchvision
* **Models:** EfficientNetB0, ResNet50 (Pre-trained on ImageNet)
* **Interpretability:** `pytorch-grad-cam`
* **Data Processing:** Scikit-Learn, NumPy, Matplotlib

## 💡 Key Features
1. **Dynamic Data Pipeline:** Automatically filters and subsets 10 specific tomato and potato classes from the raw dataset.
2. **Robust Augmentation:** Implements `RandomHorizontalFlip`, `RandomRotation`, and `ColorJitter` to mitigate overfitting.
3. **Transfer Learning Strategy:** Two-phase training:
   - **Phase 1:** Training a custom MLP head with frozen base layers.
   - **Phase 2:** Fine-tuning by unfreezing the last 20 layers with a reduced learning rate.
4. **Model Explainability:** Integrates Grad-CAM to generate heatmaps highlighting the diseased regions on leaves, proving the model learns biological features, not just background noise.

## 📂 Dataset Information
This project uses a 10-class subset of the **PlantVillage Dataset**. 
> **Note:** Due to the large size of the dataset (approx. 54,000 images), the raw image files are **not** included in this repository to keep the repository lightweight and efficient.

**How to get the data:**
1. Download the dataset from [Kaggle: PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Extract the archive and place the `PlantVillage` folder inside the `data/` directory of this project so the structure looks like `data/PlantVillage/...`.

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/farhanahmadn/plant-disease-classification.git](https://github.com/farhanahmadn/plant-disease-classification.git)
   cd plant-disease-classification
