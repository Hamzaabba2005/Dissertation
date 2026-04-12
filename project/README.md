# Eye Disease Classification — ODIR-5K

Multi-label classification of 8 ocular diseases from fundus images using 4 deep learning models trained and evaluated on the ODIR-5K dataset

## Disease Classes
Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other

## Models
- EfficientNet-B0
- ResNet-18
- ResNet-50
- Custom CNN

## Project Structure
project/
    notebooks/
        EfficientNet-B0.ipynb
        ResNet18.ipynb
        ResNet50.ipynb
        CustomCNN.ipynb
        comparison.ipynb
    src/
        auto_detect.py
        build_samples.py
        class_weights.py
        config.py
        dataset.py
        model.py
        patient_level_split.py
        reproducibility.py
        thresholds.py
        transforms_config.py
    checkpoints/
        resnet50_outputs/
        efficientnetb0_outputs/
        resnet18_outputs/
        
    data/
    ODIR-5K/
    full_df.csv
    preprocessed_images/
    requirements.txt
    README.md

## Dataset
ODIR-5K (Ocular Disease Intelligent Recognition) — 5000 patient fundus images labelled across 8 disease categories.

Downloaded from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

Dataset in the /data folder

## Running on Kaggle (Recommended)

1. Go to kaggle.com and create a free account
2. Search for "ODIR-5K" on Kaggle datasets or go to:
   https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
3. Click "Add to your notebooks"
4. Create a new Kaggle notebook and upload the .ipynb file from the notebooks/ folder
5. Connect the ODIR-5K dataset to your notebook
6. The first cell will automatically clone the repo to get the source modules
7. Set `KAGGLE = True` (this is the default)
8. Run all cells — free GPU is available on Kaggle

## Running Locally
1. Install dependencies: pip install -r requirements.txt
2. The dataset is included in the `data/` folder — no download needed
3. Set `KAGGLE = False` in the notebook
4. If using VSCode, add to `.vscode/settings.json`:
```json
   {
       "python.analysis.extraPaths": ["../project/src"]
   }
```
5. Run the notebooks in this order:
   - EfficientNet-B0.ipynb
   - ResNet18.ipynb
   - ResNet50.ipynb
   - CustomCNN.ipynb
   - comparison.ipynb (run after all 4 models are trained)

Note: training will be significantly slower without a GPU

## Data Split
- 70% Training
- 15% Validation
- 15% Test

Patient-level split used throughout to prevent data leakage between sets.

## Dependencies
torch
torchvision
numpy
pandas
matplotlib
seaborn
Pillow
scikit-learn

## AI Usage
Claude was used to assist with debugging. Prompts and outputs are documented in the appendix of the written report.

## OneDrive
