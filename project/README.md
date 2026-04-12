# Eye Disease Classification — ODIR-5K

Multi-label classification of 8 ocular diseases from fundus images using 4 deep learning models trained and evaluated on the ODIR-5K dataset

## Disease Classes
Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other

## Models
- EfficientNet-B0
- ResNet-18
- ResNet-50
- Custom CNN

## Dataset
ODIR-5K (Ocular Disease Intelligent Recognition) — 5000 patient fundus images labelled across 8 disease categories.

Downloaded from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

Dataset in the /data folder

## Folder Structure
project/
notebooks/
EfficientNet-B0.ipynb
ResNet18.ipynb
ResNet50.ipynb
CustomCNN.ipynb
comparison.ipynb
project/
src/
config.py
reproducibility.py
auto_detect.py
build_samples.py
patient_level_split.py
transforms_config.py
dataset.py
class_weights.py
model.py
thresholds.py
checkpoints/
efficientnet_b0_outputs/
resnet18_outputs/
resnet50_outputs/
custom_cnn_outputs/
data/
ODIR-5K
full_df.csv
preprocessed_images/
requirements.txt
README.md

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
9. Once training is complete, run the Export Outputs cell at the bottom of the notebook
10. Go to the Output panel on the right side of Kaggle
11. Download the zip file for that model
12. Extract the zip and copy the _eval_data.pkl file into your local checkpoints/ folder
13. Repeat for each model, then run comparison.ipynb locally

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

## GitHub Link
https://github.com/Hamzaabba2005/Dissertation