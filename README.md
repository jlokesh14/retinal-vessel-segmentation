# Retinal Vessel Segmentation

## Dataset
DRIVE retinal fundus dataset.

## Preprocessing
- Green-channel extraction
- CLAHE enhancement
- FOV masking
- Normalization
- 64x64 patch generation

## Model
U-Net for binary retinal vessel segmentation.

## Training
- Training images: 18
- Validation images: 2
- Epochs: 30
- Best checkpoint: epoch 24
- Device: CPU

## Results
Best validation Dice: 0.7358
Best validation IoU: 0.6065

Training-set mean Dice: 0.8260
Training-set mean IoU: 0.7043

## Test Predictions
20 DRIVE test images were processed successfully.

Mean predicted vessel coverage: 8.88%.

## Project Structure
- src/ - source code
- data/ - datasets and processed data
- checkpoints/ - trained model
- outputs/ - generated predictions
- results/ - final result files
- requirements.txt - Python dependencies

## Reproducibility
Install dependencies with:

    pip install -r requirements.txt

Train:

    python src/train.py

Evaluate:

    python src/evaluate.py

Generate DRIVE test predictions:

    python src/predict_test.py
