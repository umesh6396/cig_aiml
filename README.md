Distorted Visual Sequence Pattern Recognition using Deep Learning
CIG AI/ML Challenge — CAPTCHA Text Recognition

Problem Statement
Given 200×100 grayscale CAPTCHA images each containing 6 distorted alphanumeric characters, build a deep learning model that predicts the correct character sequence.

Training set: 20,000 labeled images
Test set: 5,000 images (no labels)
Evaluation metric: Character Error Rate (CER) — lower is better


My Approach — CRNN + CTC
I used a Convolutional Recurrent Neural Network (CRNN) with CTC (Connectionist Temporal Classification) loss — the industry-standard architecture for scene text and CAPTCHA recognition.
Input Image (1 × 64 × 256, grayscale)
         ↓
  CNN Backbone  ──  5-stage conv with Residual blocks
         ↓          extracts: [B, 512, 1, 64]
  Reshape to Sequence  →  [64, B, 512]
         ↓
  Bidirectional LSTM × 2 layers (256 hidden units each direction)
         ↓
  Fully Connected Layer  →  [64, B, num_classes]
         ↓
  CTC Greedy Decode  →  6-character prediction string
Why CRNN + CTC?

CNN extracts robust spatial features even under heavy distortion, blur, and noise
BiLSTM captures sequential context across the character sequence
CTC loss handles sequence learning without needing explicit character segmentation labels
End-to-end trainable — the model learns directly from distorted image → text pairs

Why Traditional OCR (Tesseract) Failed
CAPTCHAs are specifically designed to defeat rule-based OCR:

Overlapping character boundaries make segmentation impossible
Noise bridges span multiple characters
Affine warps distort glyph shapes
Background clutter is indistinguishable from foreground strokes

Testing Tesseract with 16 preprocessing combinations gave 0% accuracy. Template matching gave ~4% (random baseline). The CRNN achieves ~87% sequence accuracy.

Results
MetricValueValidation CER0.073Validation Sequence Accuracy87.1%Average Character Accuracy94.0%Training Time (T4 GPU, Colab)~10–12 minutes

Model Architecture Details
CNN Backbone
StageOperationOutput ShapeInputGrayscale + Resize1 × 64 × 256Stage 1Conv(1→64) + MaxPool(2×2)64 × 32 × 128Stage 2Conv(64→128) + MaxPool(2×2)128 × 16 × 64Stage 3Conv(128→256) + ResBlock + MaxPool(2×1)256 × 8 × 64Stage 4Conv(256→512) + ResBlock + MaxPool(2×1)512 × 4 × 64Stage 5Conv(512→512) + AdaptiveAvgPool512 × 1 × 64
Sequence Model

BiLSTM: 2 layers, 256 hidden units per direction (512 total)
Dropout: 0.3 between LSTM layers
Output: FC layer → log-softmax over 40 classes (including CTC blank)

Training Configuration
ParameterValueOptimizerAdamWLearning rate3e-4 (OneCycleLR)Batch size64Epochs50Weight decay1e-4Grad clip5.0
Data Augmentation (Training Only)

Random Gaussian blur (σ = 0.1–1.5, p=0.4)
Random affine transform (±5°, translate 5%, scale 90–110%, p=0.5)
Color jitter (brightness & contrast ±0.3)
Random erasing (p=0.2) — simulates occlusion

Inference — Test-Time Augmentation (TTA)
Each test image is predicted 7 times with slight augmentation variations. The final prediction is a majority vote per character position, which improves CER by ~0.01 over single-pass inference.

Repository Structure
├── captcha_crnn_solution.ipynb   # Complete solution notebook (run on Colab)
├── submission.csv                # Predicted labels for 5000 test images
└── README.md                     # This file

How to Run
On Google Colab (Recommended)

Open Google Colab and enable T4 GPU (Runtime → Change runtime type → T4)
Upload captcha_crnn_solution.ipynb
Upload cig_ps.zip to /content/
Run all cells — training takes ~10–12 minutes
The final cell auto-downloads your submission.csv

Requirements
torch
torchvision
Pillow
pandas
numpy
matplotlib
tqdm
scikit-learn

Character Set
The dataset uses 33 core characters — uppercase letters excluding I, L, O (to avoid visual confusion) plus digits 0–9:
0 1 2 3 4 5 6 7 8 9
A B C D E F G H J K M N P Q R S T U V W X Y Z

Key Observations from EDA

All labels are exactly 6 characters (2 outliers with Excel-garbled values excluded)
Near-uniform character distribution — balanced classification problem
Images have heavy noise floor (mean pixel ~173/255)
Characters appear in roughly fixed horizontal positions (~33px per character slot)
Each image has unique random distortions — no two images share the same noise pattern


Future Improvements
ImprovementExpected GainAttention decoder instead of CTC+2–4% SeqAccTransformer encoder (ViT-style columns)+3–5% SeqAccBeam search CTC decoding (width=5)+0.5–1% SeqAccTrain on 100% data (no val split)+1–2% SeqAccLabel smoothing (ε=0.1)+0.5% SeqAccMore TTA views (N=15)+0.5% SeqAcc

References

Shi et al. (2015) — An End-to-End Trainable Neural Network for Image-based Sequence Recognition (original CRNN paper)
Graves et al. (2006) — Connectionist Temporal Classification (CTC loss)
