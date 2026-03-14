# Emotional Classification using Audio Data

Undergraduate Research Opportunities Programme (UROP)

---

## What this project does

This project classifies human speech audio into one of six emotions using deep learning. Two approaches are implemented and compared:

1. MFCC + Delta + Delta-Delta features with a CNN classifier
2. MFCC + Pitch + Energy features with a CNN classifier

Emotions: angry, calm, fearful, happy, neutral, sad

---

## Dataset

RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song
Livingstone and Russo (2018), PLOS ONE. https://doi.org/10.1371/journal.pone.0196391

- 24 actors (12 male, 12 female)
- Around 2068 audio samples across 6 emotion classes
- Files are organised into per-emotion folders inside dataset/

The dataset folder is not included in this repo because of file size. Download RAVDESS from https://drive.google.com/drive/folders/1kAZr-x6Bo7ZjcYIoZaiV0WriIERQCxnj?usp=drive_link and organise the files into emotion subfolders under dataset/.

---

## Notebooks

### MFCC_delta.ipynb
Uses MFCC (40 coefficients) combined with delta and delta-delta derivatives as a 3-channel input to a CNN. Delta features capture how the MFCCs change over time which helps distinguish emotions with similar static spectra but different dynamics.

### MFCC_pitch_energy.ipynb
Uses MFCC (40 coefficients) alongside pitch (fundamental frequency F0) and RMS energy as additional feature rows. Pitch variation is strongly linked to emotional expression and energy distinguishes high intensity emotions from low intensity ones.

---

## Method

- Augmentation: gaussian noise, time stretching, pitch shifting applied to training data only (4x samples)
- Model: 4-block CNN with batch normalisation and dropout
- Train/test split: 80% training, 20% testing (stratified)
- Optimiser: Adam with ReduceLROnPlateau and EarlyStopping

---

## Results

| Notebook | Features | Test Accuracy |
|----------|----------|---------------|
| MFCC_delta | MFCC + Delta + Delta-Delta | 92.78% |
| MFCC_pitch_energy | MFCC + Pitch + Energy | 89.94%  |

Charts and confusion matrices are in the outputs/ folder.

---

## How to run

1. Clone the repo
2. Install dependencies
3. Download RAVDESS and set up the dataset folder
4. Open either notebook in Google Colab
5. Enable T4 GPU under Runtime -> Change runtime type
6. Run all cells top to bottom

---

## Dependencies
```
pip install librosa tensorflow scikit-learn seaborn matplotlib numpy soundfile
```

---

## References

Livingstone S, Russo F (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLOS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

---

## Authors

Yogesh R Mehta
Shreyas Gupta
