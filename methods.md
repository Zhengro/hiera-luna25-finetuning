This document outlines the top three methods on the [Closed Testing Phase Leaderboard](https://luna25.grand-challenge.org/evaluation/closed-testing-phase/leaderboard/). It aims to accurately interpret the technical details based on the provided methodology paper, though some inaccuracies or misunderstandings may remain. Please treat this as a starting point and consult the original methodology paper as well as the upcoming challenge summary paper for the definitive reference.

- [Algorithm `v1` (Score: 0.7807)](#algorithm-v1-score-07807)
- [Algorithm `Pulse-3D` (Score: 0.7466)](#algorithm-pulse-3d-score-07466)
- [Algorithm `NoduleMC` (Score: 0.7448)](#algorithm-nodulemc-score-07448)

# Algorithm `v1` (Score: 0.7807)

## Data Processing

### Datasets
Primary Dataset: LUNA25 challenge dataset

Additional Datasets:
- LIDC Dataset (TCIA: https://www.cancerimagingarchive.net/collection/lidc-idri/)
- Task06_Lung Dataset (Medical Segmentation Decathlon)

Annotation Details:
- LIDC has 9 Likert scale categories: subtlety, internal structure, calcification, sphericity, margin, lobulation, spiculation, texture, malignancy
- LIDC annotated by 4 readers
- LUNA25 labels refer to biopsy results, while LIDC labels refer to visual inspection only

### Preprocessing
1. Nodule crops extracted at 128×128 pixels in-plane and 64 slices (both in original resolution)
2. Resized to isotropic spacing of 0.75mm
3. Symmetrically cropped and padded to 160×160×160 voxels
4. Windowing (HU Clipping):
    - Lung Window (used in segmentation & malignancy estimation phase): Applied to optimize visualization of lung tissue
    - Soft Tissue Window (used in malignancy estimation phase): Applied to optimize visualization of soft tissues
5. Patch Extraction:

    For each nodule, extract a 3D patch (80×80×80 voxels) centered on the nodule
    - During training:
        - Spatial Augmentation (p=0.5 for each):
            - Rotation: -45 to 45 degrees
            - Zooming: 0.7 to 1.4 factor
            - Random flip over all three axes
        - Gray Value Augmentation (p=0.5 for each):
            - Contrast adjustment
            - Mean and variance shift
            - Blurring
            - Gaussian noise addition
    - During inference: 
        - Sliding window (used in segmentation phase): Input CT volume processed with overlapping windows (50% overlap)
        - Gaussian center weight (used in segmentation phase)
        - Random flip over all three axes (used in segmentation & malignancy estimation phase)

## Model Architecture

### Segmentation Component (U-Net)
Input:
- First channel: Preprocessed 3D patch from lung window
- Second channel: 3D Gaussian spatial prior (SD=20mm) centered at the crop center

Architecture:
- U-Net with 32 input filters and 5 stages

Training Output: 
- 2-channel probability map (80×80×80 voxels) with softmax activation (background and nodule probabilities)

Inference Output:
- Ensembling: 
    - For each sliding window position, the 3 U-Net models (from 3-fold CV) generate 2-channel probability maps. Averaged probability map computed from the 3 models
- Post-processing:
    - argmax applied to the averaged 2-channel map to produce a binary mask (80×80×80 voxels)
    - connected component analysis:
        - All connected regions in the mask are identified
        - Only the component containing the center of the crop is retained

### Malignancy Estimation Component
Input:
- First channel: Preprocessed 3D patch from lung window
- Second channel: Preprocessed 3D patch from soft tissue window
- Third channel: Binary segmentation mask

Architecture:
1. 3D ResNet Variants (19 total, varying loss weights "w" from 0 to 1):
    - Base architecture includes 32 filters in first block, 4 stages with 1, 1, 3, and 2 blocks per stage
    - For each variant, 3 models trained via 3-fold cross-validation (57 ResNet models total)

2. Vision Transformer Variants (19 total, varying loss weights "w" from 0 to 1):
    - Base architecture includes initial stride 8×8×8 voxels, embedding dimension d=256, and 4 layers with 4 heads each
    - For each variant, 3 models trained via 3-fold cross-validation (57 ViT models total)

Pre-training Output:
- Nine values for each of the categories

Training Output:
- One value with sigmoid activation

Inference Output:
- Ensembling
    1. Within-Architecture Averaging

        For each of the 38 variants, average the predictions from its 3 cross-validation models:
        ```
        architecture_prediction_i = (model_i_fold1 + model_i_fold2 + model_i_fold3) / 3
        ```
        Result: 38 intermediate predictions (one per architecture)

    2. Cross-Architecture Logistic Regression

        Apply learned logistic regression model to combine the 38 intermediate predictions:
        ```
        final_score = w₁·p₁ + w₂·p₂ + ... + w₃₈·p₃₈ + b
        final_probability = sigmoid(final_score)
        ```
        Weights (w₁...w₃₈) and bias (b) optimized on a separate validation set

## Training Recipe

### Common 
- Optimizer: SGD with Nesterov's momentum (momentum=0.99)
- Batch Size: 2
- Learning Rate Schedule (piecewise linear):
    - Warm-up 0-2.5%: 0.01 to 0.1
    - Warm-up 2.5-5%: 0.1 to 1.0
    - Decay 5-85%: 1.0 to 0.1
    - Decay 85-100%: 0.1 to 0.01
- Device: RTX 6000 Ada GPU

### Specific
- Segmentation Component:
    - 3-fold cross-validation: Used to train 3 distinct U-Net models (one per fold), which were ensembled during inference
    - Training (on LIDC and Task06_Lung datasets)
        - Loss Function: DSC (Dice Similarity Coefficient) + cross-entropy
        - Training Duration: 250,000 batches per fold
- Malignancy Estimation Componnet:
    - 3-fold cross-validation: All 38 architectures (19 ResNet + 19 ViT) trained independently. Each architecture trained with 3-fold cross-validation (3 models per architecture)
    - Pre-training (on LIDC)
        - Random Sample Selection
            - Biased selection for the second sample based on category and Likert scale, etc
            - Ensured rare cases (e.g., heavily calcified nodules) were seen sufficiently
        - Learning Rate: Max 0.0025
        - Loss Function: Cross-entropy (logistic regression style)
        - Training Duration: 3,750,000 batches
    - Continued Training (on LUNA25)
        - Sample Selection
            - Each batch contains one positive and one negative sample
        - Learning Rate: Max 0.000625
        - Loss Function: Cross-entropy with weighted positive samples
            - Positive sample weight: "w" (19 uniform values between 0 and 1)
            - Negative sample weight: "1-w"
        - Training Duration: 62,500 batches

# Algorithm `Pulse-3D` (Score: 0.7466)

## Data Processing

### Datasets
Only LUNA25 challenge dataset

### Preprocessing
1. Windowing (HU Clipping): HU values are clipped to [-1000, 600] and linearly normalized to [0, 1]
2. Patch Extraction: For each nodule, extract a 3D patch (64x64x64 voxels) centered on the nodule
3. Patch Augmentation (using MONAI):
    - Gaussian noise, smoothing
    - Intensity scaling and shifting
    - Random zoom, rotation (±10°), and flipping
    - Histogram shift and bias field distortions

## Model Architecture
Pulse3D is a hybrid deep learning architecture that combines 3D convolutional neural networks (CNNs) with transformer-based attention layers. Local texture is captured by the CNN, while global morphology is modeled by the Transformer.

### Backbone

Input: single-channel 3D patch

Architecture: a modified `r3d_18` network (a 3D CNN) from torchvision
- The first convolution is adapted to handle single-channel CT volumes by averaging pretrained RGB filters.
- The final classification head is removed.

Output: spatiotemporal feature maps

### Transformer Head

Input:
- the backbone output is flattened into a token sequence
- a learnable "[CLS]" token is prepended to that sequence
- learnable 3D positional embeddings are added to yield the final input sequence

Architecture: a stack of 6 Transformer encoder blocks. Each block includes
- Pre-norm multi-head self-attention (8 heads)
- Feedforward layers with GELU activation
- LayerScale residual connections
- DropPath regularization

Output: token sequence

### Classification Head

Input: the "[CLS]" token from the transformer output

Architecture: an MLP consisting of
- LayerNorm
- Two linear layers with GELU
- Dropout (0.1)

Output: logit

## Training Recipe

- Phases:
    - Phase 1: feature learning
    - Phase 2: fine-tuning
- Optimizer: AdamW
    - Learning rate: 1 × 10−4 (Phase 1); 1 × 10−6 (Phase 2)
    - Weight decay: 1 × 10−5
- Batch Size: 16
- Device: RTX 4060 GPU (16GB)
- 5-fold cross-validation (using GroupKFold, patient-level split)
- Class Imbalance Handling: WeightedRandomSampler
- Loss Function: BCEWithLogitsLoss
- Training Duration: 20 epochs (Phase 1); ≤80 epochs, patience=20 (Phase 2)

# Algorithm `NoduleMC` (Score: 0.7448)

## Data Processing

### Datasets
LUNA25 challenge dataset and nodule mask images (1x64x64x64 voxels) generated by MedSAM2 (Note: six samples without masks were excluded at the time they trained their model)

### Preprocessing
In addition to the baseline preprocessing, two procedures are introduced to address the relatively low computational emphasis of the I3D model along
the x axis, and mitigates the excessive downsampling of feature maps in deeper layers.
- apply flips along the y and z-axes, as well as 90° rotations, to the volumes
- upscale nodule patches from 64 × 64 × 64 to 64 × 128 × 128

## Model Architecture

Input: 3-channel (repeated) 3D patch (3x64x128x128 voxels)

Architecture: an I3D model (pretrained on Kinetics-400) as the backbone encoder, with
- a classification head: global average pooling followed by an MLP
- a segementation head: implemented as a U-Net structure that receives skip connections from the I3D features

Training Output:
- Classification head: single value
- Segementation head: 1x64x64x64 voxels

Inference Output:
- Only classification output
- Ensembling
    - 3 models trained with 3 different random seeds
    - soft voting: average the probability outputs from 3 models

## Training Recipe
- Optimizer: Adam
    - Base learning rate: 1 × 10−4
    - Momentum: β1=0.9, β2=0.999
    - Weight decay: 5 × 10−4
- Batch Size: 32 (classification loss); 16 (segmentation loss)
- Class Imbalance Handling:
    - divide data into two groups:
        - Group A: all 554 malignant samples and an equal number (554) of benign samples
        - Group B: the remaining benign samples
    - randomly select from Group A approximately 200 samples (patient-level splitting) to form the validation set
    - assign all remaining data (the rest of Group A and all of Group B) to the training set
- Loss Function: a weight sum of cross-entropy loss for classfication and dice loss for segmentation (lambda=0.5)
- Training Duration: 15 epochs
- Weight Init for Segmentation Head: He-normal
- EMA: 0.998
