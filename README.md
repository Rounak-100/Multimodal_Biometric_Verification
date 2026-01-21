# Multimodal Breast Cancer Classification
## Proposed Approach
This work proposes a multimodal biometric verification framework based on Siamese metric learning that integrates hand and iris modalities. Modality-specific Siamese networks with EfficientNet-B0 backbones are used to extract discriminative embeddings, which are projected into a normalized feature space and trained using triplet loss. For multimodal verification, hand and iris embeddings are combined through a gated feature-level fusion mechanism that adaptively balances modality contributions. The fused representation is optimized to minimize intra-class variation while maximizing inter-class separability, enabling robust and scalable unimodal and multimodal biometric verification.
## Files

- **palm-unimodal-code.ipynb**  
  Unimodal biometric verification model using hand (palm) images based on a Siamese network trained with triplet loss.

- **iris-unimodal-code.ipynb**  
  Unimodal biometric verification model using iris images based on a Siamese network trained with triplet loss.

- **proposed-multimodal-model.ipynb**  
  Proposed multimodal Siamese verification model integrating hand and iris traits using gated feature-level fusion and triplet loss.

- **proposed_model_architecture.png**  
  Visual illustration of the proposed multimodal gated Siamese architecture.
  - **siamese_model_workflow_diagram.png**  
  Workflow diagram illustrating the Siamese metric learning pipeline for biometric verification.
## First, Run

```bash
pip install -r requirements.txt
```
## Dataset
The experiments are conducted using the Breast Cancer MSI Multimodal Image Dataset.

The dataset is publicly available at:
```bash
 https://tinyurl.com/3c44m8ws
```
Update the dataset path inside each notebook before execution.

## Train
### ResNet-50 + Simple Concatenation
```bash
Open and run:
ResNet-50_and_simple_concatenation.ipynb
```
### Modality-Specific Gated Fusion Model
```bash
Open and run:
modality_specific_gated_fusion.ipynb
```
### Proposed CBAM + Gated Cross-Attention Model
```bash
Open and run:
proposed_model.ipynb
```
