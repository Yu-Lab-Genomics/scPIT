# scPIT

**Single cell based Phenotype Interpretation Transformer (scPIT)** is a deep learning model with a hybrid architecture designed to predict disease status and FEV1%pred using scRNA-seq profiles and clinical features. This README provides system requirements, installation instructions, a demo guide, and usage details for the scPIT model. Please follow the sections below to set up and run scPIT for predicting disease phenotypes using scRNA-seq data.  

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [1. System Requirements](#1-system-requirements)
- [2. Installation Guide](#2-installation-guide)
- [3. Demo](#3-demo)
- [4. Instructions for Use](#4-instructions-for-use)
- [5. Additional Reproduction Instructions](#5-additional-reproduction-instructions)
- [6. License](#6-license)

---
## Architecture Overview
scPIT is a hybrid architecture that combines Transformer encoders, multi-layer perceptrons (MLP), and single-layer perceptrons (SLP). The model processes multiple cell input tokens, where each token represents the gene expression features of a single cell. These features are projected into a 128-dimensional embedding space using a MLP. Each cell token is then embedded individually in the Cell Embedding layer, processed through a transformer Encoder, and ultimately pooled to derive a unified initial individual-embedding. Clinical features including age, sex, and BMI are projected into a 1-dimensional feature-embedding space using a SLP, merged with the initial individual-embedding to form the final individual-embedding. Based on the final individual embedding, two SLPs are utilized for regression and classification tasks, respectively, with Mean Squared Error (MSE) and Binary Cross-Entropy with Logits (BCEWithLogits) as the employed loss functions. For detailed model architecture and hyperparameters, please refer to the source code.  

---

## 1. System Requirements

- **Operating System:**  
  - CentOS Stream 9 (tested)  
- **Hardware Requirements:**  
  - NVIDIA GPU (tested on NVIDIA H800)
- **Python Version:**  
  - Python 3.13.3 (Jupyter Notebook environment)
- **CUDA Version:**  
  - CUDA 12.4
- **Python Dependencies:**
  - Listed in `requirements.txt`:

```text
scanpy==1.10.3
matplotlib==3.9.2
pandas==2.2.3
numpy==2.0.2
torch==2.5.1
captum==0.7.0
tensorboard==2.18.0
seaborn==0.13.2
umap==0.5.7
```

## 2. Installation Guide

#### Step 1: Clone the Repository
```bash
git clone https://github.com/Yu-Lab-Genomics/scPIT.git
cd scPIT
```

#### Step 2: Create Environment and Install Dependencies
```bash
conda create -n scpit_env python=3.10 -y
conda activate scpit_env
pip install -r requirements.txt
# Note: While development used Python 3.13.3, we recommend Python 3.10 for broader compatibility with current libraries.
```
## 3. Demo

### Run the Inference Pipeline

```bash
python ./04.scPIT_inference/scPIT_inference.py --gpu 5 \
--model_ckpt ./02.checkpoint/model_weights.pth \
--cell_type_ratio ./00.preprocessing/cell_type_ratios.txt \
--gene_rank_file ./00.preprocessing/GeneRanks_within_scPIT.txt \
--h5ad_file ./04.scPIT_inference/demo_input/scPIT_demo_data.h5ad \
--output_path ./04.scPIT_inference/demo_output/
```
Arguments:  
--gpu: ID of the GPU to use. Default is GPU 5.  
--model_ckpt: Path to the trained scPIT model checkpoint (already provided in ./02.checkpoint/).  
--cell_type_ratio: Cell type proportions used during training (provided in ./00.preprocessing/).  
--gene_rank_file: Ranked gene list used during training (provided in ./00.preprocessing/).  
--h5ad_file: Input .h5ad file for demo. Must contain genes listed in gene_rank_file, and .obs must include the following columns:
['sample', 'celltype', 'BMI', 'Age', 'Sex', 'FEV1%pred', 'Disease']
It is recommended that the included cell types match those in cell_type_ratio.  
--output_path: Output directory where predictions will be saved as prediction_output.tsv. 

### Expected Output
The output file prediction_output.tsv contains the following format:
```bash
sample_id   FEV1%pred_Obs.  FEV1%pred_Pred.  DiseaseStatus_Obs.  DiseaseStatus_Pred.
sample6     -0.848          -1.584           1                   1
sample10    -2.077          -2.187           1                   1
sample7     0.140           -0.401           1                   1
sample9     -0.955          -1.387           1                   1
sample4     -0.311          -0.482           1                   1
sample2     0.917           0.109            1                   0
sample3     0.106           0.413            0                   0
sample5     0.882           0.281            0                   0
sample1     1.025           0.283            0                   0
sample8     1.122           0.379            0                   0
```
Note:  
FEV1%pred_Obs. and FEV1%pred_Pred. are standardized using Z-score normalization.  
DiseaseStatus is a binary label where 0 = Healthy and 1 = COPD.  

## 4. Instructions for Use

If you are able to successfully run the demo dataset, you only need to prepare a compatible `.h5ad` file to apply scPIT on your own data. Please ensure the following requirements are met when preparing your input file:

- Your `.h5ad` file **must include all genes** listed in `./00.preprocessing/GeneRanks_within_scPIT.txt`, and the **gene names must match exactly**.
- The `.obs` of your `.h5ad` file must contain the following columns:  
   `['sample', 'celltype', 'BMI', 'Age', 'Sex', 'FEV1%pred', 'Disease']`  
   If `FEV1%pred` or `Disease` are unavailable, you can fill them with dummy values—these will **not affect inference** but are required for input formatting. Column names must match exactly.
- Ensure that the `celltype` column includes the necessary cell types as listed in `./00.preprocessing/cell_type_ratios.txt`.

Once your `.h5ad` file satisfies the above criteria, it can be directly used with the `scPIT_inference.py`, just like in the demo.

## 5. Additional Reproduction Instructions

This section outlines the steps to fully reproduce the training and interpretability workflow of scPIT, including data pre-processing, data splitting, model training, and interpretability analysis.
- Note: Before proceeding with the Additional Reproduction Instructions, please download the full .h5ad dataset used in this study.  
- If you are temporarily unable to access the full single-cell `.h5ad` dataset used in this study, you can still explore training-related metrics using TensorBoard:

```bash
tensorboard --logdir ./tensorboard/
```


### 5.1 Data Pre-Processing

The data pre-processing pipeline is implemented in the Jupyter Notebook:  
`./00.PreProcessing/Data_PreProcessing.ipynb`

The goal is to select an optimal number of highly variable genes (HVGs) and a balanced subset of single cells per donor. The steps are as follows:

**Step 1: Shared Cell Type Selection**  
To ensure that each sample contributes an equal distribution of cell types, only cell types shared across all donors are retained.

**Step 2: HVG Selection**  
The expression matrices of the shared cell types are merged across all donors and grouped by cell type. For each group, the top 500 HVGs are selected using the Seurat method. The union of all HVGs across cell types forms the input gene set.

**Step 3: Cell Sampling Per Donor**  
For each shared cell type, the median number of cells across all donors is calculated. Then, stratified random sampling is performed per donor to match these proportions.

**Step 4: Tensor Construction**  
- The number of selected cells per donor may vary, so each sample is padded with zeros to match the maximum cell count.  
- This results in a **cell tensor** with shape `[n_donors, m_cells, d_HVGs]` and an accompanying **padding mask** `[n_donors, m_cells]`.  
- Simultaneously, three additional tensors are generated:  
  - Clinical features `[n_donors, 3]`  
  - FEV1%pred values `[n_donors]`  
  - Disease labels `[n_donors]` (0 = healthy, 1 = COPD)

**Step 5: Save Processed Data**  
All tensors are saved in a dictionary format as a `.pth` file for model input.



### 5.2 Data Splitting

The dataset is split into training and validation sets using an 80/20 ratio.  
Due to the limited sample size, this splitting process is embedded directly within the training script. No manual split is required.

---

### 5.3 Model Training

Once the preprocessed `.pth` file is ready, model training can be initiated with:

```bash
python ./01.model/main.py path_to_Pre-Processing.pth
```

---

### 5.4 Interpretability Analysis
scPIT's interpretability module is based on the DeepLIFT algorithm (backpropagation-based feature attribution). The goal is to analyze which genes and clinical features contribute most to the model's decisions.  

#### Relevant Notebooks:

- ./03.interpretation/DeepLIFT.ipynb: Generates DeepLIFT feature attributions.

- ./03.interpretation/Vis_IndividualEmbed.ipynb: Visualizes patient-level embeddings and attention weights.

#### Requirements:

- A trained model checkpoint (./02.checkpoint/model_weights.pth)
- Input tensors (individual embeddings, clinical features, etc.) prepared from `5.1 Data Pre-Processing`.

## 6. License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.