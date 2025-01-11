# scPIT
single cell based Phenotype Interpretation Transformer (scPIT) is a deep learning model with a hybrid architecture based on artificial neural networks, designed to predict disease phenotypes using scRNA-seq profile and clinical features.

## Environment and Requirements
scPIT was trained using CUDA version 12.4 on an NVIDIA H800 under CentOS Stream 9.  
Python(version 3.13.3) within a Jupyter Notebook environment.
1. scanpy==1.10.3
2. matplotlib==3.9.2
3. pandas==2.2.3
4. numpy==2.0.2
5. torch==2.5.1
6. captum==0.7.0
7. tensorboard==2.18.0
8. seaborn==0.13.2
9. umap==0.5.7
   
## Architecture Overview
scPIT is a hybrid architecture that combines Transformer encoders, multi-layer perceptrons (MLP), and single-layer perceptrons (SLP). The model processes multiple cell input tokens, where each token represents the gene expression features of a single cell. These features are projected into a 128-dimensional embedding space using a MLP. Each cell token is then embedded individually in the Cell Embedding layer, processed through a transformer Encoder, and ultimately pooled to derive a unified initial individual-embedding. Clinical features including age, sex, and BMI are projected into a 1-dimensional feature-embedding space using a SLP, merged with the initial individual-embedding to form the final individual-embedding. Based on the final individual embedding, two SLPs are utilized for regression and classification tasks, respectively, with Mean Squared Error (MSE) and Binary Cross-Entropy with Logits (BCEWithLogits) as the employed loss functions. For detailed model architecture and hyperparameters, please refer to the source code.  

## Data Pre-Processing
An optimal number of Highly Variable Genes (HVGs) and randomly selected single cells per donor are carefully chosen.  
The source code is provided in Jupyter Notebook `./00.PreProcessing/Data_PreProcessing.ipynb`.  
**In summary, the process consists of five steps:**  
**Step 1**: To ensure that each sample contributes an equal proportion of cell types to the model, we selected the shared cell types present across all samples before proceeding with subsequent data preprocessing.  
**Step 2**: For the selection of HVGs, we merged the single-cell matrices of the shared cell types from all donors, grouped them by cell type, and identified the top 500 HVGs for each cell type using the Seurat method. The union of these HVGs was then used as the input for scPIT.  
**Step 3**: To randomly select single cells from each donor, we first calculated the median number of cells per shared cell type across all donors. Subsequently, for each donor, we performed stratified random sampling based on cell type to ensure that the proportion of cells in each cell type matched the previously calculated median proportion.  
**Step 4**: Since the number of cells selected from each donor varies, we standardized the token by padding all donors with zeros to match the maximum cell count, return a **single cell profile tensor [n donors, m cells, d HVGs]** and an associated **padding mask [n donors, m cells]**. Simultaneously generate **clinical feature tensor [n donors, 3 clinical features]**, **FEV1%pred tensor [n donors]** and **disease status tensor[n donors]** with 0 for control and 1 for disease.  
**Step 5**: Finally, all prepared tensors are packaged and saved as a dictionary in a .pth file for model input.

## Data Splitting
For model training, the dataset is divided into a training set and a validation set, with a ratio of 0.8 to 0.2.  
Due to the small size of the dataset, the data splitting portion has been integrated into the model training, eliminating the need for a separate execution.

## Model Training
scPIT is trained by running `python ./01.model/main.py  path_to_Pre-Processing.pth`.   
All hyperparameters for the training process have been pre-defined in the source code, and the training requires only the file path of the pre-processed dataset as input.  

## Interpretation
The scPIT interpretability analysis utilizes the DeepLIFT method based on backpropagation, the source code is provided in Jupyter Notebook `./03.Interpretation/Vis_IndividualEmbed.ipynb.ipynb` and `./03.Interpretation/DeepLIFT.ipynb`.  
