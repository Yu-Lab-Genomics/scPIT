import sys
import os
import argparse
import torch
import numpy as np
import scanpy as sc
sys.path.append(os.path.abspath('./01.model'))
from model import TransformerModel
from set_seed import set_seed


def load_model(checkpoint_path, gene_count=4118, dim=128, device='cpu'):
    model = TransformerModel(gene_count=gene_count, dim=dim).to(device)
    model_weight = torch.load(checkpoint_path)
    model.load_state_dict(model_weight)
    model.eval()
    return model


def load_cell_type_ratios(filepath):
    cell_type_ratios = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                cell_type, ratio = parts
                cell_type_ratios[cell_type] = float(ratio)
    return cell_type_ratios


def load_gene_ranks(filepath):
    with open(filepath, "r") as f:
        return [line.strip().split()[0] for line in f]


def validate_adata(adata, gene_list, required_columns):
    missing_genes = [gene for gene in gene_list if gene not in adata.var_names]
    if missing_genes:
        raise ValueError(f"Missing genes: {missing_genes}")

    missing_cols = [col for col in required_columns if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in adata.obs: {missing_cols}")


def subsample_cells(adata, cell_type_ratios):
    sample_cell_matrices = {}
    samples = adata.obs['sample'].unique()
    np.random.seed(123)

    for sample in samples:
        sample_data = adata[adata.obs['sample'] == sample]
        cell_type_counts = sample_data.obs['celltype'].value_counts()
        total_cells_to_draw = sample_data.shape[0]

        while True:
            target_counts = {
                ct: max(1, int(total_cells_to_draw * cell_type_ratios.get(ct, 0)))
                for ct in cell_type_ratios
            }
            if all(cell_type_counts.get(ct, 0) >= target_counts[ct] for ct in cell_type_ratios):
                break
            total_cells_to_draw -= 1

        selected_cells = []
        for ct, count in target_counts.items():
            cells = sample_data[sample_data.obs['celltype'] == ct].obs.index
            selected = np.random.choice(cells, count, replace=False) if len(cells) > count else cells
            selected_cells.extend(selected)

        sample_cell_matrices[sample] = sample_data[selected_cells, :]

    return sample_cell_matrices


def prepare_tensors(combined_adata):
    samples = combined_adata.obs['sample'].unique()
    n_samples = len(samples)
    max_cells = max(combined_adata.obs['sample'].value_counts())
    n_genes = combined_adata.shape[1]

    meta_tensor = torch.zeros((n_samples, 3), dtype=torch.float32)
    expr_tensor = torch.zeros((n_samples, max_cells, n_genes), dtype=torch.float32)
    expr_mask = torch.ones((n_samples, max_cells), dtype=torch.float32)
    celltype_tensor = torch.zeros((n_samples, max_cells), dtype=torch.long)

    fev1_values, bmi_values, age_values, sex_values, disease_values, sample_rank = ([] for _ in range(6))

    unique_cell_types = {ctype: idx for idx, ctype in enumerate(np.unique(combined_adata.obs['celltype']))}

    for i, sample in enumerate(samples):
        sample_rank.append(sample)
        sample_data = combined_adata[combined_adata.obs['sample'] == sample].X
        sample_data = torch.tensor(sample_data.toarray(), dtype=torch.float32)
        n_cells = sample_data.shape[0]
        expr_tensor[i, :n_cells, :] = sample_data
        expr_mask[i, :n_cells] = 0.0

        cell_types = combined_adata[combined_adata.obs['sample'] == sample].obs['celltype'].values
        encoded_types = torch.tensor([unique_cell_types[ct] for ct in cell_types], dtype=torch.long)
        celltype_tensor[i, :n_cells] = encoded_types

        obs = combined_adata[combined_adata.obs['sample'] == sample].obs.iloc[0]
        bmi_values.append(obs['BMI'])
        age_values.append(obs['Age'])
        sex_values.append(1 if obs['Sex'] == 'Male' else 0)
        disease_values.append(1 if obs['Disease'] == 'COPD' else 0)
        fev1_values.append(obs['FEV1%pred'])

    def zscore(vals):
        vals = np.array(vals)
        return (vals - vals.mean()) / vals.std()

    fev1_tensor = torch.tensor(zscore(fev1_values), dtype=torch.float32)
    disease_tensor = torch.tensor(disease_values, dtype=torch.float32)
    for i, (bmi, age, sex) in enumerate(zip(zscore(bmi_values), zscore(age_values), sex_values)):
        meta_tensor[i] = torch.tensor([bmi, age, sex], dtype=torch.float32)

    return expr_tensor, meta_tensor, expr_mask.bool(), fev1_tensor, disease_tensor, sample_rank


def run_predictions(model, expr_tensor, meta_tensor, expr_mask, fev1_tensor, disease_tensor, sample_rank, device, output_path):
    with open(output_path, 'w') as f:
        f.write('sample_id\tFEV1%pred_Obs.\tFEV1%pred_Pred.\tDiseaseStatus_Obs.\tDiseaseStatus_Pred.\n')
        for i in range(meta_tensor.shape[0]):
            sample_id = sample_rank[i]
            sample_expr = expr_tensor[i:i+1].to(device)
            sample_meta = meta_tensor[i:i+1].to(device)
            sample_mask = expr_mask[i:i+1].to(device)

            fev1 = fev1_tensor[i].item()
            disease = disease_tensor[i].item()

            output = model(sample_expr, sample_meta, sample_mask, mode='training')
            pred_val = output[0].cpu().item()
            prob = torch.sigmoid(output[1]).item()
            pred_class = int(prob > 0.5)

            f.write(f"{sample_id}\t{fev1:.3f}\t{pred_val:.3f}\t{int(disease)}\t{pred_class}\n")


def main(args):
    set_seed(42)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_ckpt, device=device)
    cell_type_ratios = load_cell_type_ratios(args.cell_type_ratio)
    gene_ranks = load_gene_ranks(args.gene_rank_file)

    adata = sc.read_h5ad(args.h5ad_file)
    validate_adata(adata, gene_ranks, ['sample', 'celltype', 'BMI', 'Age', 'Sex', 'FEV1%pred', 'Disease'])
    adata = adata[:, [g for g in gene_ranks if g in adata.var_names]].copy()

    sampled_data = subsample_cells(adata, cell_type_ratios)
    combined_adata = list(sampled_data.values())[0].concatenate(list(sampled_data.values())[1:], join='outer', index_unique='-')

    tensors = prepare_tensors(combined_adata)

    output_file = os.path.join(args.output_path, 'prediction_output.tsv')
    run_predictions(model, *tensors, device, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--cell_type_ratio', type=str, required=True, help='Path to cell type ratio file')
    parser.add_argument('--gene_rank_file', type=str, required=True, help='Path to gene ranks file')
    parser.add_argument('--h5ad_file', type=str, required=True, help='Path to h5ad input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to h5ad input file')
    args = parser.parse_args()
    main(args)
