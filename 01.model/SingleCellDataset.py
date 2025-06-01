from torch.utils.data import Dataset, DataLoader, Subset
class SingleCellDataset(Dataset):
    def __init__(self, expr_tensor, expr_mask, disease_tensor, meta_tensor, celltype_tensor, target_tensor):
        self.expr_tensor = expr_tensor
        self.expr_mask = expr_mask
        self.disease_tensor = disease_tensor
        self.meta_tensor = meta_tensor
        self.celltype_tensor = celltype_tensor
        self.target_tensor = target_tensor
        
        self.expr_mean = expr_tensor.mean(dim=(0,1), keepdim=True)
        self.expr_std = expr_tensor.std(dim=(0,1), keepdim=True) + 1e-8

        self.disease_mean = disease_tensor.mean()
        self.disease_std = disease_tensor.std() + 1e-8

    def __len__(self):
        return len(self.expr_tensor)

    def __getitem__(self, idx):
        return {
            'expr': self.expr_tensor[idx],
            'mask': self.expr_mask[idx],
            'label': self.disease_tensor[idx],
            'meta': self.meta_tensor[idx],
            'celltype': self.celltype_tensor[idx],
            'target': self.target_tensor[idx]
        }

    def get_normalization_params(self):
        return {
            'expr_mean': self.expr_tensor,
            'expr_std': self.expr_std,
            'disease_mean': self.disease_mean,
            'disease_std': self.disease_std
        }