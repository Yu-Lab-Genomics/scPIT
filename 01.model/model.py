import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, gene_count=3825, dim=128):
        super().__init__()
        # MLP 4118 dim -> 128
        self.cell_mlp = nn.Sequential(
            nn.Linear(gene_count, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=8, 
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=8,
            norm=nn.LayerNorm(dim)
        )

        # SLP -> FEV1%pred
        self.fc_out = nn.Linear(dim+1, 1) # logit output 
        # SLP -> disease status
        self.fc_class = nn.Linear(dim + 1, 1)  # logit output 
        
        # SLP -> clinical fearures embedding
        self.meta_fc = nn.Linear(3, 1)

        
    def forward(self, cell_matrix, meta, mask, mode='deeplift'):
        padding_mask = mask
        
        # return gene embedding
        batch_size, num_cells, _ = cell_matrix.shape
        cell_matrix = cell_matrix.view(-1, cell_matrix.size(-1))
        cell_embed = self.cell_mlp(cell_matrix)
        cell_embed = cell_embed.view(batch_size, num_cells, -1)
        # return cell embedding
        transformer_out = self.transformer_encoder(
            cell_embed,
            src_key_padding_mask=padding_mask
        )
        # average pooling with mask, return initial individual-embedding
        mask_expanded = ~padding_mask.unsqueeze(-1) 
        masked_out = transformer_out * mask_expanded
        sum_out = masked_out.sum(dim=1)
        count = (~padding_mask).sum(dim=1).unsqueeze(-1).float()
        pooled_output = sum_out / (count + 1e-8)

        # return featurn embedding
        meta_embed = self.meta_fc(meta) #[101,4]>[101,1]
        # featurn embed + individual embed
        combined_embed = torch.cat((meta_embed, pooled_output), dim=1)  # (batch_size, c+1, dim)

        # return logit
        fev1_pred = self.fc_out(combined_embed)
        class_pred = self.fc_class(combined_embed)

        if mode in ['training', 'inference']:
            return fev1_pred, class_pred
        elif mode == 'extract':
            return fev1_pred, class_pred, combined_embed
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from ['training', 'inference', 'extract'].")