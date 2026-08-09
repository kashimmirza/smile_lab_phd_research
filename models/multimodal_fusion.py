import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, octa_feat: torch.Tensor, motion_feat: torch.Tensor, text_feat: torch.Tensor = None) -> torch.Tensor:
        features = [octa_feat, motion_feat]
        if text_feat is not None:
            features.append(text_feat)
        fused = torch.cat(features, dim=-1)
        return self.fc(fused)
