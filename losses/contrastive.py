import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return self.cross_entropy(logits, labels)
