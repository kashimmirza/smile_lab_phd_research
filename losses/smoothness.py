import torch


def gradient_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).mean()
    dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).mean()
    dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).mean()
    return dx + dy + dz
