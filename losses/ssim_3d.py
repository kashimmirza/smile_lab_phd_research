import torch
import torch.nn.functional as F


def ssim3d(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    # Minimal 3D SSIM placeholder for prototype use.
    mu_x = F.avg_pool3d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool3d(y, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool3d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x * mu_x
    sigma_y = F.avg_pool3d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool3d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean() if size_average else torch.clamp((1 - ssim_map) / 2, 0, 1)
