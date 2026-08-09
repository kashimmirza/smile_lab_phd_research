import torch


def compute_directional_cosine(displacement: torch.Tensor, reference: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute cosine similarity between displacement vectors and a reference direction."""
    displacement_norm = torch.norm(displacement, dim=1, keepdim=True).clamp_min(eps)
    reference_norm = torch.norm(reference, dim=1, keepdim=True).clamp_min(eps)
    cosine = torch.sum(displacement * reference, dim=1, keepdim=True) / (displacement_norm * reference_norm)
    return cosine


def average_motion_descriptor(alpha: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Average alpha values over a spatial region with optional weighting."""
    if mask is None:
        return alpha.mean(dim=[2, 3, 4])
    weighted_sum = (alpha * mask).sum(dim=[2, 3, 4])
    norm = mask.sum(dim=[2, 3, 4]).clamp_min(1e-7)
    return weighted_sum / norm


def build_radial_reference(shape: tuple, centroid: tuple) -> torch.Tensor:
    """Construct a radial reference field pointing toward the centroid."""
    _, _, depth, height, width = shape
    z = torch.linspace(-1.0, 1.0, depth, device='cpu')
    y = torch.linspace(-1.0, 1.0, height, device='cpu')
    x = torch.linspace(-1.0, 1.0, width, device='cpu')
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    ref = torch.stack((xx - centroid[2], yy - centroid[1], zz - centroid[0]), dim=0)
    return ref.unsqueeze(0)
