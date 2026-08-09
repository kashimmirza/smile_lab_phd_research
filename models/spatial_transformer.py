import torch
import torch.nn.functional as F


class SpatialTransformer:
    @staticmethod
    def warp_image(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp a 3D image using a displacement field."""
        batch_size, _, depth, height, width = image.shape
        device = image.device

        grid = SpatialTransformer._create_sampling_grid(batch_size, depth, height, width, device)
        flow = flow.permute(0, 2, 3, 4, 1)
        warped_grid = grid + flow

        return F.grid_sample(image, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)

    @staticmethod
    def _create_sampling_grid(batch_size, depth, height, width, device):
        z = torch.linspace(-1.0, 1.0, depth, device=device)
        y = torch.linspace(-1.0, 1.0, height, device=device)
        x = torch.linspace(-1.0, 1.0, width, device=device)
        z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing='ij')
        grid = torch.stack((x_grid, y_grid, z_grid), dim=-1)
        return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
