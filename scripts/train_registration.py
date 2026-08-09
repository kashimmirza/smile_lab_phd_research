import argparse
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader

from data.dataset_mri import MRIDataset
from data.transforms import get_mri_transforms
from losses.ssim_3d import ssim3d
from losses.smoothness import gradient_smoothness_loss
from models.spatial_transformer import SpatialTransformer
from models.unet3d_registration import UNet3DRegistration


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D registration model')
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()


def load_config(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(model, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'registration_epoch_{epoch}.pt')
    torch.save(model.state_dict(), path)
    print(f'Saved checkpoint: {path}')


def save_training_metadata(output_dir, history, final_flow):
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    flow_path = os.path.join(output_dir, 'final_flow.pt')
    torch.save({'flow': final_flow.detach().cpu()}, flow_path)
    print(f'Saved final flow: {flow_path}')
    return history_path, flow_path


def train(config):
    dataset = MRIDataset(config['dataset']['mri_root'], transform=get_mri_transforms())
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)

    model = UNet3DRegistration(in_channels=1, base_filters=config['model']['base_filters'], out_channels=config['model']['out_channels'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    history = []
    final_flow = None

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        for sample in loader:
            image = sample['image'].to(device)
            flow = model(image)
            warped = SpatialTransformer.warp_image(image, flow)

            loss_ssim = ssim3d(warped, image)
            loss_smooth = gradient_smoothness_loss(flow)
            loss = config['loss']['ssim_weight'] * loss_ssim + config['loss']['smoothness_weight'] * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            final_flow = flow

        avg_loss = epoch_loss / len(loader)
        history.append({'epoch': epoch + 1, 'loss': avg_loss})
        print(f'Epoch {epoch + 1}/{config['training']['epochs']} loss={avg_loss:.6f}')

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, config['paths']['checkpoint_dir'], epoch + 1)

    save_checkpoint(model, config['paths']['checkpoint_dir'], 'final')
    save_training_metadata(config['paths']['output_dir'], history, final_flow)
    return history


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    train(config)
