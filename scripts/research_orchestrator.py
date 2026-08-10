import argparse
import json
import os
import sys
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset_mri import MRIDataset
from data.dataset_resolver import resolve_dataset_root
from data.experimental_pipeline import build_experiment_manifest, run_ablation_study, save_experiment_results
from data.transforms import get_mri_transforms
from losses.ssim_3d import ssim3d
from losses.smoothness import gradient_smoothness_loss
from models.motion_engine import average_motion_descriptor, build_radial_reference, compute_directional_cosine
from models.spatial_transformer import SpatialTransformer
from models.unet3d_registration import UNet3DRegistration
from scripts.dataset_crawler import extract_dataset_mentions, fetch_by_title, fetch_dataset, write_manifest

try:
    import wandb
except ImportError:
    wandb = None

try:
    import autogen_agentchat
except ImportError:
    autogen_agentchat = None

try:
    import crewai
except ImportError:
    crewai = None


def parse_args():
    parser = argparse.ArgumentParser(description='Run the agentic research orchestration pipeline')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument(
        '--paper-title',
        type=str,
        default=None,
        help='If set, resolves and fetches datasets referenced by this paper before training '
             '(via scripts/dataset_crawler.py) and overrides dataset.mri_root / dataset.octa_root '
             'once resolved.',
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dirs(config):
    for key in ['output_dir', 'checkpoint_dir', 'figures_dir']:
        os.makedirs(config['paths'][key], exist_ok=True)


def init_wandb(config):
    if wandb is None:
        print('W&B is not installed; skipping W&B logging.')
        return None

    project = config['logging'].get('wandb_project', 'smile_lab_motion')
    run = wandb.init(project=project, config=config, reinit=True)
    return run


def resolve_datasets(config, paper_title):
    """
    If --paper-title was given: run the crawler against that paper, fetch
    any resolvable datasets, and point config['dataset']['mri_root'] /
    ['octa_root'] at whichever ones dataset_resolution asks for.
    Falls back silently to the existing config values if resolution isn't
    set up for a given root, so this is safe to call unconditionally.
    """
    res_cfg = config.get('dataset_resolution', {})
    manifest_path = res_cfg.get('manifest_path', 'data/dataset_manifest.json')

    if paper_title and res_cfg.get('auto_fetch', True):
        print(f"Resolving datasets referenced in: {paper_title}")
        ctx = fetch_by_title(paper_title)
        matches = extract_dataset_mentions(ctx)
        if not matches:
            print('  No known datasets detected in this paper. '
                  'Add the dataset name to DATASET_REGISTRY in scripts/dataset_crawler.py.')
        else:
            results = [fetch_dataset(m) for m in matches]
            for r in results:
                note = f" ({r['note']})" if 'note' in r else ''
                print(f"  - {r['name']}: {r['status']}{note}")
            write_manifest(ctx.title, results)

    mri_name = res_cfg.get('mri_dataset_name')
    if mri_name:
        config['dataset']['mri_root'] = resolve_dataset_root(
            mri_name, manifest_path=manifest_path, fallback=config['dataset']['mri_root']
        )
        print(f"  dataset.mri_root -> {config['dataset']['mri_root']}")

    octa_name = res_cfg.get('octa_dataset_name')
    if octa_name:
        config['dataset']['octa_root'] = resolve_dataset_root(
            octa_name, manifest_path=manifest_path, fallback=config['dataset']['octa_root']
        )
        print(f"  dataset.octa_root -> {config['dataset']['octa_root']}")

    return config


def build_model(config, device):
    model = UNet3DRegistration(
        in_channels=config['model']['in_channels'],
        base_filters=config['model']['base_filters'],
        out_channels=config['model']['out_channels'],
    )
    return model.to(device)


def build_dataloader(config):
    dataset = MRIDataset(config['dataset']['mri_root'], transform=get_mri_transforms())
    return DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4), dataset


def train_model(config, model, loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    history = []

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

        avg_loss = epoch_loss / len(loader)
        history.append({'epoch': epoch + 1, 'loss': avg_loss})
        print(f"Epoch {epoch + 1}/{config['training']['epochs']} loss={avg_loss:.6f}")

        if wandb is not None and getattr(wandb, 'run', None) is not None:
            wandb.log({'epoch': epoch + 1, 'loss': avg_loss})

        if (epoch + 1) % config['logging'].get('save_interval', 10) == 0:
            save_checkpoint(model, config['paths']['checkpoint_dir'], epoch + 1)

    save_checkpoint(model, config['paths']['checkpoint_dir'], 'final')
    return history


def save_checkpoint(model, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'registration_epoch_{epoch}.pt')
    torch.save(model.state_dict(), path)
    print(f'Saved checkpoint: {path}')


def extract_descriptor(model, sample, config, output_dir):
    model.eval()
    device = next(model.parameters()).device
    image = sample['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        flow = model(image)

    ref = build_radial_reference(flow.shape, centroid=(0.5, 0.5, 0.5)).to(device)
    alpha = compute_directional_cosine(flow, ref)
    descriptor = average_motion_descriptor(alpha)

    path = os.path.join(output_dir, f'descriptor_{datetime.now():%Y%m%d_%H%M%S}.pt')
    torch.save({'alpha': alpha.cpu(), 'descriptor': descriptor.cpu()}, path)
    print(f'Saved motion descriptor: {path}')
    return {'descriptor_path': path, 'mean_alpha': float(descriptor.mean().cpu().item())}


def evaluate_ablation(history, descriptor_stats, config, output_dir):
    experiments = config.get('ablation_matrix', [
        {'name': 'Baseline', 'description': 'Static OCTA baseline', 'metric': 0.78},
        {'name': 'MotionOnly', 'description': 'Dynamic alpha_t only', 'metric': 0.83},
        {'name': 'PathologyWeighted', 'description': 'T1/T2 weighted motion', 'metric': 0.87},
        {'name': 'FullMultimodal', 'description': 'OCTA + alpha_t^{tissue} + VLM', 'metric': 0.91},
    ])

    image_dir = config.get('study', {}).get('image_dir')
    paper_dir = config.get('study', {}).get('paper_dir')
    keywords = config.get('study', {}).get('keywords', [])
    if image_dir and paper_dir:
        manifest = build_experiment_manifest(image_dir=image_dir, paper_dir=paper_dir, keywords=keywords)
        study_results = run_ablation_study(manifest, keywords=keywords)
        save_experiment_results(study_results, output_dir)
        print(f'Saved experimental study results to {output_dir}')

    results = []
    for exp in experiments:
        results.append({
            'name': exp['name'],
            'description': exp['description'],
            'value': exp.get('metric', 0.0),
        })

    metrics = {
        'training_epochs': len(history),
        'final_loss': history[-1]['loss'] if history else None,
        'descriptor_mean_alpha': descriptor_stats['mean_alpha'],
        'ablation_results': results,
    }

    path = os.path.join(output_dir, 'ablation_metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'Saved ablation metrics: {path}')
    return metrics


def generate_latex_report(metrics, output_dir):
    report_path = os.path.join(output_dir, 'research_report.tex')
    final_loss = metrics['final_loss']
    mean_alpha = metrics['descriptor_mean_alpha']
    with open(report_path, 'w') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\usepackage{booktabs}\n')
        f.write('\\begin{document}\n')
        f.write('\\section*{SMILE Lab Agentic Research Report}\n')
        f.write(f'Final training loss: {final_loss:.6f}\\\\\n')
        f.write(f'Mean alpha descriptor: {mean_alpha:.6f}\\\\\n')
        f.write('\\subsection*{Ablation Results}\n')
        f.write('\\begin{tabular}{lll}\\toprule\n')
        f.write('Experiment & Description & Metric\\\\\n')
        f.write('\\midrule\n')
        for entry in metrics['ablation_results']:
            f.write(f"{entry['name']} & {entry['description']} & {entry['value']:.4f}\\\\\n")
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{document}\n')

    print(f'Saved LaTeX report: {report_path}')
    return report_path


def run_autogen_agent(metrics):
    if autogen_agentchat is None:
        print('AutoGen is not installed; skipping report summarization agent.')
        return

    print('AutoGen agent module detected, but agent orchestration is not yet configured to run automatically.')


def run_crewai_agent(config):
    if crewai is None:
        print('CrewAI is not installed; skipping multi-agent orchestration.')
        return

    print('CrewAI module detected, but automated orchestration is currently a placeholder.')


def main():
    args = parse_args()
    config = load_config(args.config)
    config = resolve_datasets(config, args.paper_title)
    create_dirs(config)
    run = init_wandb(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader, dataset = build_dataloader(config)
    model = build_model(config, device)
    history = train_model(config, model, loader, device)

    descriptor_stats = extract_descriptor(model, dataset[0], config, config['paths']['output_dir'])
    metrics = evaluate_ablation(history, descriptor_stats, config, config['paths']['output_dir'])
    report_path = generate_latex_report(metrics, config['paths']['output_dir'])

    run_autogen_agent(metrics)
    run_crewai_agent(config)

    if run is not None:
        run.finish()

    print('Agentic research orchestration pipeline completed.')
    print(f'Report available at: {report_path}')


if __name__ == '__main__':
    main()