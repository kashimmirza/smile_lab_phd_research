import argparse
import json
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ablation experiments')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--metrics_path', type=str, required=False)
    parser.add_argument('--output_dir', type=str, required=False)
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def summarize_metrics(metrics):
    summary = {
        'training_epochs': metrics.get('training_epochs'),
        'final_loss': metrics.get('final_loss'),
        'descriptor_mean_alpha': metrics.get('descriptor_mean_alpha'),
        'ablation_results': metrics.get('ablation_results', []),
    }
    return summary


def main(args):
    config = load_config(args.config)
    output_dir = args.output_dir or config['paths'].get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = args.metrics_path or os.path.join(output_dir, 'ablation_metrics.json')
    if not os.path.exists(metrics_path):
        print(f'No metrics file found at {metrics_path}. Creating an empty ablation summary from config.')
        metrics = {
            'training_epochs': 0,
            'final_loss': None,
            'descriptor_mean_alpha': None,
            'ablation_results': config.get('ablation_matrix', []),
        }
    else:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    summary = summarize_metrics(metrics)
    summary_path = os.path.join(output_dir, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Ablation evaluation completed.')
    print(f'Summary written to: {summary_path}')
    for entry in summary['ablation_results']:
        name = entry.get('name', 'unknown')
        value = entry.get('value', None)
        description = entry.get('description', '')
        print(f'- {name}: {value} | {description}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
