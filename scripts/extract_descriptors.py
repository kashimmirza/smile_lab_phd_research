import argparse
import torch

from models.motion_engine import average_motion_descriptor, build_radial_reference, compute_directional_cosine


def parse_args():
    parser = argparse.ArgumentParser(description='Extract motion descriptors from displacement fields')
    parser.add_argument('--flow_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    return parser.parse_args()


def load_flow(path):
    return torch.load(path)


def main(args):
    flow = load_flow(args.flow_path)
    ref = build_radial_reference(flow.shape, centroid=(0.5, 0.5, 0.5))
    ref = ref.to(flow.device)
    alpha = compute_directional_cosine(flow, ref)
    descriptor = average_motion_descriptor(alpha)
    torch.save({'alpha': alpha.detach().cpu(), 'descriptor': descriptor.detach().cpu()}, args.output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
