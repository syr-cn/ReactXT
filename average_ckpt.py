import argparse
import torch

def average_checkpoints(checkpoint_paths):
    averaged_ckpt = torch.load(checkpoint_paths[-1], map_location=torch.device('cpu'))
    param_sum_dict = {}
    for key, value in averaged_ckpt['state_dict'].items():
        param_sum_dict[key] = value.clone()

    num_checkpoints = len(checkpoint_paths)
    for ckpt_path in checkpoint_paths[:-1]:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        for key, value in checkpoint['state_dict'].items():
            param_sum_dict[key] += value

    for key in param_sum_dict.keys():
        param_sum_dict[key] = param_sum_dict[key] / num_checkpoints
    averaged_ckpt['state_dict'] = param_sum_dict

    return averaged_ckpt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Averages the weights of multiple transformer model checkpoints.")
    parser.add_argument('--checkpoint_paths', nargs='+', required=True,
                        help='List of paths to the checkpoints to be averaged. Example: --checkpoint_paths path1 path2 path3')
    parser.add_argument('--output_path', type=str, required=True,)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    averaged_state_dict = average_checkpoints(args.checkpoint_paths)
    torch.save(averaged_state_dict, args.output_path)
    print(f"Averaged checkpoint saved to {args.output_path}")
