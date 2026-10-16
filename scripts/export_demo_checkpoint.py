"""Export weights and architecture metadata only from the simulated AE run."""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import json
    completion = json.loads((args.run / 'completion.json').read_text())
    if completion.get('status') != 'PASS' or not completion.get('simulated'):
        raise ValueError('Only a successfully validated simulated run can be exported here')
    source = torch.load(args.run / 'autoencoder/organmnist3d_seed42/best.pt', map_location='cpu', weights_only=False)
    allowed = ['model_state', 'channels', 'latent_channels', 'block', 'input_shape', 'latent_shape', 'seed', 'best_epoch']
    exported = {key: source[key] for key in allowed}
    exported.update({'dataset': 'organmnist3d', 'simulated': True, 'not_paper_checkpoint': True,
                     'training_data': 'independent simulated binary sphere volumes, 28 cubed',
                     'purpose': 'interface and serialization validation only'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
